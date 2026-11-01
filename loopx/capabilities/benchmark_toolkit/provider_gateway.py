"""Runner-owned HTTP gateway for benchmark model-provider calls.

The gateway keeps the upstream bearer credential in the runner process and
exposes only a loopback Responses API endpoint to an isolated agent runtime.
It deliberately is not an agent sandbox: callers must place the app-server in
an OS authority boundary that cannot inspect the runner process or its files.
"""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import SplitResult, urlsplit


@dataclass(frozen=True)
class RunnerOwnedProviderGateway:
    """Public-safe handle for one bounded loopback gateway."""

    base_url: str
    allowed_paths: tuple[str, ...]


_DEFAULT_ALLOWED_PATHS = ("/responses", "/responses/compact")
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_CLIENT_CREDENTIAL_HEADERS = {
    "api-key",
    "authorization",
    "cookie",
    "proxy-authorization",
    "x-api-key",
}


def _validated_upstream(value: str) -> SplitResult:
    parsed = urlsplit(value.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("upstream_base_url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("upstream_base_url must not contain credentials or query data")
    if parsed.scheme == "http" and parsed.hostname not in {
        "127.0.0.1",
        "::1",
        "localhost",
    }:
        raise ValueError("non-loopback upstream_base_url must use HTTPS")
    return parsed


def _validated_allowed_paths(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("allowed_paths must be a sequence")
    normalized = tuple(dict.fromkeys(str(value).strip() for value in values))
    if not normalized or any(
        not value.startswith("/")
        or "?" in value
        or "#" in value
        or ".." in value.split("/")
        for value in normalized
    ):
        raise ValueError("allowed_paths must contain safe absolute URL paths")
    return normalized


def _upstream_connection(
    parsed: SplitResult, *, timeout_sec: float
) -> http.client.HTTPConnection:
    host = parsed.hostname
    if host is None:  # Defensive narrowing; _validated_upstream rejects this.
        raise ValueError("upstream_base_url must include a host")
    port = parsed.port
    if parsed.scheme == "https":
        return http.client.HTTPSConnection(host, port, timeout=timeout_sec)
    return http.client.HTTPConnection(host, port, timeout=timeout_sec)


@contextmanager
def serve_runner_owned_provider_gateway(
    *,
    upstream_base_url: str,
    upstream_bearer_token: str,
    allowed_paths: Sequence[str] = _DEFAULT_ALLOWED_PATHS,
    request_timeout_sec: float = 600,
    max_request_bytes: int = 64 * 1024 * 1024,
) -> Iterator[RunnerOwnedProviderGateway]:
    """Serve a loopback-only Responses API gateway for one benchmark run.

    The returned URL and request surface contain no upstream credential. The
    caller owns the stronger process/filesystem boundary between this gateway
    and the agent runtime.
    """

    upstream = _validated_upstream(upstream_base_url)
    token = str(upstream_bearer_token)
    if not token.strip():
        raise ValueError("upstream_bearer_token must be non-empty")
    if request_timeout_sec <= 0 or max_request_bytes <= 0:
        raise ValueError("gateway limits must be positive")
    admitted_paths = _validated_allowed_paths(allowed_paths)

    class GatewayHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, _format: str, *args: object) -> None:
            return

        def _json_error(self, status: int, reason: str) -> None:
            body = json.dumps(
                {"error": {"type": "loopx_provider_gateway", "code": reason}},
                sort_keys=True,
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            requested = urlsplit(self.path)
            if requested.path not in admitted_paths:
                self._json_error(404, "provider_gateway_path_not_admitted")
                return
            raw_length = self.headers.get("Content-Length")
            try:
                content_length = int(raw_length or "")
            except ValueError:
                content_length = -1
            if content_length < 0 or content_length > max_request_bytes:
                self._json_error(413, "provider_gateway_request_size_invalid")
                return
            body = self.rfile.read(content_length)
            upstream_path = f"{upstream.path.rstrip('/')}{requested.path}"
            if requested.query:
                upstream_path = f"{upstream_path}?{requested.query}"
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower()
                not in {
                    *_HOP_BY_HOP_HEADERS,
                    *_CLIENT_CREDENTIAL_HEADERS,
                    "content-length",
                    "host",
                }
            }
            headers["Authorization"] = f"Bearer {token}"
            headers["Content-Length"] = str(len(body))
            connection = _upstream_connection(upstream, timeout_sec=request_timeout_sec)
            response_started = False
            try:
                connection.request("POST", upstream_path, body=body, headers=headers)
                response = connection.getresponse()
                self.send_response(response.status, response.reason)
                for key, value in response.getheaders():
                    if key.lower() not in {
                        *_HOP_BY_HOP_HEADERS,
                        "content-length",
                    }:
                        self.send_header(key, value)
                self.end_headers()
                response_started = True
                # Forward available stream bytes without waiting to fill the buffer.
                while chunk := response.read1(64 * 1024):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (OSError, http.client.HTTPException):
                if not response_started and not self.wfile.closed:
                    self._json_error(502, "provider_gateway_upstream_unavailable")
                else:
                    self.close_connection = True
            finally:
                connection.close()

        def do_GET(self) -> None:
            self._json_error(405, "provider_gateway_method_not_admitted")

    server = ThreadingHTTPServer(("127.0.0.1", 0), GatewayHandler)
    server.daemon_threads = True
    thread = threading.Thread(
        target=server.serve_forever,
        name="loopx-provider-gateway",
        daemon=True,
    )
    thread.start()
    try:
        host, port = server.server_address[:2]
        rendered_host = host.decode("ascii") if isinstance(host, bytes) else str(host)
        yield RunnerOwnedProviderGateway(
            base_url=f"http://{rendered_host}:{port}",
            allowed_paths=admitted_paths,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


__all__ = [
    "RunnerOwnedProviderGateway",
    "serve_runner_owned_provider_gateway",
]
