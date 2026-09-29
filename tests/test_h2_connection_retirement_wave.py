"""Upstream connection-retirement resilience over a real HTTP/2 connection.

Production incidents on this proxy show providers periodically retiring the proxy's long-lived HTTP/2 upstream connections while requests are streaming mid-response: individual streams receive RST_STREAM(INTERNAL_ERROR) and whole connections receive GOAWAY. A retirement is routine load-balancer behaviour, not an error, so the proxy must absorb it: an in-flight request whose connection is retired must be replayed on a connection that survives, and the downstream client must never see the retirement.

These tests run a real ``h2`` upstream and a real ``httpx.AsyncClient`` with HTTP/2 enabled, because the failure mode lives in the interaction between the upstream pool and the retry path, which mocks cannot express: consecutive attempts reuse pooled, co-aged connections, and a retirement wave kills the retry along with the original attempt.
"""

from __future__ import annotations

import asyncio
import datetime
import ipaddress
import pathlib
import ssl
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import h2.config
import h2.connection
import h2.errors
import h2.events
import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from headroom.proxy.server import HeadroomProxy

SSE_HEADERS = [
    (":status", "200"),
    ("content-type", "text/event-stream"),
]

SSE_FIRST_CHUNK = b'event: message_start\ndata: {"type":"message_start"}\n\n'
SSE_LAST_CHUNK = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

# How long a doomed connection waits after its first chunk before the retirement lands: long enough that the request is always mid-response, short enough to keep the suite fast.
RETIREMENT_DELAY_S = 0.05


def _tls_contexts() -> tuple[ssl.SSLContext, ssl.SSLContext]:
    """A self-signed loopback pair, so httpx negotiates HTTP/2 via ALPN.

    httpx only enables HTTP/2 over TLS (no cleartext upgrade), so the upstream must present a certificate advertising h2 and the client must trust it.
    """
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(hours=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    certificate_pem = certificate.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        (directory / "cert.pem").write_bytes(certificate_pem)
        (directory / "key.pem").write_bytes(key_pem)
        server_context.load_cert_chain(directory / "cert.pem", directory / "key.pem")
    server_context.set_alpn_protocols(["h2"])

    client_context = ssl.create_default_context()
    client_context.load_verify_locations(cadata=certificate_pem.decode("ascii"))
    return server_context, client_context


class RetirementUpstream:
    """A minimal HTTP/2 server that retires connections on a fixed policy.

    ``doom_first_n``: the first N connections opened are retired mid-response (RST_STREAM on the in-flight stream, then GOAWAY and close), matching the observed upstream signature; connections opened after that serve to completion.

    ``retire_all_at_streams``: once that many streams are in flight across all connections, every currently-open connection is retired simultaneously (a retirement wave), once; connections opened afterwards serve normally.
    """

    def __init__(self, doom_first_n: int = 0, retire_all_at_streams: int | None = None) -> None:
        self.doom_first_n = doom_first_n
        self.retire_all_at_streams = retire_all_at_streams
        self.connections_opened = 0
        self.retirements = 0
        self.wave_fired = False
        self.in_flight = 0
        self._server: asyncio.AbstractServer | None = None
        self.port: int | None = None

    async def start(self, tls: ssl.SSLContext) -> None:
        self._server = await asyncio.start_server(self._handle, host="127.0.0.1", port=0, ssl=tls)
        assert self._server.sockets
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self) -> str:
        assert self.port is not None
        return f"https://127.0.0.1:{self.port}/v1/messages"

    async def _maybe_wave(
        self, conns: list[tuple[h2.connection.H2Connection, asyncio.StreamWriter]]
    ) -> bool:
        if (
            self.retire_all_at_streams is not None
            and not self.wave_fired
            and self.in_flight >= self.retire_all_at_streams
        ):
            self.wave_fired = True
            for conn, writer in conns:
                for stream_id in conn.streams:
                    if stream_id % 2 == 1:
                        try:
                            conn.reset_stream(
                                stream_id, error_code=h2.errors.ErrorCodes.INTERNAL_ERROR
                            )
                        except h2.exceptions.StreamIDTooLowError:
                            pass
                conn.close_connection(error_code=h2.errors.ErrorCodes.INTERNAL_ERROR)
                writer.write(conn.data_to_send())
                await writer.drain()
            self.retirements += 1
            return True
        return False

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections_opened += 1
        connection_index = self.connections_opened
        doomed = connection_index <= self.doom_first_n

        config = h2.config.H2Configuration(client_side=False, header_encoding="utf-8")
        conn = h2.connection.H2Connection(config=config)
        conn.initiate_connection()
        writer.write(conn.data_to_send())
        await writer.drain()

        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    return
                try:
                    events = conn.receive_data(data)
                except h2.exceptions.ProtocolError:
                    return

                for event in events:
                    if isinstance(event, h2.events.DataReceived):
                        conn.acknowledge_received_data(
                            event.flow_controlled_length, event.stream_id
                        )
                    elif isinstance(event, h2.events.RequestReceived):
                        asyncio.ensure_future(
                            self._serve_stream(conn, writer, event.stream_id, doomed)
                        )
                    elif isinstance(event, h2.events.ConnectionTerminated):
                        return

                outgoing = conn.data_to_send()
                if outgoing:
                    writer.write(outgoing)
                    await writer.drain()
        finally:
            self.in_flight = max(0, self.in_flight)
            writer.close()

    async def _serve_stream(
        self,
        conn: h2.connection.H2Connection,
        writer: asyncio.StreamWriter,
        stream_id: int,
        doomed: bool,
    ) -> None:
        self.in_flight += 1
        try:
            conn.send_headers(stream_id, SSE_HEADERS)
            conn.send_data(stream_id, SSE_FIRST_CHUNK)
            outgoing = conn.data_to_send()
            if outgoing:
                writer.write(outgoing)
                await writer.drain()
            await asyncio.sleep(RETIREMENT_DELAY_S)
            if doomed or await self._maybe_wave([(conn, writer)]):
                # Retirement signature from production: reset the in-flight stream, GOAWAY the connection, close the socket.
                try:
                    conn.reset_stream(stream_id, error_code=h2.errors.ErrorCodes.INTERNAL_ERROR)
                except h2.exceptions.StreamIDTooLowError:
                    pass
                conn.close_connection(error_code=h2.errors.ErrorCodes.INTERNAL_ERROR)
                outgoing = conn.data_to_send()
                if outgoing:
                    writer.write(outgoing)
                    await writer.drain()
                self.retirements += 1
                return
            conn.send_data(stream_id, SSE_LAST_CHUNK, end_stream=True)
            outgoing = conn.data_to_send()
            if outgoing:
                writer.write(outgoing)
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError, h2.exceptions.ProtocolError):
            pass
        finally:
            self.in_flight -= 1


def _proxy_with_real_client(upstream_url: str, tls: ssl.SSLContext) -> HeadroomProxy:
    proxy = object.__new__(HeadroomProxy)
    proxy.http_client = httpx.AsyncClient(http2=True, verify=tls, timeout=httpx.Timeout(30.0))
    proxy._config = MagicMock()
    proxy._config.memory_enabled = False
    proxy._config.ccr_inject_tool = False
    proxy._config.retry_enabled = True
    proxy._config.retry_max_attempts = 3
    proxy._config.retry_base_delay_ms = 0
    proxy._config.retry_max_delay_ms = 0
    proxy.config = proxy._config
    proxy.memory_handler = None
    proxy.metrics = MagicMock()
    proxy._parse_sse_usage_from_buffer = MagicMock(return_value=None)
    proxy._finalize_stream_response = AsyncMock(return_value=None)
    return proxy


async def _run_stream(proxy: HeadroomProxy, url: str, session_key: str = "k") -> Any:
    return await proxy._stream_response(
        url=url,
        headers={"x-api-key": "sk-test"},
        body={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        request_id="retirement-wave-test",
        original_tokens=10,
        optimized_tokens=10,
        tokens_saved=0,
        transforms_applied=[],
        tags={},
        optimization_latency=0.0,
        session_key=session_key,
    )


async def _body(result: Any) -> bytes:
    return b"".join([chunk async for chunk in result.body_iterator])


@pytest.mark.asyncio
async def test_single_connection_retirement_is_absorbed():
    server_tls, client_tls = _tls_contexts()
    upstream = RetirementUpstream(doom_first_n=1)
    await upstream.start(server_tls)
    try:
        proxy = _proxy_with_real_client(upstream.url, client_tls)
        result = await _run_stream(proxy, upstream.url)
        body = await _body(result)
        await proxy.http_client.aclose()

        assert upstream.retirements == 1
        assert b"message_start" in body
        assert b"message_stop" in body
    finally:
        await upstream.stop()


@pytest.mark.asyncio
async def test_consecutive_connection_retirements_are_absorbed():
    """The production failure: a retirement wave kills the retry too.

    The upstream retires its first two connections mid-response, so the original attempt dies on one connection and the first replay dies on a second co-aged connection, exactly as observed against live providers where pool connections created together are retired together. A client must still receive the complete stream.
    """
    upstream = RetirementUpstream(doom_first_n=2)
    server_tls, client_tls = _tls_contexts()
    await upstream.start(server_tls)
    try:
        proxy = _proxy_with_real_client(upstream.url, client_tls)
        result = await _run_stream(proxy, upstream.url)
        body = await _body(result)
        await proxy.http_client.aclose()

        assert upstream.retirements == 2
        assert b"message_start" in body
        assert b"message_stop" in body
    finally:
        await upstream.stop()


@pytest.mark.asyncio
async def test_simultaneous_retirement_wave_across_concurrent_requests():
    """One wave retires every open connection while several requests stream.

    Every request is mid-response when the wave lands, so every one of them must be replayed and complete; none may surface the retirement.
    """
    upstream = RetirementUpstream(retire_all_at_streams=4)
    server_tls, client_tls = _tls_contexts()
    await upstream.start(server_tls)
    try:
        proxy = _proxy_with_real_client(upstream.url, client_tls)

        async def one(i: int) -> tuple[int, int, bytes]:
            result = await _run_stream(proxy, upstream.url, session_key=f"s{i}")
            body = await _body(result)
            return i, result.status_code, body

        outcomes = await asyncio.gather(*(one(i) for i in range(4)))
        await proxy.http_client.aclose()

        assert upstream.wave_fired
        for i, status, body in outcomes:
            assert status == 200, f"request {i} surfaced the retirement: {status}"
            assert b"message_start" in body, f"request {i} got a truncated stream"
            assert b"message_stop" in body, f"request {i} got a truncated stream"
    finally:
        await upstream.stop()
