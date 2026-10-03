from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import httpx

from headroom.proxy.handlers.openai import OpenAIHandlerMixin

EVENTS_PATH = "/v1/code/sessions/cse_example/worker/events/stream"
CHUNKS = [b"event: ping\n\n", b"event: message\ndata: {}\n\n"]


class _Request:
    def __init__(self, accept: str, path: str = EVENTS_PATH) -> None:
        self.method = "GET"
        self.headers = {"accept": accept}
        self.url = SimpleNamespace(path=path, query="")

    async def body(self) -> bytes:
        return b""


class _UpstreamResponse:
    status_code = 200
    headers = httpx.Headers({"content-type": "text/event-stream"})

    def __init__(self) -> None:
        self.closed = False

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for chunk in CHUNKS:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class _StreamingClient:
    """An upstream client whose buffered `request` must never be used for an event stream."""

    def __init__(self) -> None:
        self.upstream = _UpstreamResponse()

    def build_request(self, method: str, url: str, **kwargs: Any) -> httpx.Request:
        return httpx.Request(method, url)

    async def send(self, request: httpx.Request, *, stream: bool) -> _UpstreamResponse:
        assert stream
        return self.upstream

    async def request(self, **kwargs: Any) -> httpx.Response:
        raise AssertionError("an event stream must not be buffered through request()")


def _handler() -> tuple[OpenAIHandlerMixin, _StreamingClient, list[Any]]:
    handler = object.__new__(OpenAIHandlerMixin)
    client = _StreamingClient()
    outcomes: list[Any] = []

    async def next_request_id() -> str:
        return "req_events"

    async def record(outcome: Any) -> None:
        outcomes.append(outcome)

    handler.http_client = client
    handler._next_request_id = next_request_id
    handler._record_request_outcome = record
    return handler, client, outcomes


def test_event_stream_request_is_relayed_chunk_by_chunk_and_closes_upstream() -> None:
    handler, client, outcomes = _handler()

    async def run() -> list[bytes]:
        response = await handler.handle_passthrough(
            _Request("text/event-stream"), "https://api.example.test"
        )
        assert not client.upstream.closed
        received = [chunk async for chunk in response.body_iterator]
        return received

    assert asyncio.run(run()) == CHUNKS
    assert client.upstream.closed
    assert outcomes == []


def test_abandoned_event_stream_releases_the_upstream_connection() -> None:
    handler, client, _ = _handler()

    async def run() -> None:
        response = await handler.handle_passthrough(
            _Request("text/event-stream"), "https://api.example.test"
        )
        iterator = response.body_iterator
        assert await iterator.__anext__() == CHUNKS[0]
        await iterator.aclose()

    asyncio.run(run())
    assert client.upstream.closed


def test_accept_header_match_is_case_insensitive_and_tolerates_parameters() -> None:
    handler, client, _ = _handler()

    async def run() -> list[bytes]:
        response = await handler.handle_passthrough(
            _Request("Text/Event-Stream; q=0.9, */*;q=0.1"), "https://api.example.test"
        )
        return [chunk async for chunk in response.body_iterator]

    assert asyncio.run(run()) == CHUNKS
    assert client.upstream.closed


def test_non_event_stream_request_keeps_the_buffered_path() -> None:
    handler, client, _ = _handler()

    async def run() -> None:
        await handler.handle_passthrough(_Request("application/json"), "https://api.example.test")

    try:
        asyncio.run(run())
    except AssertionError as error:
        assert "must not be buffered" in str(error)
    else:
        raise AssertionError("the buffered path was not taken")
    assert not client.upstream.closed
