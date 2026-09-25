"""Local protocol fixtures verify boundaries; no network, GPU or quality scoring."""
import json

import httpx
import pytest

import checkin.generator as generator


def state():
    return {"emotion": {"label": "neutral", "source": "text_fallback", "probabilities": {"neutral": 1.}},
            "vision": {"available": False}}


def event(content=None, finish=None):
    return ('data: ' + json.dumps({"choices": [{"delta": {"content": content}, "finish_reason": finish}]}, ensure_ascii=False) + '\n\n').encode()


class TrackedStream(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            if callable(chunk):
                chunk()
            elif isinstance(chunk, Exception):
                raise chunk
            else:
                yield chunk

    def close(self):
        self.closed = True


def install_server(monkeypatch, chunks=None, token_count=None, overrides=None):
    original_client = httpx.Client
    requests, clients, streams = [], [], []
    overrides = overrides or {}
    chunks = chunks if chunks is not None else [event('A grounded reply.', 'stop')]

    def handle(request):
        payload = json.loads(request.content) if request.content else None
        requests.append((request.url.path, payload, request.extensions))
        if request.url.path in overrides:
            result = overrides[request.url.path]
            response = result(request) if callable(result) else result
            return response
        if request.url.path == '/props':
            return httpx.Response(200, json={'default_generation_settings': {'n_ctx': 4096}})
        if request.url.path == '/apply-template':
            return httpx.Response(200, json={'prompt': json.dumps(payload['messages'])})
        if request.url.path == '/tokenize':
            count = token_count(json.loads(payload['content'])) if token_count else 500
            return httpx.Response(200, json={'tokens': [1] * count})
        if request.url.path == '/v1/chat/completions':
            stream = TrackedStream(chunks)
            streams.append(stream)
            return httpx.Response(200, stream=stream)
        raise AssertionError(f'Unexpected fixture request {request.url}')

    def client_factory(**kwargs):
        client = original_client(transport=httpx.MockTransport(handle), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(generator.httpx, 'Client', client_factory)
    return requests, clients, streams


def test_stop_finishes_without_waiting_for_done_and_keeps_sampler(monkeypatch):
    def must_not_read():
        raise AssertionError('Reading continued after terminal stop')
    requests, clients, streams = install_server(monkeypatch, [event('Complete.', 'stop'), must_not_read])
    assert list(generator.LocalGenerator().stream('Hello', state(), [])) == ['Complete.']
    payload = requests[-1][1]
    assert {key: payload[key] for key in ('temperature', 'top_p', 'top_k', 'min_p', 'max_tokens')} == {
        'temperature': .5, 'top_p': .8, 'top_k': 20, 'min_p': 0., 'max_tokens': 96}
    assert payload['messages'][0]['content'] == generator.SYSTEM_PROMPT
    assert all(client.is_closed for client in clients) and all(stream.closed for stream in streams)


def test_context_eviction_preserves_whole_newest_groups_and_current_text(monkeypatch):
    history = [{'role': role, 'content': f'{index}-{role}'} for index in range(3) for role in ('user', 'assistant')]
    requests, _, _ = install_server(monkeypatch, token_count=lambda messages: {8: 5000, 6: 4200, 4: 3900}[len(messages)])
    list(generator.LocalGenerator().stream('Current words unchanged 🙂', state(), history))
    payload = requests[-1][1]
    assert payload['messages'][1:-1] == history[-2:]
    assert json.loads(payload['messages'][-1]['content'])['message'] == 'Current words unchanged 🙂'
    assert len([path for path, _, _ in requests if path == '/tokenize']) == 3
    assert all(payload['add_special'] is True and payload['parse_special'] is True for path, payload, _ in requests if path == '/tokenize')


@pytest.mark.parametrize('tokens,allowed', [(3984, True), (3985, False)])
def test_current_only_context_reserves_96_output_and_16_margin(monkeypatch, tokens, allowed):
    requests, clients, _ = install_server(monkeypatch, token_count=lambda messages: tokens)
    stream = generator.LocalGenerator().stream('Current message', state(), [])
    if allowed:
        assert list(stream)
    else:
        with pytest.raises(generator.GeneratorError, match='shorten'):
            list(stream)
        assert all(path != '/v1/chat/completions' for path, _, _ in requests)
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize('path,response', [
    ('/props', {'default_generation_settings': {'n_ctx': True}}),
    ('/props', {'default_generation_settings': {'n_ctx': 0}}),
    ('/apply-template', {'prompt': []}),
    ('/tokenize', {'tokens': [True]}),
    ('/tokenize', {'tokens': [1.5]}),
    ('/tokenize', {'tokens': [-1]}),
    ('/tokenize', {'tokens': []}),
])
def test_malformed_preflight_fails_without_generation(monkeypatch, path, response):
    requests, clients, _ = install_server(monkeypatch, overrides={path: httpx.Response(200, json=response)})
    with pytest.raises(generator.GeneratorError, match='context'):
        list(generator.LocalGenerator().stream('Hello', state(), []))
    assert all(path != '/v1/chat/completions' for path, _, _ in requests)
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize('response', [httpx.Response(500, json={'error': 'fixture'}), httpx.Response(200, content=b'not json')])
def test_preflight_http_and_json_failures_close_client(monkeypatch, response):
    requests, clients, _ = install_server(monkeypatch, overrides={'/props': response})
    with pytest.raises(generator.GeneratorError, match='context'):
        list(generator.LocalGenerator().stream('Hello', state(), []))
    assert len(requests) == 1
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize('last,match', [
    (event(None, 'length'), 'length limit'),
    (event(None, 'invented_reason'), 'unsupported completion'),
    (b'data: not-json\n\n', 'invalid streaming'),
    (b'data: [DONE]\n\n', 'without a completion reason'),
    (b'', 'ended before completion'),
    (httpx.ReadTimeout('fixture silence'), 'timed out'),
])
def test_partial_output_is_preserved_and_failure_is_not_completed(monkeypatch, last, match):
    requests, clients, streams = install_server(monkeypatch, [event('Partial text'), last])
    stream = generator.LocalGenerator().stream('Hello', state(), [])
    assert next(stream) == 'Partial text'
    with pytest.raises(generator.GeneratorError, match=match):
        next(stream)
    assert all(client.is_closed for client in clients) and all(stream.closed for stream in streams)
    assert requests[-1][2]['timeout']['read'] <= generator.READ_TIMEOUT_SECONDS


def test_consumer_close_closes_the_active_http_stream(monkeypatch):
    _, clients, streams = install_server(monkeypatch, [event('First'), event('Second', 'stop')])
    stream = generator.LocalGenerator().stream('Hello', state(), [])
    assert next(stream) == 'First'
    stream.close()
    assert all(client.is_closed for client in clients) and all(stream.closed for stream in streams)


@pytest.mark.parametrize('body', [b': heartbeat\n\n', b'data: unfinished'])
def test_deadline_covers_keepalives_and_unfinished_lines(monkeypatch, body):
    clock = [0.]
    monkeypatch.setattr(generator.time, 'monotonic', lambda: clock[0])
    def advance():
        clock[0] += 31
    _, clients, streams = install_server(monkeypatch, [advance, body, advance, body, advance, body])
    with pytest.raises(generator.GeneratorError, match='time budget'):
        list(generator.LocalGenerator().stream('Hello', state(), []))
    assert all(client.is_closed for client in clients) and all(stream.closed for stream in streams)


def test_deadline_covers_drip_fed_preflight_body(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(generator.time, 'monotonic', lambda: clock[0])
    def advance():
        clock[0] += 31
    body = TrackedStream([advance, b'{', advance, b' ', advance, b' '])
    requests, clients, _ = install_server(monkeypatch, overrides={'/props': httpx.Response(200, stream=body)})
    with pytest.raises(generator.GeneratorError, match='time budget'):
        list(generator.LocalGenerator().stream('Hello', state(), []))
    assert len(requests) == 1 and body.closed and all(client.is_closed for client in clients)


@pytest.mark.parametrize('chunks,match', [
    ([b'data: ' + b'x' * generator.MAX_SSE_LINE_BYTES], 'oversized stream line'),
    ([b'data: ' + b'x' * 40000 + b'\ndata: ' + b'y' * 40000 + b'\n\n'], 'oversized stream event'),
    ([b'x' * (generator.MAX_STREAM_BYTES + 1)], 'stream size limit'),
])
def test_stream_buffers_are_bounded(monkeypatch, chunks, match):
    _, clients, streams = install_server(monkeypatch, chunks)
    with pytest.raises(generator.GeneratorError, match=match):
        list(generator.LocalGenerator().stream('Hello', state(), []))
    assert all(client.is_closed for client in clients) and all(stream.closed for stream in streams)


def test_utf8_character_can_span_network_chunks(monkeypatch):
    data = event('Hello 🙂', 'stop')
    index = data.index('🙂'.encode()) + 2
    install_server(monkeypatch, [data[:index], data[index:]])
    assert list(generator.LocalGenerator().stream('Hello', state(), [])) == ['Hello 🙂']
