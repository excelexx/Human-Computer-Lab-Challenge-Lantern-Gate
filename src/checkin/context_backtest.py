"""Actual local context-boundary checks with transparent HTTP observation.

Authored stress inputs are not an emotion/response-quality benchmark. No fake
server responses, model output, prompt changes, sampler override or retries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import httpx

from .interaction_backtest import fingerprints
from .reliability_backtest import completed, snapshot, terminal
from .robustness import identity
from .schema import CheckInState
from .train import atomic_json, file_sha256, load_cache


def cases():
    history = []
    for index in range(3):
        history.extend([{'role': 'user', 'content': str(index) + '🙂' * 449},
                        {'role': 'assistant', 'content': str(index) + '🙂' * 449}])
    return [
        {'name': 'short_control', 'text': 'My day was quiet.', 'history': [], 'expected': 'complete'},
        {'name': 'emoji_current_overflow', 'text': '🙂' * 4000, 'history': [], 'expected': 'reject_before_generation'},
        {'name': 'cjk_current_overflow', 'text': '你' * 4000, 'history': [], 'expected': 'reject_before_generation'},
        {'name': 'whole_history_eviction', 'text': '🙂' * 2000, 'history': history, 'expected': 'fit_after_eviction'},
    ]


def prepare(home, output):
    home, output = Path(home).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty directory; boundary attempts are never overwritten')
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'cases': cases(),
            'fingerprints': fingerprints(home, load_cache(home, 'dev')),
            'source_sha256': file_sha256(__file__),
            'reliability_helper_sha256': file_sha256(Path(__file__).with_name('reliability_backtest.py')),
            'scope': 'Four authored context stress cases through the actual pipeline and local server. No accuracy or clinical/response-quality claim.',
            'checks': ['Short control completes', 'Oversize current input retains actual emotion state and fails before chat POST',
                       'History eviction removes oldest whole user/assistant groups only',
                       'Current JSON message remains exact', 'Submitted prompt plus max response plus16 fits active context',
                       'Single terminal event and pipeline ownership released'],
            'observer': 'An httpx request hook records real payloads without reading responses. After inference, an independent local props/tokenize recount verifies the exact last submitted prompt. No response interception or supplied output.',
            'limitations': ['Repeated emoji/CJK are artificial tokenizer stress, not representative check-ins.',
                            'The history stress output is retained without quality scoring; an explicit token-limit error is acceptable if context fitting succeeds.']}
    plan['protocol_identity'] = identity(plan)
    atomic_json(output / 'protocol.json', plan)
    return plan


def run(home, output):
    from .pipeline import CheckInPipeline
    home, output = Path(home).resolve(), Path(output).resolve()
    plan = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    declared = {k: v for k, v in plan.items() if k != 'protocol_identity'}
    if identity(declared) != plan['protocol_identity'] or file_sha256(__file__) != plan['source_sha256']:
        raise ValueError('Declared source or protocol changed')
    if file_sha256(Path(__file__).with_name('reliability_backtest.py')) != plan['reliability_helper_sha256']:
        raise ValueError('Reliability event-checking helper changed after declaration')
    if fingerprints(home, load_cache(home, 'dev')) != plan['fingerprints']:
        raise ValueError('Runtime changed after declaration')
    if (output / 'run-started.json').exists():
        raise ValueError('This attempt already started; declare a new output directory')
    atomic_json(output / 'run-started.json', {'started_at': datetime.now(timezone.utc).isoformat()})
    observed = []
    original = httpx.Client

    def request_hook(request):
        observed.append({'path': request.url.path, 'method': request.method,
                         'payload': json.loads(request.content) if request.content else None})

    class ObservedClient(original):
        def __init__(self, *args, **kwargs):
            if 'event_hooks' in kwargs:
                raise ValueError('Observer must not replace existing hooks')
            super().__init__(*args, event_hooks={'request': [request_hook]}, **kwargs)

    pipe = CheckInPipeline(home)
    report = {'protocol_identity': plan['protocol_identity'], 'cases': [], 'status': 'running'}
    started = time.perf_counter()
    httpx.Client = ObservedClient
    try:
        for case in plan['cases']:
            observed.clear()
            events = [snapshot(e) for e in pipe.stream(case['text'], None, case['history'], case['name'], '1')]
            final = terminal(events)
            chats = [r for r in observed if r['path'] == '/v1/chat/completions']
            checks = {'one_terminal': len(final) == 1, 'actual_state_precedes_terminal': bool(events) and events[0]['type'] == 'state',
                      'ownership_released': not pipe._lock.locked() and pipe._active_session is None}
            states = [e['state'] for e in events if isinstance(e.get('state'), dict)]
            state = states[-1] if states else {}
            status = state.get('response', {}).get('status')
            try:
                CheckInState.model_validate(state)
                checks['valid_actual_state'] = True
            except ValueError:
                checks['valid_actual_state'] = False
            if case['expected'] == 'reject_before_generation':
                checks.update(no_chat_request=not chats, error_state=status == 'error',
                              actionable_error='shorten' in json.dumps(final).lower())
            else:
                checks['one_chat_request'] = len(chats) == 1
                if case['expected'] == 'complete':
                    checks['completed'] = completed(events)
                else:
                    checks['generation_completed_or_explicit_length_limit'] = (
                        completed(events) or status == 'error' and
                        'length limit' in state.get('response', {}).get('error', '').lower())
                streamed = ''.join(e.get('text', '') for e in events if e['type'] == 'text_delta')
                checks['actual_text_preserved'] = bool(streamed.strip()) and streamed == state.get('response', {}).get('text')
                if chats:
                    messages = chats[0]['payload']['messages']
                    checks['current_message_exact'] = json.loads(messages[-1]['content'])['message'] == case['text']
                    token_payload = [r['payload'] for r in observed if r['path'] == '/tokenize'][-1]
                    with original(trust_env=False, follow_redirects=False, timeout=10) as verification:
                        properties_reply = verification.get('http://127.0.0.1:8081/props')
                        properties_reply.raise_for_status()
                        token_reply = verification.post('http://127.0.0.1:8081/tokenize', json=token_payload)
                        token_reply.raise_for_status()
                    properties = properties_reply.json()
                    count = len(token_reply.json()['tokens'])
                    context = properties['default_generation_settings']['n_ctx']
                    observed.append({'independent_recount': {'context': context, 'prompt_tokens': count}})
                    checks['fits_context'] = count + chats[0]['payload']['max_tokens'] + 16 <= context
                    kept = messages[1:-1]
                    if case['expected'] == 'fit_after_eviction':
                        checks['history_evicted'] = len(kept) < len(case['history'])
                        checks['whole_newest_groups_exact'] = len(kept) % 2 == 0 and (not kept or kept == case['history'][-len(kept):])
            report['cases'].append({'name': case['name'], 'checks': checks, 'passed': all(checks.values()),
                                    'events': events, 'requests': snapshot(observed)})
            atomic_json(output / 'report.json', report)
            print(case['name'], checks, flush=True)
        report['status'] = 'passed' if all(c['passed'] for c in report['cases']) else 'contract_failures'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', partial_requests=snapshot(observed))
    finally:
        httpx.Client = original
        report.update(elapsed_seconds=time.perf_counter()-started, completed_at=datetime.now(timezone.utc).isoformat())
        atomic_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    result = (prepare if args.action == 'prepare' else run)(args.home, args.output)
    print(json.dumps({k: result[k] for k in ('protocol_identity', 'status', 'elapsed_seconds') if k in result}, indent=2))
    if args.action == 'run' and result.get('status') != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
