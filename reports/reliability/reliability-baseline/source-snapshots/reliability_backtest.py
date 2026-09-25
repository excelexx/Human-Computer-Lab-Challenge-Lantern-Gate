"""Declared real-model cancellation, ownership and bounded burst reliability checks.

No classifier/prompt selection or emotion-accuracy score. A transparent observer
counts real generator entries/closures and, in one labeled scheduling probe,
requests cancellation just as the real generator finishes. It supplies no text.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

import psutil
import torch

from .data import read_manifest
from .interaction_backtest import fingerprints
from .robustness import identity
from .schema import CheckInState
from .train import atomic_json, file_sha256, load_cache


CASES = ['cold_load_cancel', 'cancel_after_state', 'cancel_after_first_delta',
         'close_after_state', 'close_after_first_delta', 'wrong_session_and_busy',
         'cancel_at_generator_exhaustion', 'terminal_ownership']


class ObservedGenerator:
    """Transparent instrumentation around the real local streaming iterator."""
    def __init__(self, inner):
        self.inner = inner
        self.starts = self.closes = 0
        self.at_exhaustion = None

    def status(self):
        return self.inner.status()

    def stream(self, *args, **kwargs):
        self.starts += 1
        stream = self.inner.stream(*args, **kwargs)
        try:
            yield from stream
            if self.at_exhaustion is not None:
                self.at_exhaustion()
        finally:
            self.closes += 1
            close = getattr(stream, 'close', None)
            if callable(close):
                close()


def snapshot(value):
    return json.loads(json.dumps(value, allow_nan=False))


def terminal(events):
    return [event for event in events if event.get('type') in {'done', 'cancelled', 'error'}]


def completed(events):
    final = terminal(events)
    if len(final) != 1 or final[0].get('type') != 'done':
        return False
    try:
        state = CheckInState.model_validate(final[0]['state'])
    except (ValueError, KeyError):
        return False
    text = ''.join(e.get('text', '') for e in events if e.get('type') == 'text_delta')
    return state.response.status == 'complete' and bool(text.strip()) and state.response.text == text


def cancelled(events):
    final = terminal(events)
    return len(final) == 1 and (final[0].get('type') == 'cancelled' or
           final[0].get('state', {}).get('response', {}).get('status') == 'cancelled')


def prepare(home, output, rounds=48):
    home, output = Path(home).resolve(), Path(output).resolve()
    if isinstance(rounds, bool) or not isinstance(rounds, int) or not 0 <= rounds <= 128:
        raise ValueError('rounds must be an integer between 0 and 128')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty directory; declared reliability runs are never overwritten')
    cache = load_cache(home, 'dev')
    eligible = {str(i) for i, q in zip(cache['ids'], cache['quality']) if q[2] > 0}
    rows = [row for row in read_manifest(home, 'dev') if row['id'] in eligible][24:32]
    if len(rows) != len(CASES):
        raise ValueError('Eight eligible development examples are required')
    for row in rows:
        if file_sha256(row['video_path']) != row['media_sha256']:
            raise ValueError('Selected development media hash differs')
    plan = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
            'home': str(home), 'fingerprints': fingerprints(home, cache),
            'source_sha256': file_sha256(__file__), 'cases': CASES, 'rows': rows, 'rounds': rounds,
            'selection': 'Eligible dev rows 25–32 in manifest order; no label/prediction selection.',
            'expected_contract': ['Accepted cancellation has one terminal event',
                'Already-cancelled state does not open a generator request',
                'No new text deltas after accepted cancellation',
                'Another session cannot cancel or enter a held pipeline',
                'Closing a consumer releases owned resources',
                'Terminal events release ownership even if the consumer stops iterating',
                'Closing an old completed iterator cannot release a newer owner',
                'Every recovery and burst turn completes with matching stream/final text'],
            'burst': 'Alternate real paired dev input and no-video text in a single warmed pipeline; reset history every four turns, retain measured responses within each group. No deliberate delays or latency SLA score.',
            'instrumentation': 'ObservedGenerator forwards actual deltas unchanged. The exhaustion case injects only cancel(session) at the real iterator exhaustion boundary; it is an explicit scheduling probe.',
            'memory': 'Process RSS/handle count, PyTorch allocated/reserved VRAM, generator RSS/handle count sampled after each case and burst turn. No forced garbage collection or allocator empty_cache.',
            'limitations': ['A short burst does not establish overnight stability or absence of slow memory leaks.',
                            'Control scheduling probes are not human browser timing or webcam tests.',
                            'Response content is retained but not given a quality score.',
                            'Development data already informed models; no accuracy or independent holdout claim.']}
    plan['protocol_identity'] = identity(plan)
    atomic_json(output / 'protocol.json', plan)
    return plan


def resources(generator_pid):
    process = psutil.Process()
    value = {'monotonic_seconds': time.perf_counter(), 'python_rss_mib': process.memory_info().rss / 2**20,
             'python_handles': process.num_handles() if hasattr(process, 'num_handles') else process.num_fds(),
             'cuda_allocated_mib': torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else None,
             'cuda_reserved_mib': torch.cuda.memory_reserved() / 2**20 if torch.cuda.is_available() else None}
    if generator_pid:
        try:
            generator = psutil.Process(generator_pid)
            value.update(generator_rss_mib=generator.memory_info().rss / 2**20,
                         generator_handles=generator.num_handles() if hasattr(generator, 'num_handles') else generator.num_fds())
        except psutil.Error:
            value['generator_unavailable'] = True
    return value


def exercise_case(pipe, observer, name, row):
    """Real inference with controlled consumer actions, returning all observations."""
    session = 'reliability-' + name
    events, checks = [], {}
    starts_before = observer.starts
    before = time.perf_counter()
    def consume(stream):
        for event in stream:
            events.append(snapshot(event))
    stream = pipe.stream(row['text'], row['video_path'], [], session, '1')
    try:
        if name == 'cold_load_cancel':
            failures = []
            def worker():
                try:
                    consume(stream)
                except Exception as error:
                    failures.append(repr(error))
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            deadline = time.monotonic() + 30
            while pipe._active_session is None and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(.005)
            checks['request_preceded_finished_load'] = not pipe._loaded
            checks['cancel_accepted'] = pipe.cancel(session)
            thread.join(120)
            if thread.is_alive():
                raise RuntimeError('Cold cancellation worker did not finish within 120 seconds')
            checks.update(worker_errors_absent=not failures, one_cancel_terminal=cancelled(events),
                          generator_not_started=observer.starts == starts_before)
        else:
            events.append(snapshot(next(stream)))
            checks['first_event_is_state'] = events[0].get('type') == 'state'
            if name == 'cancel_after_state':
                checks['cancel_accepted'] = pipe.cancel(session)
                consume(stream)
                checks.update(one_cancel_terminal=cancelled(events),
                              generator_not_started=observer.starts == starts_before,
                              no_text_delta=not any(e['type'] == 'text_delta' for e in events))
            elif name in {'cancel_after_first_delta', 'close_after_first_delta'}:
                event = next(stream)
                events.append(snapshot(event))
                checks['actual_first_delta'] = event.get('type') == 'text_delta' and bool(event.get('text'))
                deltas_before = sum(e['type'] == 'text_delta' for e in events)
                if name.startswith('cancel'):
                    checks['cancel_accepted'] = pipe.cancel(session)
                    consume(stream)
                    checks.update(one_cancel_terminal=cancelled(events),
                                  no_extra_delta=sum(e['type'] == 'text_delta' for e in events) == deltas_before)
                else:
                    stream.close()
                    checks['real_generator_closed'] = observer.closes == observer.starts
            elif name == 'close_after_state':
                stream.close()
                checks['generator_not_started'] = observer.starts == starts_before
            elif name == 'wrong_session_and_busy':
                checks['wrong_session_rejected'] = pipe.cancel(session + '-other') is False
                other = pipe.stream(row['text'], None, [], session + '-other', '1')
                try:
                    next(other)
                    checks['busy_rejected'] = False
                except RuntimeError as error:
                    checks['busy_rejected'] = 'Another turn' in str(error)
                finally:
                    other.close()
                checks['original_owner_retained'] = pipe._active_session == session and pipe._lock.locked()
                consume(stream)
                checks['original_completes'] = completed(events)
            elif name == 'cancel_at_generator_exhaustion':
                accepted = []
                observer.at_exhaustion = lambda: accepted.append(pipe.cancel(session))
                consume(stream)
                checks.update(exhaustion_cancel_accepted=accepted == [True], one_cancel_terminal=cancelled(events))
            elif name == 'terminal_ownership':
                while not terminal(events):
                    events.append(snapshot(next(stream)))
                checks['completed_terminal'] = completed(events)
                checks['released_before_consumer_resumes'] = not pipe._lock.locked() and pipe._active_session is None
                if checks['released_before_consumer_resumes']:
                    newer = pipe.stream(row['text'], None, [], session + '-newer', '1')
                    newer_events = [snapshot(next(newer))]
                    stream.close()
                    checks['old_close_retains_new_owner'] = pipe._lock.locked() and pipe._active_session == session + '-newer'
                    try:
                        newer_events.extend(snapshot(event) for event in newer)
                    finally:
                        newer.close()
                    checks['new_owner_completes'] = completed(newer_events)
                else:
                    checks['old_close_retains_new_owner'] = False
                    checks['new_owner_completes'] = False
            else:
                raise ValueError('Unknown declared reliability case')
    finally:
        observer.at_exhaustion = None
        stream.close()
    checks['ownership_released'] = not pipe._lock.locked() and pipe._active_session is None
    checks['generator_entries_closed'] = observer.starts == observer.closes
    return {'case': name, 'id': row['id'], 'checks': checks, 'passed': all(checks.values()),
            'events': events, 'generator_entries': observer.starts-starts_before,
            'elapsed_ms': (time.perf_counter()-before)*1000}


def run(home, output):
    from .pipeline import CheckInPipeline
    home, output = Path(home).resolve(), Path(output).resolve()
    plan = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    declared = {key: value for key, value in plan.items() if key != 'protocol_identity'}
    if identity(declared) != plan['protocol_identity'] or file_sha256(__file__) != plan['source_sha256']:
        raise ValueError('Reliability protocol or source changed')
    if fingerprints(home, load_cache(home, 'dev')) != plan['fingerprints']:
        raise ValueError('Runtime fingerprints changed after declaration')
    if (output / 'run-started.json').exists():
        raise ValueError('This attempt already started; preserve it and declare another directory')
    for row in plan['rows']:
        if file_sha256(row['video_path']) != row['media_sha256']:
            raise ValueError('Declared clip changed')
    record = json.loads((home / 'runtime/generator.json').read_text(encoding='utf-8-sig'))
    atomic_json(output / 'run-started.json', {'started_at': datetime.now(timezone.utc).isoformat(), 'protocol_identity': plan['protocol_identity']})
    pipe = CheckInPipeline(home)
    observer = ObservedGenerator(pipe.generator)
    pipe.generator = observer
    report = {'protocol_identity': plan['protocol_identity'], 'status': 'running', 'cases': [],
              'recovery': [], 'burst': [], 'resources': [], 'response_quality_score': None}
    started = time.perf_counter()
    report['resources'].append({'phase': 'before_load', **resources(record['pid'])})
    try:
        for name, row in zip(plan['cases'], plan['rows']):
            result = exercise_case(pipe, observer, name, row)
            report['cases'].append(result)
            # Recovery is actual generation on the same pipeline, not a new instance.
            events = [snapshot(event) for event in pipe.stream(row['text'], row['video_path'], [], 'recovery-' + name, '1')]
            recovery = {'after': name, 'events': events, 'passed': completed(events) and not pipe._lock.locked()}
            report['recovery'].append(recovery)
            report['resources'].append({'phase': name, **resources(record['pid'])})
            atomic_json(output / 'report.json', report)
            print(f"reliability {name}: {'pass' if result['passed'] else 'contract_failure'}; recovery={recovery['passed']}", flush=True)
            if not recovery['passed']:
                raise RuntimeError('Pipeline did not recover; retained events describe the failure')
        history = []
        for index in range(plan['rounds']):
            if index % 4 == 0:
                history = []
            row = plan['rows'][index % len(plan['rows'])]
            video = row['video_path'] if index % 2 == 0 else None
            events = [snapshot(event) for event in pipe.stream(row['text'], video, history, f"burst-{index//4}", str(index%4+1))]
            passed = completed(events)
            report['burst'].append({'index': index, 'id': row['id'], 'video_present': bool(video),
                                    'history_messages': len(history), 'passed': passed, 'events': events})
            if not passed:
                raise RuntimeError('Burst turn failed; no hidden retry')
            history.extend([{'role': 'user', 'content': row['text']}, {'role': 'assistant', 'content': terminal(events)[0]['state']['response']['text']}])
            report['resources'].append({'phase': f'burst-{index}', **resources(record['pid'])})
            atomic_json(output / 'report.json', report)
            print(f"reliability burst {index+1}/{plan['rounds']}", flush=True)
        report['status'] = 'passed' if all(case['passed'] for case in report['cases']) else 'contract_failures'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started, completed_at=datetime.now(timezone.utc).isoformat(),
                      generator_entries=observer.starts, generator_closes=observer.closes,
                      ownership_released=not pipe._lock.locked() and pipe._active_session is None)
        atomic_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rounds', type=int, default=48)
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    report = prepare(args.home, args.output, args.rounds) if args.action == 'prepare' else run(args.home, args.output)
    print(json.dumps({key: report[key] for key in ['protocol_identity', 'status', 'elapsed_seconds'] if key in report}, indent=2))


if __name__ == '__main__':
    main()
