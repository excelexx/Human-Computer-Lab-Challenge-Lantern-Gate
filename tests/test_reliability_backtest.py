"""CPU tests of instrumentation/report checks; fake text is never a model result."""
import pytest

from checkin.reliability_backtest import ObservedGenerator, cancelled, completed, snapshot, prepare


class ExampleGenerator:
    def __init__(self):
        self.closed = False

    def stream(self, *args):
        try:
            yield 'one'
            yield 'two'
        finally:
            self.closed = True


def test_observer_preserves_text_and_closes_on_disconnect():
    inner = ExampleGenerator()
    observer = ObservedGenerator(inner)
    iterator = observer.stream()
    assert next(iterator) == 'one'
    iterator.close()
    assert observer.starts == observer.closes == 1
    assert inner.closed


def test_exhaustion_probe_only_runs_after_real_iterator_exhausts():
    observer = ObservedGenerator(ExampleGenerator())
    order = []
    observer.at_exhaustion = lambda: order.append('exhausted')
    iterator = observer.stream()
    order.append(next(iterator))
    assert order == ['one']
    order.extend(iterator)
    assert order == ['one', 'two', 'exhausted']


def test_snapshot_does_not_mutate_when_stream_state_changes():
    state = {'response': {'status': 'streaming', 'text': ''}}
    saved = snapshot(state)
    state['response']['text'] = 'later'
    assert saved['response']['text'] == ''


def test_cancellation_requires_exactly_one_terminal_event():
    assert cancelled([{'type': 'cancelled'}])
    assert cancelled([{'type': 'done', 'state': {'response': {'status': 'cancelled'}}}])
    assert not cancelled([])
    assert not cancelled([{'type': 'done', 'state': {'response': {'status': 'complete'}}}])
    assert not cancelled([{'type': 'cancelled'}, {'type': 'error'}])


def test_empty_or_invalid_completion_is_not_success():
    assert not completed([])
    assert not completed([{'type': 'done', 'state': {}}])
    assert not completed([{'type': 'error', 'error': 'failure'}])


def test_close_failure_is_not_counted_as_successful_cleanup():
    class FailingClose:
        def __iter__(self): return self
        def __next__(self): return 'partial'
        def close(self): raise RuntimeError('close failed')

    class Generator:
        def stream(self): return FailingClose()

    observer = ObservedGenerator(Generator())
    stream = observer.stream()
    assert next(stream) == 'partial'
    with pytest.raises(RuntimeError, match='close failed'):
        stream.close()
    assert observer.starts == observer.close_attempts == observer.close_failures == 1
    assert observer.closes == 0


@pytest.mark.parametrize('module_name', ['reliability_backtest', 'context_backtest'])
def test_failed_cli_returns_nonzero(monkeypatch, module_name):
    import importlib
    import sys
    module = importlib.import_module('checkin.' + module_name)
    monkeypatch.setattr(sys, 'argv', ['check', '--home', 'unused', '--output', 'unused', 'run'])
    monkeypatch.setattr(module, 'run', lambda *args: {'status': 'contract_failures'})
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == 1


@pytest.mark.parametrize('rounds', [-1, 129, True, 1.1])
def test_preparation_refuses_invalid_burst_size_before_reading_artifacts(tmp_path, rounds):
    with pytest.raises(ValueError, match='rounds'):
        prepare(tmp_path / 'home', tmp_path / 'study', rounds)
