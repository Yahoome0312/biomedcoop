import json
import sys
from types import SimpleNamespace

import pytest
from scripts import evaluate_three_methods as evaluation


@pytest.mark.parametrize('metadata', [
    {'ENABLED': True, 'WEIGHT': 1., 'TEMPERATURE': .2},
    {'ENABLED': False, 'WEIGHT': 1., 'TEMPERATURE': .35},
    None,
])
def test_evaluator_records_checkpoint_parameters_instead_of_current_defaults(tmp_path, monkeypatch, metadata):
    (tmp_path/'best_validation_accuracy.json').write_text(json.dumps({'epoch': 10, 'selection_value': 60.}))
    checkpoint = {} if metadata is None else {'semantic_distill': metadata}
    trainer = SimpleNamespace(load_model=lambda path: checkpoint,
                              test=lambda split: None, last_eval_results={'accuracy': 61.})
    monkeypatch.setattr(evaluation, 'build_trainer', lambda cfg: trainer)
    monkeypatch.setattr(evaluation, 'set_random_seed', lambda seed: None)
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--run-dir', str(tmp_path),
                        '--method', 'semantic_distill', '--dataset', 'DermaMNIST',
                        '--shots', '4', '--seed', '1'])
    evaluation.main()
    result = json.loads((tmp_path/'test_metrics.json').read_text())
    assert result['test_accuracy'] == 61.
    expected = metadata or {}
    assert result['semantic_enabled'] == expected.get('ENABLED')
    assert result['semantic_weight'] == expected.get('WEIGHT')
    assert result['semantic_temperature'] == expected.get('TEMPERATURE')
    assert result['semantic_metadata_source'] == ('checkpoint' if metadata else 'unavailable')
