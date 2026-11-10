"""Verify baseline predictions and split using only the lean handoff inputs."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    out = ROOT / 'results/model1_baseline_v1'
    metrics = json.loads((out / 'metrics.json').read_text())
    split = json.loads((ROOT / 'data/splits/split_config.json').read_text())
    for key, relative in [('train_csv', 'data/raw/train.csv'),
                          ('split_config', 'data/splits/split_config.json'),
                          ('training_script', 'src/train_baseline.py'),
                          ('dependency_lock', 'requirements.lock.txt')]:
        assert sha(ROOT / relative) == metrics['source_hashes'][key], relative
    for name, record in split['files'].items():
        assert sha(ROOT / 'data/splits' / name) == record['sha256'], name
    source = pd.read_csv(ROOT / 'data/raw/train.csv', dtype={'id': str})
    expected_train, expected_val = train_test_split(source, test_size=0.2, random_state=42, stratify=source.label)
    model = joblib.load(ROOT / metrics['model_path'])
    assert model.classes_.tolist() == [0, 1]
    checks = {}
    for name, filename, expected in [('train', 'train_8000.csv', expected_train),
                                      ('validation', 'validation_2000.csv', expected_val)]:
        frame = pd.read_csv(ROOT / 'data/splits' / filename, dtype={'id': str})
        assert frame.equals(expected.reset_index(drop=True))
        saved = pd.read_csv(out / f'{name}_predictions.csv', dtype={'id': str})
        assert frame.id.tolist() == saved.id.tolist()
        assert frame.label.tolist() == saved.true_label.tolist()
        with threadpool_limits(limits=1):
            probabilities = model.predict_proba(frame.text)[:, 1]
        assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
        assert np.allclose(probabilities, saved.p_ai, rtol=0, atol=1e-12)
        positive = frame.label.to_numpy() == 1
        n1 = positive.sum()
        n0 = (~positive).sum()
        auc = (rankdata(saved.p_ai)[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        assert abs(float(auc) - metrics[name]['auroc']) < 1e-12
        labels = (probabilities >= 0.5).astype(int)
        assert np.array_equal(labels, saved.predicted_label_at_0_5)
        assert abs(accuracy_score(frame.label, labels) - metrics[name]['accuracy_at_0_5']) < 1e-12
        assert abs(f1_score(frame.label, labels) - metrics[name]['f1_ai_at_0_5']) < 1e-12
        assert confusion_matrix(frame.label, labels, labels=[0, 1]).tolist() == metrics[name]['confusion_matrix_true_rows_predicted_columns']
        checks[name] = {'ordered_split_matches': True, 'predictions_match': True, 'independent_auroc': float(auc)}
    print(json.dumps({'status': 'passed', 'reference_files_unchanged': True, 'checks': checks}, indent=2))

if __name__ == '__main__':
    main()
