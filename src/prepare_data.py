"""Restore local-only data and reference predictions from official train.csv."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    source = ROOT / 'data/raw/train.csv'
    if not source.is_file():
        raise SystemExit('Download official train.csv from the competition and place it at data/raw/train.csv. See README.md.')
    reference = json.loads((ROOT / 'REFERENCE_MANIFEST.json').read_text())
    expected = {item['path']: item['sha256'] for item in reference['files']}
    assert sha(source.read_bytes()) == expected['data/raw/train.csv'], 'Official train.csv checksum mismatch.'
    raw = pd.read_csv(source, dtype={'id': str})
    train, validation = train_test_split(raw, test_size=0.2, random_state=42, stratify=raw.label)
    frames = {'train_8000.csv': train, 'validation_2000.csv': validation,
              'train_ids.csv': train[['id', 'label']], 'validation_ids.csv': validation[['id', 'label']]}
    payloads = {}
    for name, frame in frames.items():
        relative = f'data/splits/{name}'
        data = frame.to_csv(index=False, lineterminator='\n').encode()
        assert sha(data) == expected[relative], f'Generated split differs: {relative}; use locked dependencies.'
        payloads[relative] = data
    model_path = ROOT / 'models/model1_baseline_v1.joblib'
    assert sha(model_path.read_bytes()) == expected['models/model1_baseline_v1.joblib']
    model = joblib.load(model_path)
    assert model.classes_.tolist() == [0, 1]
    for name, frame in [('train', train), ('validation', validation)]:
        with threadpool_limits(limits=1):
            probability = model.predict_proba(frame.text)[:, 1]
        assert np.isfinite(probability).all()
        predicted = pd.DataFrame({'id': frame.id, 'true_label': frame.label, 'p_ai': probability,
                                  'predicted_label_at_0_5': (probability >= 0.5).astype(int)})
        relative = f'results/model1_baseline_v1/{name}_predictions.csv'
        data = predicted.to_csv(index=False).encode()
        assert sha(data) == expected[relative], f'Restored predictions differ: {relative}; use locked dependencies and inspect platform differences.'
        payloads[relative] = data
    for relative, data in payloads.items():
        path = ROOT / relative
        if path.exists() and path.read_bytes() != data:
            raise RuntimeError(f'Existing different file preserved: {relative}')
    for relative, data in payloads.items():
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print('PASS: fixed split and predictions restored; all six CSVs match original reference SHA-256 hashes. No retraining.')

if __name__ == '__main__':
    main()
