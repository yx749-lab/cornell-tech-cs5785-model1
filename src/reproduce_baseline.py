"""Reproduce the unchanged baseline in a new directory; preserve reference files."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    import numpy as np
    import pandas as pd
    (ROOT / 'work').mkdir(exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix='baseline_reproduction_', dir=ROOT / 'work'))
    for relative in ['src/train_baseline.py', 'requirements.lock.txt', 'data/raw/train.csv']:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    shutil.copytree(ROOT / 'data/splits', destination / 'data/splits')
    (destination / 'models').mkdir()
    completed = subprocess.run([sys.executable, str(destination / 'src/train_baseline.py')],
                               cwd=destination, capture_output=True, text=True)
    (destination / 'training.log').write_text(completed.stdout + '\n' + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f'Reproduction failed. See {destination / "training.log"}')
    checks = {}
    for name in ['train', 'validation']:
        reference = pd.read_csv(ROOT / f'results/model1_baseline_v1/{name}_predictions.csv', dtype={'id': str})
        actual = pd.read_csv(destination / f'results/model1_baseline_v1/{name}_predictions.csv', dtype={'id': str})
        assert reference.id.tolist() == actual.id.tolist()
        assert reference.true_label.tolist() == actual.true_label.tolist()
        difference = float(np.max(np.abs(reference.p_ai.to_numpy() - actual.p_ai.to_numpy())))
        assert difference <= 1e-8, f'{name}: probability difference {difference} exceeds 1e-8'
        assert reference.predicted_label_at_0_5.equals(actual.predicted_label_at_0_5)
        checks[name] = {'ordered_ids_match': True, 'max_probability_difference': difference,
                        'threshold_decisions_match': True}
    report = {'status': 'passed', 'same_configuration_retrained': True, 'hyperparameter_search': False,
              'reference_artifacts_preserved': True, 'probability_tolerance': 1e-8, 'checks': checks,
              'output_directory': str(destination.relative_to(ROOT))}
    (destination / 'reproduction_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
