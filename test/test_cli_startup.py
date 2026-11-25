"""Keep discovery commands independent of numerical-library imports."""
import importlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest


@pytest.mark.parametrize('argv,code', [
    ([], 0), (['--help'], 0), (['--version'], 0),
    (['predict'], 1), (['predict', '--help'], 0),
    (['predict-scan', '--help'], 0), (['predict-scan'], 1),
    (['downloads', 'info'], 0), (['downloads', 'list', '--json'], 0),
    (['downloads', 'releases', 'models_class1_presentation', '--json'], 0),
    (['downloads', 'info', 'models_class1_presentation'], 0),
    (['downloads', 'info', 'models_class1_pan'], 0),
    (['downloads', 'info', 'models_class1_processing', '--json'], 0),
    (['downloads', '--verbose', 'info'], 0),
    (['predict', '--model-release', 'not-a-release'], 2),
])
def test_discovery_does_not_import_numerical_stack(argv, code):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('MHCFLURRY_')}
    root = str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH'] = root
    script = '''
import sys
from mhcflurry.cli.main import main
try:
    code = main(%r) or 0
except SystemExit as error:
    code = error.code
assert code == %r, code
unexpected = set(sys.modules).intersection(['torch', 'numpy', 'pandas', 'sklearn', 'mhcgnomes'])
assert not unexpected, unexpected
''' % (argv, code)
    result = subprocess.run([sys.executable, '-c', script], env=env, cwd=root,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_classes_keep_identity_and_pickle_paths():
    import mhcflurry
    for name, module in mhcflurry._PUBLIC_CLASSES.items():
        value = getattr(mhcflurry, name)
        assert value is getattr(importlib.import_module('mhcflurry.' + module), name)
        assert pickle.loads(pickle.dumps(value)) is value
        assert name in dir(mhcflurry)
    with pytest.raises(AttributeError):
        getattr(mhcflurry, 'missing_public_class')


def test_parallelism_exports_keep_identity():
    from mhcflurry import parallelism

    for name, module in parallelism._EXPORTS.items():
        value = getattr(parallelism, name)
        assert value is getattr(importlib.import_module('mhcflurry.parallelism.' + module), name)
        assert name in dir(parallelism)
    with pytest.raises(AttributeError):
        getattr(parallelism, 'missing_worker_helper')


def test_catalogue_discovery_with_unknown_environment_release():
    env = dict(os.environ, MHCFLURRY_DOWNLOADS_CURRENT_RELEASE='unknown-release')
    env.pop('MHCFLURRY_DOWNLOADS_DIR', None)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    script = 'from mhcflurry.cli.main import main; main(["downloads", "releases", "--json"])'
    result = subprocess.run([sys.executable, '-c', script], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['default_release'] == '2.3.0'


def test_list_supported_alleles_and_lengths_skip_torch(tmp_path):
    """Listing supported alleles/lengths reads model metadata without importing torch."""
    from mhcflurry import Class1AffinityPredictor, Class1NeuralNetwork

    net = Class1NeuralNetwork(layer_sizes=[4], max_epochs=1, validation_split=0.0)
    net.fit(peptides=['SIINFEKL', 'NLVPMVATV'], affinities=[50.0, 5000.0], verbose=0)
    predictor = Class1AffinityPredictor(
        allele_to_allele_specific_models={'HLA-A*02:01': [net]},
        allele_to_sequence={'HLA-A*02:01': 'Y' * 37, 'HLA-B*07:02': 'A' * 37},
    )
    affinity_dir = tmp_path / 'affinity'
    predictor.save(str(affinity_dir))

    presentation_dir = tmp_path / 'presentation'
    presentation_dir.mkdir()
    (presentation_dir / 'weights.csv').write_text('intercept\n0.0\n')
    predictor.save(str(presentation_dir / 'affinity_predictor'))

    expected_alleles = predictor.supported_alleles
    min_len, max_len = predictor.supported_peptide_lengths
    expected_lengths = [str(length) for length in range(min_len, max_len + 1)]

    root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=root)
    cases = [
        (['predict', '--models', str(affinity_dir), '--list-supported-alleles'], expected_alleles),
        (['predict', '--models', str(presentation_dir), '--list-supported-alleles'], expected_alleles),
        (['predict-scan', '--models', str(presentation_dir), '--list-supported-alleles'], expected_alleles),
        (['predict', '--models', str(affinity_dir), '--list-supported-peptide-lengths'], expected_lengths),
        (['predict', '--models', str(presentation_dir), '--list-supported-peptide-lengths'], expected_lengths),
        (['predict-scan', '--models', str(presentation_dir), '--list-supported-peptide-lengths'], expected_lengths),
    ]
    for argv, expected_lines in cases:
        script = (
            'import sys; from mhcflurry.cli.main import main; '
            'main(%r); '
            'assert "torch" not in sys.modules, "torch was unexpectedly imported"'
            % (argv,)
        )
        result = subprocess.run(
            [sys.executable, '-c', script], env=env, cwd=root,
            capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip().splitlines() == expected_lines
        assert result.stderr == '', result.stderr
