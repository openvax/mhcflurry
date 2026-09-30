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


def test_parallelism_exports_keep_identity_and_legacy_imports():
    from mhcflurry import parallelism, local_parallelism
    for name, module in parallelism._EXPORTS.items():
        value = getattr(parallelism, name)
        assert value is getattr(importlib.import_module('mhcflurry.parallelism.' + module), name)
        assert getattr(local_parallelism, name) is value
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
