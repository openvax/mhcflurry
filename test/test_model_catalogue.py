"""Discovery and explicit release selection must agree about paths and sources."""
import csv
import io
import json
import re
import sys
from pathlib import Path

import pytest

from mhcflurry import downloads
from mhcflurry.cli import downloads_command, predict_command, predict_scan_command

BUNDLE = 'models_class1_presentation'


@pytest.fixture
def versioned_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, '_CURRENT_RELEASE', '2.3.0')
    monkeypatch.setattr(downloads, '_DOWNLOADS_DIR', str(tmp_path / '2.3.0'))
    return tmp_path


def record_source(path, release):
    path.mkdir(parents=True, exist_ok=True)
    metadata = downloads.get_release_downloads(release)[BUNDLE]['metadata']
    with (path / 'DOWNLOAD_INFO.csv').open('w', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow(['url'])
        writer.writerows([url] for url in downloads.get_download_urls(metadata))


def test_all_bundles_have_descriptions():
    metadata = downloads.get_downloads_metadata()
    names = {item['name'] for release in metadata['releases'].values()
             for item in release['downloads']}
    assert names == set(metadata['bundles'])
    for name in names:
        info = downloads.get_bundle_description(name)
        assert info['kind'] in ('models', 'data')
        assert info['description'] and info['group']


def test_shared_archive_versions_are_grouped():
    groups = downloads.get_bundle_versions(BUNDLE)
    assert any(item['releases'] == ['2.2.0', '2.0.0'] for item in groups)
    assert len({tuple(item['urls']) for item in groups}) == len(groups)


def test_list_json_and_historical_source(versioned_cache, capsys):
    downloads_command.run(['list', '--kind', 'models', '--release', '2.2.0', '--json'])
    result = json.loads(capsys.readouterr().out)
    assert result['release'] == '2.2.0'
    assert all(item['kind'] == 'models' for item in result['downloads'])
    presentation = next(item for item in result['downloads'] if item['name'] == BUNDLE)
    assert presentation['path'] == str(versioned_cache / '2.2.0' / BUNDLE)
    assert '/pre-2.0/' in presentation['urls'][0]
    assert presentation['status'] == 'not installed'


def test_info_resolves_defaults_and_source_status(versioned_cache, capsys):
    target = versioned_cache / '2.3.0' / BUNDLE
    target.mkdir(parents=True)
    downloads_command.run(['info', BUNDLE, '--json'])
    assert json.loads(capsys.readouterr().out)['source_matches'] is None
    record_source(target, '2.3.0')
    downloads_command.run(['info', BUNDLE, '--json'])
    assert json.loads(capsys.readouterr().out)['source_matches'] is True
    record_source(target, '2.2.0')
    downloads_command.run(['info', BUNDLE, '--json'])
    assert json.loads(capsys.readouterr().out)['source_matches'] is False
    downloads_command.run(['--verbose', 'info'])
    text = capsys.readouterr().out
    assert text.index('Resolved configuration') < text.index('Environment variables')
    assert 'optional overrides' in text
    assert 'not file integrity' in text


def test_info_puts_weight_versions_before_history_and_configuration(versioned_cache, capsys):
    record_source(versioned_cache / '2.3.0' / BUNDLE, '2.3.0')
    record_source(versioned_cache / '2.2.0' / BUNDLE, '2.2.0')
    downloads_command.run(['info'])
    text = capsys.readouterr().out
    assert text.index('LATEST') < text.index('Historical models')
    assert text.index('Historical models') < text.index('Resolved configuration')
    assert 'OTHER VERSIONS' in text and 'INSTALLED' in text
    assert 'Recommended: models_class1_presentation (includes affinity and processing).' in text
    model_rows = [line for line in text.splitlines() if line.startswith('models_class1_')]
    assert model_rows[0].startswith(BUNDLE + ' ')
    assert 'Full presentation predictor, including' not in text
    assert 'Default prediction paths' not in text
    assert 'mhcflurry downloads --verbose info' in text
    row = next(line for line in text.splitlines() if line.startswith(BUNDLE + ' '))
    assert row.index('2.3.0') < row.index('2.2.0')
    # The actual installed catalogue directories remain selectable, including
    # aliases; availability groups shared archive URLs only once.
    cache = {}
    latest, other, installed = downloads_command._model_versions(BUNDLE, cache)
    assert latest == '2.3.0'
    assert other == '2.2.0, 1.7.0, 1.6.0'
    assert installed == '2.3.0, 2.2.0'


def test_model_table_marks_unverified_sources_and_custom_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, '_CURRENT_RELEASE', None)
    monkeypatch.setattr(downloads, '_DOWNLOADS_DIR', str(tmp_path))
    path = tmp_path / BUNDLE
    path.mkdir()
    assert downloads_command._model_versions(BUNDLE, {})[2] == 'custom?'
    record_source(path, '2.2.0')
    assert downloads_command._model_versions(BUNDLE, {})[2] == 'custom: 2.2.0'
    (path / 'DOWNLOAD_INFO.csv').write_text('url\nhttps://example.org/custom-models\n')
    assert downloads_command._model_versions(BUNDLE, {})[2] == 'custom!'


@pytest.mark.parametrize('environment', [{}, {'NO_COLOR': '1'}, {'TERM': 'dumb'}])
def test_table_color_preserves_alignment_and_plain_output(monkeypatch, environment):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv('NO_COLOR', raising=False)
    monkeypatch.setenv('TERM', 'xterm-256color')
    monkeypatch.setenv('COLUMNS', '80')
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    plain = io.StringIO()
    terminal = Terminal()
    rows = [(BUNDLE, '2.3.0', '2.2.0, 1.7.0, 1.6.0', '2.3.0, 2.2.0'),
            ('models_class1_pan', '2.3.0', '2.2.0, 1.7.0', '—')]
    for stream in (plain, terminal):
        monkeypatch.setattr(sys, 'stdout', stream)
        downloads_command._print_table(
            ('MODEL', 'LATEST', 'OTHER VERSIONS', 'INSTALLED'), rows,
            colors={1: '36', 3: downloads_command._status_color}, wrap_columns=(2, 3))
    assert '\x1b[' not in plain.getvalue()
    assert ('\x1b[' in terminal.getvalue()) == (not environment)
    assert re.sub(r'\x1b\[[0-9;]*m', '', terminal.getvalue()) == plain.getvalue()
    assert max(map(len, plain.getvalue().splitlines())) <= 80


def test_json_remains_uncolored_on_a_terminal(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, 'isatty', lambda: True)
    monkeypatch.delenv('NO_COLOR', raising=False)
    monkeypatch.setenv('TERM', 'xterm-256color')
    downloads_command.run(['info', '--json'])
    text = capsys.readouterr().out
    assert '\x1b[' not in text
    result = json.loads(text)
    assert 'default_model_paths' in result and 'environment_overrides' in result


def test_releases_filter_and_compatibility(capsys):
    downloads_command.run(['releases', BUNDLE, '--json'])
    records = json.loads(capsys.readouterr().out)['releases']
    assert [item['release'] for item in records] == ['2.3.0', '2.2.0', '2.0.0', '1.7.0', '1.6.0']
    downloads_command.run(['releases', '--json'])
    records = json.loads(capsys.readouterr().out)['releases']
    assert not next(item for item in records if item['release'] == '0.2.0')['compatible']


@pytest.mark.parametrize('argv,message', [
    (['list', '--release', '2.1.5'], 'Unknown download release'),
    (['info', 'missing'], 'not in release'),
    (['url', 'missing'], 'not in release'),
    (['path', 'missing'], 'not in release'),
    (['releases', 'missing'], 'Unknown download'),
    (['url'], 'required'),
])
def test_actionable_errors_without_traceback(argv, message, capsys):
    with pytest.raises(SystemExit) as error:
        downloads_command.run(argv)
    assert error.value.code == 2
    text = capsys.readouterr().err
    assert message in text
    assert 'Traceback' not in text


def test_explicit_release_overrides_active_and_model_defaults(versioned_cache, monkeypatch):
    old_models = versioned_cache / '2.2.0' / BUNDLE / 'models'
    old_models.mkdir(parents=True)
    monkeypatch.setattr(downloads, '_MHCFLURRY_DEFAULT_CLASS1_PRESENTATION_MODELS_DIR', '/other/models')
    assert downloads.get_model_release_dir('2.2.0') == str(old_models)
    assert downloads.get_default_class1_presentation_models_dir(False) == '/other/models'


def test_missing_release_download_suggests_exact_command(versioned_cache):
    with pytest.raises(RuntimeError, match='fetch models_class1_presentation --release 2.2.0'):
        downloads.get_model_release_dir('2.2.0')
    with pytest.raises(ValueError, match='no presentation bundle'):
        downloads.get_model_release_dir('1.5.0')
    with pytest.raises(ValueError, match='incompatible'):
        downloads.get_model_release_dir('0.2.0')


def test_custom_root_requires_matching_source(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, '_CURRENT_RELEASE', None)
    monkeypatch.setattr(downloads, '_DOWNLOADS_DIR', str(tmp_path))
    models = tmp_path / BUNDLE / 'models'
    models.mkdir(parents=True)
    with pytest.raises(ValueError, match='Cannot confirm'):
        downloads.get_model_release_dir('2.2.0')
    record_source(models.parent, '2.3.0')
    with pytest.raises(ValueError, match='Cannot confirm'):
        downloads.get_model_release_dir('2.2.0')
    record_source(models.parent, '2.2.0')
    assert downloads.get_model_release_dir('2.2.0') == str(models)


def test_path_and_url_select_requested_release(versioned_cache, capsys):
    path = versioned_cache / '2.2.0' / BUNDLE
    path.mkdir(parents=True)
    downloads_command.run(['path', BUNDLE, '--release', '2.2.0'])
    assert Path(capsys.readouterr().out.strip()) == path
    downloads_command.run(['url', BUNDLE, '--release', '2.2.0'])
    assert '/pre-2.0/' in capsys.readouterr().out


@pytest.mark.parametrize('module', [predict_command, predict_scan_command])
def test_model_selection_flags_are_exclusive(module):
    with pytest.raises(SystemExit):
        module.parser.parse_args(['--models', '/models', '--model-release', '2.2.0'])
    args = module.parser.parse_args(['--model-release', '2.2.0'])
    assert args.model_release == '2.2.0' and args.models is None


@pytest.mark.parametrize('module', [predict_command, predict_scan_command])
def test_prediction_loads_selected_release(module, versioned_cache, monkeypatch, capsys):
    path = versioned_cache / '2.2.0' / BUNDLE / 'models'
    path.mkdir(parents=True)
    (path / 'weights.csv').touch()
    seen = []

    class Predictor:
        supported_alleles = ['test-allele']

    def load(directory):
        seen.append(directory)
        if module is predict_command:
            return Predictor(), False
        return Predictor()

    monkeypatch.setattr(module, '_load_predictor_for_command', load)
    module.run(['--model-release', '2.2.0', '--list-supported-alleles'])
    assert seen == [str(path)]
    assert capsys.readouterr().out.strip() == 'test-allele'
