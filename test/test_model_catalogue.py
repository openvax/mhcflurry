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


def record_source(path, release, bundle=BUNDLE):
    path.mkdir(parents=True, exist_ok=True)
    metadata = downloads.get_release_downloads(release)[bundle]['metadata']
    with (path / 'DOWNLOAD_INFO.csv').open('w', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow(['url'])
        writer.writerows([url] for url in downloads.get_download_urls(metadata))


def component_manifest(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'manifest.csv').write_text('model_name,config_json\n')


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
    assert not (versioned_cache / '2.2.0').exists()


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


def test_model_table_shows_components_available_through_presentation(
        versioned_cache):
    presentation = versioned_cache / '2.3.0' / BUNDLE
    models = presentation / 'models'
    component_manifest(models / 'affinity_predictor')
    component_manifest(models / 'processing_predictor_with_flanks')
    component_manifest(models / 'processing_predictor_without_flanks')
    record_source(presentation, '2.3.0')
    standalone = versioned_cache / '2.2.0' / 'models_class1_pan'
    record_source(standalone, '2.2.0', 'models_class1_pan')

    assert downloads_command._model_versions('models_class1_pan', {})[2] == (
        '2.3.0 via presentation, 2.2.0')
    assert downloads_command._model_versions('models_class1_processing', {})[2] == (
        '2.3.0 via presentation')
    assert downloads_command._model_versions(BUNDLE, {})[2] == '2.3.0'
    assert downloads.get_default_class1_processing_models_dir(test_exists=False) == str(
        versioned_cache / '2.3.0' / 'models_class1_processing' /
        'models.selected.with_flanks')
    record_source(versioned_cache / '2.3.0' / 'models_class1_pan',
                  '2.3.0', 'models_class1_pan')
    assert downloads_command._model_versions('models_class1_pan', {})[2] == (
        '2.3.0, 2.3.0 via presentation, 2.2.0')


def test_model_table_checks_embedded_component_directories(versioned_cache, capsys):
    presentation = versioned_cache / '2.3.0' / BUNDLE
    models = presentation / 'models'
    models.mkdir(parents=True)
    record_source(presentation, '2.3.0')

    assert downloads_command._model_versions('models_class1_pan', {})[2] == '—'
    assert downloads_command._model_versions('models_class1_processing', {})[2] == '—'

    (models / 'affinity_predictor').mkdir()
    (models / 'processing_predictor_with_flanks').mkdir()
    assert downloads_command._model_versions('models_class1_pan', {})[2] == '—'
    assert downloads_command._model_versions('models_class1_processing', {})[2] == '—'
    downloads_command.run(['info', 'models_class1_processing'])
    text = capsys.readouterr().out
    assert str(models / 'processing_predictor_with_flanks') + ' [missing manifest]' in text
    assert str(models / 'processing_predictor_without_flanks') + ' [not installed]' in text
    assert 'Standalone processing loading does not fall back' in text

    component_manifest(models / 'affinity_predictor')
    component_manifest(models / 'processing_predictor_with_flanks')
    assert downloads_command._model_versions('models_class1_pan', {})[2] == (
        '2.3.0 via presentation')
    assert downloads_command._model_versions('models_class1_processing', {})[2] == (
        '2.3.0 via presentation (with flanks only)')

    component_manifest(models / 'processing_predictor_without_flanks')
    assert downloads_command._model_versions('models_class1_processing', {})[2] == (
        '2.3.0 via presentation')
    (models / 'processing_predictor_with_flanks' / 'manifest.csv').unlink()
    assert downloads_command._model_versions('models_class1_processing', {})[2] == (
        '2.3.0 via presentation (without flanks only)')


def test_embedded_component_source_status_and_custom_roots(
        tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, '_CURRENT_RELEASE', None)
    monkeypatch.setattr(downloads, '_DOWNLOADS_DIR', str(tmp_path))
    presentation = tmp_path / BUNDLE
    component_manifest(presentation / 'models' / 'affinity_predictor')

    assert downloads_command._model_versions('models_class1_pan', {})[2] == (
        'custom? via presentation')
    record_source(presentation, '2.3.0')
    assert downloads_command._model_versions('models_class1_pan', {})[2] == (
        'custom: 2.3.0 via presentation')
    (presentation / 'DOWNLOAD_INFO.csv').write_text(
        'url\nhttps://example.org/custom-models\n')
    assert downloads_command._model_versions('models_class1_pan', {})[2] == (
        'custom! via presentation')


def test_versioned_embedded_component_preserves_source_markers(versioned_cache):
    presentation = versioned_cache / '2.3.0' / BUNDLE
    component_manifest(presentation / 'models' / 'affinity_predictor')

    assert downloads_command._model_versions('models_class1_pan', {})[2] == (
        '2.3.0? via presentation')
    (presentation / 'DOWNLOAD_INFO.csv').write_text(
        'url\nhttps://example.org/different-presentation\n')
    assert downloads_command._model_versions('models_class1_pan', {})[2] == (
        '2.3.0! via presentation')


def test_component_details_preserve_standalone_status_and_paths(versioned_cache, capsys):
    root = versioned_cache / '2.3.0'
    affinity = root / BUNDLE / 'models' / 'affinity_predictor'
    component_manifest(affinity)
    record_source(root / BUNDLE, '2.3.0')
    before = {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in root.rglob('*') if p.is_file()}
    downloads_command.run(['info', 'models_class1_pan', '--json'])
    record = json.loads(capsys.readouterr().out)
    assert record['downloaded'] is False
    assert record['status'] == 'not installed'
    assert record['source_matches'] is None
    assert record['path'] == str(root / 'models_class1_pan')
    assert record['presentation_components'] == [dict(
        name='affinity_predictor', path=str(affinity), directory_exists=True,
        manifest_exists=True, source_matches=True)]

    downloads_command.run(['info', BUNDLE])
    text = capsys.readouterr().out
    assert str(affinity) + ' [manifest present]' in text
    assert str(root / BUNDLE / 'models' / 'processing_predictor_with_flanks') in text
    assert 'not installed' in text
    assert 'Presentation loading uses these embedded components' in text
    assert 'equivalence to standalone weights' in text
    downloads_command.run(['info', 'models_class1_pan'])
    text = capsys.readouterr().out
    assert 'Default affinity loading can fall back to this component' in text
    downloads_command.run(['list', '--json'])
    records = json.loads(capsys.readouterr().out)['downloads']
    assert next(r for r in records if r['name'] == 'models_class1_pan') == {
        key: value for key, value in record.items() if key not in ('versions', 'fetch_command')}
    assert {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()} == before
    assert not (root / 'models_class1_pan').exists()


def test_component_details_follow_browsed_release_not_default_override(
        versioned_cache, monkeypatch, capsys):
    custom = versioned_cache / 'custom predictor'
    component_manifest(custom / 'affinity_predictor')
    component_manifest(versioned_cache / '2.2.0' / BUNDLE / 'models' /
                       'processing_predictor_without_flanks')
    monkeypatch.setattr(downloads, '_MHCFLURRY_DEFAULT_CLASS1_PRESENTATION_MODELS_DIR', str(custom))
    affinity_override = str(versioned_cache / 'missing affinity override')
    monkeypatch.setattr(downloads, '_MHCFLURRY_DEFAULT_CLASS1_MODELS_DIR', affinity_override)
    downloads_command.run(['info', 'models_class1_processing', '--release', '2.2.0', '--json'])
    components = json.loads(capsys.readouterr().out)['presentation_components']
    assert [item['manifest_exists'] for item in components] == [False, True]
    assert all('/2.2.0/' in item['path'] for item in components)

    downloads_command.run(['info', '--release', '2.2.0', '--json'])
    config = json.loads(capsys.readouterr().out)
    assert config['default_model_paths']['affinity'] == dict(path=affinity_override, exists=False)
    assert config['default_model_paths']['presentation'] == dict(path=str(custom), exists=True)
    assert config['default_presentation_components'][0]['path'] == str(custom / 'affinity_predictor')
    assert config['default_model_paths']['processing']['path'] == str(
        versioned_cache / '2.3.0' / 'models_class1_processing' / 'models.selected.with_flanks')
    # Discovery neither rewrites overrides nor introduces a processing fallback.
    with pytest.raises(IOError, match='No such directory'):
        downloads.get_default_class1_models_dir()
    with pytest.raises(RuntimeError):
        downloads.get_default_class1_processing_models_dir()
    downloads_command.run(['--verbose', 'info'])
    text = capsys.readouterr().out
    assert str(custom / 'affinity_predictor') in text
    assert 'no affinity-path override' in text


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
