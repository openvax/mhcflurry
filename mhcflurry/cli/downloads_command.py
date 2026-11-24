# PYTHON_ARGCOMPLETE_OK
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Download MHCflurry released datasets and trained models.

Examples

Fetch the default downloads:
    $ mhcflurry-downloads fetch

Fetch a specific download:
    $ mhcflurry-downloads fetch models_class1_pan

Get the path to a download:
    $ mhcflurry-downloads path models_class1_pan

Get the URL of a download:
    $ mhcflurry-downloads url models_class1_pan

Summarize available and fetched downloads:
    $ mhcflurry-downloads info
"""
import sys
import json
import logging
import os
from shlex import quote
import textwrap
from itertools import zip_longest
from shutil import get_terminal_size, rmtree

import argcomplete

from ..downloads import (
    get_current_release,
    get_downloads_metadata,
    get_bundle_description,
    get_bundle_versions,
    get_download_urls,
    install_download,
    resolve_data_location,
    resolve_release,
    get_release_downloads,
    get_downloads_dir,
    get_default_class1_models_dir,
    get_default_class1_presentation_models_dir,
    get_default_class1_processing_models_dir,
    get_path,
    ENVIRONMENT_VARIABLES)
from ..version import __version__
from .completion import is_completing
from .help import HelpArgumentParser, color_enabled

parser = HelpArgumentParser(
    description="""Browse, download and locate released weights and supporting data.

Start here:
  mhcflurry downloads releases models_class1_presentation
  mhcflurry downloads list --kind models
  mhcflurry downloads info models_class1_presentation
  mhcflurry downloads fetch models_class1_presentation --release 2.2.0
  mhcflurry predict INPUT.csv --model-release 2.2.0

Download releases are catalogue versions, separate from the installed code.
Browsing uses the catalogue shipped with this package; no network is required.
""")

parser.add_argument(
    "--quiet",
    action="store_true",
    default=False,
    help="Output less")

parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    default=False,
    help="Output more")

subparsers = parser.add_subparsers(dest="subparser_name")

parser_fetch = subparsers.add_parser('fetch')
parser_fetch.add_argument(
    'download_name',
    metavar="DOWNLOAD",
    nargs="*",
    help="Items to download")
parser_fetch.add_argument(
    "--keep",
    action="store_true",
    default=False,
    help="Deprecated (ignored): datacache manages archive staging automatically; "
         "use --already-downloaded-dir to retain local archives")
parser_fetch.add_argument(
    "--force",
    action="store_true",
    default=False,
    help="Re-download and install a fresh bundle generation even if already installed")
parser_fetch.add_argument(
    "--release",
    # The current release comes from the download catalogue, which a Tab
    # press does not need to read.
    default=None if is_completing() else get_current_release(),
    help="Release to download. Default: %(default)s")
parser_fetch.add_argument(
    "--already-downloaded-dir",
    metavar="DIR",
    help="Don't download files, get them from DIR")
parser_fetch.add_argument(
    "--offline",
    action="store_true",
    default=False,
    help="Disallow network downloads (also honors OPENVAX_OFFLINE / DATACACHE_OFFLINE)")

parser_dirs = subparsers.add_parser(
    'dirs',
    help='Show data cache locations, sizes, and offline status (read-only)')
parser_dirs.add_argument(
    '--json', action='store_true', help='Print machine-readable JSON')

parser_info = subparsers.add_parser('info')
parser_info.add_argument('download_name', nargs='?', metavar='DOWNLOAD')
parser_info.add_argument('--json', action='store_true', help='Print machine-readable JSON')
parser_info.add_argument(
    '--verify',
    action='store_true',
    default=False,
    help='Hash extracted files in datacache-managed bundles to verify integrity')

parser_list = subparsers.add_parser('list', help='Browse models and supporting data')
parser_list.add_argument('--kind', choices=('all', 'models', 'data'), default='all')
parser_list.add_argument('--json', action='store_true', help='Print machine-readable JSON')

parser_releases = subparsers.add_parser('releases', help='List valid catalogue releases')
parser_releases.add_argument('download_name', nargs='?', metavar='DOWNLOAD')
parser_releases.add_argument('--json', action='store_true', help='Print machine-readable JSON')

parser_path = subparsers.add_parser('path')
parser_path.add_argument(
    "download_name",
    nargs="?",
    default='')
parser_path.add_argument('--json', action='store_true', help='Print machine-readable JSON')

parser_prune = subparsers.add_parser(
    'prune',
    help='Remove non-current downloaded release directories (dry-run by default)')
parser_prune.add_argument(
    '--yes',
    action='store_true',
    default=False,
    help='Actually delete non-current release directories (default is dry-run)')
parser_prune.add_argument(
    '--keep-release',
    action='append',
    default=None,
    metavar='RELEASE',
    help='Additional catalogue release(s) to preserve (repeatable)')
parser_prune.add_argument(
    '--json', action='store_true', help='Print machine-readable JSON')

parser_url = subparsers.add_parser('url')
parser_url.add_argument(
    "download_name")

for subparser in (parser_info, parser_list, parser_path, parser_url):
    subparser.add_argument('--release', metavar='RELEASE',
                          help='Catalogue release (default: configured release)')


def run(argv=sys.argv[1:]):
    import datacache

    argcomplete.autocomplete(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=(logging.DEBUG if args.verbose else
                               logging.WARNING if args.quiet else logging.INFO))

    command_functions = {
        "dirs": dirs_subcommand,
        "fetch": fetch_subcommand,
        "info": info_subcommand,
        "list": list_subcommand,
        "releases": releases_subcommand,
        "path": path_subcommand,
        "prune": prune_subcommand,
        "url": url_subcommand,
        None: lambda args: parser.print_help(),
    }
    try:
        command_functions[args.subparser_name](args)
    except (
        ValueError,
        RuntimeError,
        datacache.OfflineError,
        datacache.FileValidationError,
    ) as error:
        parser.error(str(error))


def yes_no(boolean):
    return "YES" if boolean else "NO"


def fetch_subcommand(args):
    def qprint(msg):
        if not args.quiet:
            print(msg, file=sys.stderr)

    if not args.release:
        raise RuntimeError(
            "No release defined. This can happen when you are specifying "
            "a custom models directory. Specify --release to indicate "
            "the release to download.")

    if getattr(args, "keep", False):
        logging.warning(
            "--keep is deprecated and ignored; datacache manages archive "
            "staging automatically. Use --already-downloaded-dir to retain "
            "and reuse local archives."
        )

    force = bool(getattr(args, "force", False))
    downloads = get_release_downloads(args.release)
    invalid_download_names = set(
        item for item in args.download_name if item not in downloads)
    if invalid_download_names:
        raise ValueError("Unknown download(s): %s. Valid downloads are: %s" % (
            ', '.join(invalid_download_names), ', '.join(downloads)))

    items_to_fetch = []
    for (name, info) in downloads.items():
        default = not args.download_name and info['metadata']['default']
        requested = name in args.download_name or default
        if not requested:
            continue
        if info['downloaded'] and not force:
            if name in args.download_name:
                print(
                    (
                        "*" * 40
                        + "\nThe requested download '%s' is already installed at:\n\t%s\n"
                        "To re-download or repair this data, re-run with --force.\n"
                        + "*" * 40
                    )
                    % (name, quote(get_path(name, release=args.release))),
                    file=sys.stderr,
                )
            continue
        items_to_fetch.append(name)

    offline_flag = True if getattr(args, "offline", False) else None
    if items_to_fetch and not args.already_downloaded_dir:
        import datacache

        for item in items_to_fetch:
            for url in get_download_urls(downloads[item]["metadata"]):
                datacache.require_online(url, offline=offline_flag)

    qprint("Fetching %d/%d downloads from release %s" % (
        len(items_to_fetch), len(downloads), args.release))
    format_string = "%-40s  %-20s   %-20s  %-20s "
    qprint(format_string % (
        "DOWNLOAD NAME", "ALREADY DOWNLOADED?", "WILL DOWNLOAD NOW?", "URL"))

    items_to_fetch_set = set(items_to_fetch)
    for (item, info) in downloads.items():
        urls = get_download_urls(info['metadata'])
        url_description = urls[0]
        if len(urls) > 1:
            url_description += " + %d more parts" % (len(urls) - 1)

        qprint(format_string % (
            item,
            yes_no(info['downloaded']),
            yes_no(item in items_to_fetch_set),
            url_description))

    for item in items_to_fetch:
        bundle_path = install_download(
            item,
            release=args.release,
            metadata=downloads[item]['metadata'],
            already_downloaded_dir=args.already_downloaded_dir,
            force=force,
            offline=offline_flag,
            show_progress=(False if args.quiet else "auto"),
        )
        qprint("Installed %s to: %s" % (item, quote(str(bundle_path))))


def _download_records(release, kind="all", verify_files=False):
    records = []
    downloads = get_release_downloads(release, verify_files=verify_files)
    presentation = downloads.get('models_class1_presentation')
    for name, info in downloads.items():
        description = get_bundle_description(name)
        if kind != "all" and description["kind"] != kind:
            continue
        if not info["downloaded"]:
            status = (
                "invalid installation"
                if info.get("install_status") == "invalid"
                else "not installed"
            )
        elif info.get("verified"):
            status = "installed; verified"
        elif info["up_to_date"] is None:
            status = "installed; source unknown"
        elif info["up_to_date"]:
            status = "installed; source matches"
        else:
            status = "installed; source differs"
        records.append(dict(
            name=name, release=release, **description,
            status=status, downloaded=info["downloaded"],
            source_matches=info["up_to_date"],
            path=os.path.abspath(get_path(name, test_exists=False, release=release)),
            urls=get_download_urls(info["metadata"]),
            default=info["metadata"].get("default", False),
            layout=info.get("layout"),
            install_status=info.get("install_status"),
            verified=bool(info.get("verified")),
            trusted_checksum=bool(info.get("trusted_checksum")),
            fetched_at=info.get("fetched_at"),
            archive_size=info.get("archive_size"),
            recorded_sha256=info.get("recorded_sha256"),
            recorded_urls=info.get("source_urls"),
        ))
        if presentation and (name == 'models_class1_presentation' or
                             name in _PRESENTATION_COMPONENT_DIRS):
            records[-1]['presentation_components'] = _presentation_components(
                get_path('models_class1_presentation', 'models',
                         test_exists=False, release=release),
                name, presentation['up_to_date'])
    return records


def _find_download(release, name, verify_files=False):
    records = _download_records(release, verify_files=verify_files)
    for record in records:
        if record["name"] == name:
            return record
    raise ValueError(
        "Download %r is not in release %s. Run 'mhcflurry downloads list "
        "--release %s' or 'mhcflurry downloads releases %s'."
        % (name, release, release, name))


def _style(text, code):
    if not code or not color_enabled(sys.stdout):
        return text
    return "\033[%sm%s\033[0m" % (code, text)


def _heading(text):
    print("\n" + _style(text, "1;36"))


def _status_color(text):
    if text in ("—", "NO"):
        return "2"
    if text in ("unknown", "differs") or "?" in text or "!" in text:
        return "33"
    return None


_PRESENTATION_COMPONENT_DIRS = {
    'models_class1_pan': ('affinity_predictor',),
    'models_class1_processing': (
        'processing_predictor_with_flanks',
        'processing_predictor_without_flanks'),
}


def _presentation_components(models_dir, name=None, source_matches=None):
    """Inspect component paths without loading weights or verifying integrity."""
    components = []
    for bundle, directories in _PRESENTATION_COMPONENT_DIRS.items():
        if name not in (None, 'models_class1_presentation', bundle):
            continue
        for directory in directories:
            path = os.path.abspath(os.path.join(models_dir, directory))
            components.append(dict(
                name=directory, path=path,
                directory_exists=os.path.isdir(path),
                manifest_exists=os.path.isfile(os.path.join(path, 'manifest.csv')),
                source_matches=source_matches))
    return components


def _print_presentation_components(components):
    """Show each component's exact path and limited filesystem checks."""
    for component in components:
        status = ('manifest present' if component['manifest_exists'] else
                  'missing manifest' if component['directory_exists'] else
                  'not installed')
        print('  %s: %s [%s]' % (component['name'], component['path'], status))
    print('  Presence does not verify model files or imply equivalence to standalone weights.')


def _print_table(headers, rows, colors=None, wrap_columns=()):
    """Align plain cell widths before adding optional terminal color."""
    if not rows:
        return
    widths = [max(len(row[i]) for row in [headers, *rows])
              for i in range(len(headers))]
    available = max(60, min(120, get_terminal_size().columns))
    for column in wrap_columns:
        excess = sum(widths) + 2 * (len(widths) - 1) - available
        if excess > 0:
            widths[column] = max(len(headers[column]), 16, widths[column] - excess)
    print(_style("  ".join(value.ljust(width) for value, width in
                           zip(headers, widths)).rstrip(), "1;36"))
    for row in rows:
        cells = [textwrap.wrap(value, width, break_long_words=False,
                               break_on_hyphens=False) or [""]
                 for value, width in zip(row, widths)]
        for line in zip_longest(*cells, fillvalue=""):
            output = []
            for i, (value, width) in enumerate(zip(line, widths)):
                color = (colors or {}).get(i)
                code = color(row[i]) if callable(color) else color
                cell = value if i == len(widths) - 1 else value.ljust(width)
                output.append(_style(cell, code))
            print("  ".join(output).rstrip())


def _model_versions(name, releases):
    """Describe archive sources and installed standalone or embedded models."""
    versions = get_bundle_versions(name)
    installed = []
    custom = get_current_release() is None
    custom_candidates = {}
    standalone_releases = {
        release for version in versions for release in version['releases']}
    presentation_releases = (
        {release for version in get_bundle_versions('models_class1_presentation')
         for release in version['releases']}
        if name in _PRESENTATION_COMPONENT_DIRS else set())
    candidate_releases = standalone_releases | presentation_releases

    for release in get_downloads_metadata()['releases']:
        if release not in candidate_releases:
            continue
        if release not in releases:
            releases[release] = get_release_downloads(release)
        release_downloads = releases[release]
        sources = []
        standalone = release_downloads.get(name)
        if standalone and standalone['downloaded']:
            sources.append((standalone, ''))
        presentation = release_downloads.get('models_class1_presentation')
        if presentation and name in _PRESENTATION_COMPONENT_DIRS:
            components = _presentation_components(get_path(
                'models_class1_presentation', 'models', test_exists=False,
                release=release), name)
            present = [item for item in components if item['manifest_exists']]
            if present:
                suffix = ' via presentation'
                if len(present) != len(components):
                    variant = ('with flanks' if present[0]['name'].endswith('_with_flanks')
                               else 'without flanks')
                    suffix += ' (%s only)' % variant
                sources.append((presentation, suffix))
        for info, suffix in sources:
            marker = ("" if info['up_to_date'] is True else
                      "?" if info['up_to_date'] is None else "!")
            if custom:
                custom_candidates.setdefault(suffix, []).append((release, marker))
            else:
                installed.append(release + marker + suffix)

    if custom:
        for suffix, candidates in custom_candidates.items():
            if not candidates:
                continue
            matching = next((release for release, marker in candidates
                             if not marker), None)
            if matching:
                installed.append('custom: ' + matching + suffix)
            else:
                marker = '?' if any(marker == '?' for _, marker in candidates) else '!'
                installed.append('custom' + marker + suffix)
    return (versions[0]['releases'][0],
            ', '.join(v['releases'][0] for v in versions[1:]) or '—',
            ', '.join(installed) or '—')


def _print_downloads(records):
    primary = [record for record in records if record['group'] == 'Prediction models']
    primary.sort(key=lambda record: record['name'] != 'models_class1_presentation')
    historical = [record for record in records if record['group'] != 'Prediction models']
    if primary:
        _heading('Prediction models — latest weights and available versions')
        if primary[0]['name'] == 'models_class1_presentation':
            print('Recommended: models_class1_presentation (includes affinity and processing).')
            print()
        releases = {}
        rows = [(record['name'] + (' (affinity)' if record['name'] == 'models_class1_pan' else ''),
                 *_model_versions(record['name'], releases))
                for record in primary]
        _print_table(('MODEL', 'LATEST', 'OTHER VERSIONS', 'INSTALLED'), rows,
                     colors={1: '36', 3: _status_color}, wrap_columns=(2, 3))
        print("Shared archive aliases are grouped under one version; 'releases NAME' lists all.")
        print("'via presentation' identifies embedded components, not standalone bundle installs.")
        if any('?' in row[3] or '!' in row[3] for row in rows):
            print('Installed: ? unknown source; ! recorded source differs from that catalogue.')
    for group, kind in (('Historical models', 'models'), ('Supporting data', 'data')):
        items = [record for record in historical if record['kind'] == kind]
        if not items:
            continue
        _heading(group)
        rows = []
        for record in items:
            source = ('—' if not record['downloaded'] else
                      'unknown' if record['source_matches'] is None else
                      'matches' if record['source_matches'] else 'differs')
            rows.append((record['name'], yes_no(record['downloaded']), source))
        _print_table(('DOWNLOAD', 'LOCAL', 'SOURCE'), rows,
                     colors={1: _status_color, 2: _status_color})
    print("\nLocal status checks directories, component manifests and recorded source URLs, not file integrity.")
    print("Details: mhcflurry downloads info NAME | Versions: mhcflurry downloads releases NAME")


def list_subcommand(args):
    """Show a readable catalogue or machine-readable bundle records."""
    release = resolve_release(args.release)
    records = _download_records(release, args.kind)
    if args.json:
        print(json.dumps(dict(release=release, downloads=records), indent=2))
        return
    print("Download release %s (code %s)" % (release, __version__))
    _print_downloads(records)
    print("\nCatalogue directory: " + os.path.abspath(get_downloads_dir(release)))


def releases_subcommand(args):
    """List catalogue IDs and group identical sources for an optional bundle."""
    metadata = get_downloads_metadata()
    name = args.download_name
    versions = get_bundle_versions(name) if name else []
    records = []
    for release, info in metadata['releases'].items():
        names = [item['name'] for item in info['downloads']]
        if name and name not in names:
            continue
        records.append(dict(
            release=release, default=release == metadata['current-release'],
            configured=release == get_current_release(),
            compatible=info['compatibility-version'] == metadata['current-compatibility-version'],
            models=[item for item in names if get_bundle_description(item)['kind'] == 'models']))
    result = dict(code_version=__version__, default_release=metadata['current-release'],
                  releases=records, download=name, versions=versions)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print("Download releases (separate from code %s)" % __version__)
    rows = []
    for record in records:
        types = [label for bundle, label in (
            ('models_class1_presentation', 'presentation'),
            ('models_class1_pan', 'pan-affinity'),
            ('models_class1_processing', 'processing'),
            ('models_class1', 'legacy affinity')) if bundle in record['models']]
        if not types:
            types = ['legacy affinity']
        if record['default']:
            types.append('default')
        if record['configured']:
            types.append('configured')
        rows.append((
            record['release'], 'compatible' if record['compatible'] else 'incompatible',
            ', '.join(types)))
    _print_table(('RELEASE', 'FORMAT', 'PREDICTORS / NOTES'), rows,
                 colors={0: '36'}, wrap_columns=(2,))
    print("\nFormat compatibility is catalogue metadata, not a test of every archive.")
    if name:
        _heading("%s: archive sources (shared URLs grouped)" % name)
        for version in versions:
            print("  " + ', '.join(version['releases']))
            for url in version['urls']:
                print("    " + url)
    else:
        print("Filter by bundle: mhcflurry downloads releases models_class1_presentation")
    print("Code 2.1.5, 2.2.0 and 2.2.1 used the 2020 weights in catalogue 2.2.0.")
    print("Archive storage tags such as pre-2.0 are historical locations, not code requirements.")


def info_subcommand(args):
    """Show resolved configuration or a bundle's purpose, sources and usage."""
    release = resolve_release(args.release)
    verify_files = bool(getattr(args, "verify", False))
    if args.download_name:
        record = _find_download(release, args.download_name, verify_files=verify_files)
        record['versions'] = get_bundle_versions(record['name'])
        fetch = "mhcflurry downloads fetch %s --release %s" % (record['name'], release)
        record['fetch_command'] = fetch
        if record['name'] == 'models_class1_presentation':
            record['predict_command'] = 'mhcflurry predict INPUT.csv --model-release ' + release
        if args.json:
            print(json.dumps(record, indent=2))
            return
        print(_style(record['name'] + " — " + release, '1;36'))
        print(record['description'])
        print(('Standalone bundle status: ' if record['name'] in _PRESENTATION_COMPONENT_DIRS
               else 'Status: ') + record['status'])
        print("Directory: " + record['path'])
        if record.get('layout'):
            print("Storage layout: " + record['layout'])
        if record.get('fetched_at'):
            print("Fetched at: " + record['fetched_at'])
        if record.get('recorded_sha256'):
            size_part = (
                " (%d bytes)" % record['archive_size']
                if record.get('archive_size') is not None
                else ""
            )
            print("Observed archive SHA-256: %s%s" % (record['recorded_sha256'], size_part))
        print("\nFetch: " + fetch)
        print("Locate: mhcflurry downloads path %s --release %s" % (record['name'], release))
        if 'predict_command' in record:
            print("Predict: " + record['predict_command'])
        elif record['name'] == 'models_class1_pan':
            print("Predict: mhcflurry predict INPUT.csv --affinity-only --models " +
                  quote(os.path.join(record['path'], 'models.combined')))
        if 'presentation_components' in record:
            _heading('Components inside presentation bundle')
            _print_presentation_components(record['presentation_components'])
            if record['name'] == 'models_class1_pan':
                print('  Default affinity loading can fall back to the default presentation bundle')
                print('  when its standalone path is absent and no affinity-path override is set.')
                print('  Browsing does not select weights; use --models DIR or --model-release RELEASE.')
            elif record['name'] == 'models_class1_processing':
                print('  Default processing loading can fall back to the default presentation bundle')
                print('  (with-flanks component) when its standalone path is absent and no')
                print('  processing-path override is set. Pass a path explicitly for without-flanks.')
            else:
                print('  Presentation loading uses these embedded components.')
            source = record['presentation_components'][0]['source_matches']
            print('  Presentation source URLs: ' + (
                'match catalogue' if source is True else
                'differ from catalogue' if source is False else 'unknown'))
        _heading("Versions / archive sources (shared URLs grouped)")
        for version in record['versions']:
            print("  " + ', '.join(version['releases']))
            for url in version['urls']:
                print("    " + url)
        if verify_files:
            if record['verified']:
                print("\nFull file verification passed against trusted catalogue checksums.")
            else:
                print("\nFull verification was not established against trusted catalogue checksums.")
        else:
            print("\nSource match compares recorded URLs, not file integrity (pass --verify to hash files).")
        return

    defaults = dict(
        presentation=get_default_class1_presentation_models_dir(test_exists=False),
        affinity=get_default_class1_models_dir(test_exists=False),
        processing=get_default_class1_processing_models_dir(test_exists=False))
    config = dict(
        code_version=__version__,
        default_release=get_downloads_metadata()['current-release'],
        configured_release=get_current_release(),
        catalogue_release=release,
        downloads_dir=os.path.abspath(get_downloads_dir(release)),
        default_model_paths={key: dict(path=os.path.abspath(value), exists=os.path.exists(value))
                             for key, value in defaults.items()},
        default_presentation_components=_presentation_components(defaults['presentation']),
        environment_overrides={key: os.environ.get(key) or None for key in ENVIRONMENT_VARIABLES},
        downloads=_download_records(release, verify_files=verify_files))
    if args.json:
        print(json.dumps(config, indent=2))
        return
    print("Download catalogue %s (code %s)" % (release, __version__))
    _print_downloads(config['downloads'])
    _heading("Resolved configuration")
    print("  Code version:       " + __version__)
    print("  Default weights:    " + config['default_release'])
    print("  Active catalogue:   " + (get_current_release() or 'custom unversioned directory'))
    print("  Browsing catalogue: " + release)
    print("  Downloads directory: " + config['downloads_dir'])
    if args.verbose:
        _heading("Default prediction paths (before --models / --model-release)")
        for kind, item in config['default_model_paths'].items():
            print("  %s: %s [%s]" % (kind, item['path'], 'exists' if item['exists'] else 'not installed'))
        print('  Affinity and processing can fall back to the default presentation bundle when')
        print('  their default standalone paths are absent and no path override is set.')
        _heading('Components inside default presentation predictor')
        _print_presentation_components(config['default_presentation_components'])
    overrides = {key: value for key, value in config['environment_overrides'].items()
                 if value or args.verbose}
    if overrides:
        _heading("Environment variables (optional overrides)")
        for key, value in overrides.items():
            print("  %s: %s" % (key, quote(value) if value else 'unset; using defaults'))
    else:
        print("  Environment overrides: none")
    if not args.verbose:
        print("\nFull paths and overrides: mhcflurry downloads --verbose info")


def _dir_size_bytes(path):
    if not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                continue
    return total


def _installed_release_dirs():
    """Return installed known-release directories under the configured data root."""
    current = get_current_release()
    if current is None:
        return []
    root = os.path.dirname(os.path.abspath(get_downloads_dir()))
    if not os.path.isdir(root):
        return []
    known_releases = set(get_downloads_metadata()['releases'])
    entries = []
    for name in sorted(os.listdir(root)):
        if name not in known_releases:
            continue
        full = os.path.join(root, name)
        if os.path.islink(full) or not os.path.isdir(full):
            continue
        entries.append(dict(
            release=name,
            path=full,
            size_bytes=_dir_size_bytes(full),
            is_current=(name == current),
        ))
    return entries


def dirs_subcommand(args):
    """Show resolved data locations, disk usage, and offline state (read-only)."""
    import datacache

    loc = resolve_data_location()
    active_dir = os.path.abspath(get_downloads_dir())
    custom_dir = os.environ.get("MHCFLURRY_DOWNLOADS_DIR") or None
    current_release = get_current_release()
    installed_releases = _installed_release_dirs()
    offline_src = datacache.offline_source()

    data_root_exists = os.path.exists(loc.path)
    data_root_bytes = _dir_size_bytes(loc.path)
    active_exists = os.path.exists(active_dir)
    active_bytes = _dir_size_bytes(active_dir)

    payload = dict(
        package="mhcflurry",
        version=__version__,
        offline=offline_src is not None,
        offline_source=offline_src,
        configured_release=current_release,
        custom_downloads_dir=custom_dir,
        data_root=dict(
            path=os.path.abspath(loc.path),
            origin=loc.origin,
            detail=loc.detail,
            exists=data_root_exists,
            size_bytes=data_root_bytes,
        ),
        active_downloads_dir=dict(
            path=active_dir,
            exists=active_exists,
            size_bytes=active_bytes,
        ),
        installed_releases=installed_releases,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    origin_text = (
        "%s (%s)" % (loc.origin, loc.detail) if loc.detail else loc.origin
    )
    print("mhcflurry data locations (code %s)" % __version__)
    print("  data_root        : %s [%s; %s]" % (
        payload["data_root"]["path"],
        origin_text,
        datacache.format_bytes(data_root_bytes),
    ))
    print("  active_downloads : %s [%s]" % (
        active_dir,
        datacache.format_bytes(active_bytes) if active_exists else "not created",
    ))
    if custom_dir:
        print("  custom_override  : MHCFLURRY_DOWNLOADS_DIR=%s" % quote(custom_dir))
    if installed_releases:
        print("  installed_releases:")
        for entry in installed_releases:
            tag = " (current)" if entry["is_current"] else ""
            print("    %-10s %10s  %s%s" % (
                entry["release"],
                datacache.format_bytes(entry["size_bytes"]),
                entry["path"],
                tag,
            ))
    if offline_src:
        print("  offline          : enabled (%s)" % offline_src)
    else:
        print("  offline          : disabled")


def prune_subcommand(args):
    """Remove non-current downloaded release directories (dry-run by default)."""
    import datacache

    dry_run = not args.yes
    current = get_current_release()
    if current is None:
        payload = dict(
            dry_run=dry_run,
            configured_release=None,
            data_root=None,
            kept=[],
            candidates=[],
            deleted=[],
            reclaimable_bytes=0,
            freed_bytes=0,
            note="MHCFLURRY_DOWNLOADS_DIR points to a custom unversioned directory; skipping prune.",
        )
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(payload["note"])
        return

    keep_set = {current}
    for rel in (args.keep_release or ()):
        keep_set.add(resolve_release(rel))

    entries = _installed_release_dirs()
    kept = [e for e in entries if e["release"] in keep_set]
    candidates = [e for e in entries if e["release"] not in keep_set]
    reclaimable_bytes = sum(e["size_bytes"] for e in candidates)
    deleted = []
    if not dry_run:
        for entry in candidates:
            rmtree(entry["path"])
            deleted.append(entry)
    freed_bytes = sum(e["size_bytes"] for e in deleted)
    root = os.path.dirname(os.path.abspath(get_downloads_dir()))
    payload = dict(
        dry_run=dry_run,
        configured_release=current,
        data_root=root,
        kept=kept,
        candidates=candidates,
        deleted=deleted,
        reclaimable_bytes=reclaimable_bytes,
        freed_bytes=freed_bytes,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print("mhcflurry downloads root: %s" % root)
    if not entries:
        print("No installed release directories found; nothing to prune.")
        return
    for entry in entries:
        action = "DELETE" if entry in candidates else "keep  "
        marker = " (current)" if entry["is_current"] else ""
        print("  %s  %-10s %10s  %s%s" % (
            action,
            entry["release"],
            datacache.format_bytes(entry["size_bytes"]),
            entry["path"],
            marker,
        ))
    if not candidates:
        print("Nothing to prune.")
        return
    if dry_run:
        print(
            "\nDry run — would free %s across %d release dir(s). "
            "Re-run with --yes to delete."
            % (datacache.format_bytes(reclaimable_bytes), len(candidates))
        )
        return
    print(
        "\nDeleted %d release dir(s), freed %s."
        % (len(deleted), datacache.format_bytes(freed_bytes))
    )


def path_subcommand(args):
    """Print a bundle directory or the download root for the selected release."""
    release = resolve_release(args.release)
    if args.download_name:
        _find_download(release, args.download_name)
        path = get_path(args.download_name, release=release)
    else:
        path = get_downloads_dir(release)
    if getattr(args, "json", False):
        print(json.dumps(dict(
            release=release,
            download=args.download_name or None,
            path=path,
        ), indent=2))
    else:
        print(path)


def url_subcommand(args):
    """Print every archive URL for a bundle in the selected release."""
    release = resolve_release(args.release)
    print("\n".join(_find_download(release, args.download_name)['urls']))
