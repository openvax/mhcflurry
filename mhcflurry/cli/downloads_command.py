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
import errno
import tarfile
import textwrap
from itertools import zip_longest
from shutil import copyfileobj
from shutil import get_terminal_size
from tempfile import NamedTemporaryFile
from tqdm import tqdm

import posixpath
import csv

try:
    from urllib.request import urlretrieve
    from urllib.parse import urlsplit
except ImportError:
    from urllib import urlretrieve
    from urlparse import urlsplit

from ..downloads import (
    get_current_release,
    get_downloads_metadata,
    get_bundle_description,
    get_bundle_versions,
    get_download_urls,
    resolve_release,
    get_release_downloads,
    get_downloads_dir,
    get_default_class1_models_dir,
    get_default_class1_presentation_models_dir,
    get_default_class1_processing_models_dir,
    get_path,
    ENVIRONMENT_VARIABLES)
from ..version import __version__
from .help import HelpArgumentParser, color_enabled

tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

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
    help="Don't delete archives after they are extracted")
parser_fetch.add_argument(
    "--release",
    default=get_current_release(),
    help="Release to download. Default: %(default)s")
parser_fetch.add_argument(
    "--already-downloaded-dir",
    metavar="DIR",
    help="Don't download files, get them from DIR")

parser_info = subparsers.add_parser('info')
parser_info.add_argument('download_name', nargs='?', metavar='DOWNLOAD')
parser_info.add_argument('--json', action='store_true', help='Print machine-readable JSON')

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

parser_url = subparsers.add_parser('url')
parser_url.add_argument(
    "download_name")

for subparser in (parser_info, parser_list, parser_path, parser_url):
    subparser.add_argument('--release', metavar='RELEASE',
                          help='Catalogue release (default: configured release)')


def run(argv=sys.argv[1:]):
    args = parser.parse_args(argv)
    logging.basicConfig(level=(logging.DEBUG if args.verbose else
                               logging.WARNING if args.quiet else logging.INFO))

    command_functions = {
        "fetch": fetch_subcommand,
        "info": info_subcommand,
        "list": list_subcommand,
        "releases": releases_subcommand,
        "path": path_subcommand,
        "url": url_subcommand,
        None: lambda args: parser.print_help(),
    }
    try:
        command_functions[args.subparser_name](args)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))


def mkdir_p(path):
    """
    Make directories as needed, similar to mkdir -p in a shell.

    From:
    http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def yes_no(boolean):
    return "YES" if boolean else "NO"


def suspicious_tar_member(member):
    """Return whether a tar member should not be extracted."""
    name = member.name.strip()
    if not name or posixpath.isabs(name):
        return True
    if ".." in name.split("/"):
        return True
    # Keep a normalized guard for unusual path spellings that still resolve
    # above the extraction directory.
    normalized = posixpath.normpath(name)
    if normalized == ".." or normalized.startswith("../"):
        return True
    return member.issym() or member.islnk()


# For progress bar on download. See https://pypi.python.org/pypi/tqdm
class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def fetch_subcommand(args):
    def qprint(msg):
        if not args.quiet:
            print(msg)

    if not args.release:
        raise RuntimeError(
            "No release defined. This can happen when you are specifying "
            "a custom models directory. Specify --release to indicate "
            "the release to download.")

    downloads = get_release_downloads(args.release)
    invalid_download_names = set(
        item for item in args.download_name if item not in downloads)
    if invalid_download_names:
        raise ValueError("Unknown download(s): %s. Valid downloads are: %s" % (
            ', '.join(invalid_download_names), ', '.join(downloads)))

    items_to_fetch = set()
    for (name, info) in downloads.items():
        default = not args.download_name and info['metadata']['default']
        if name in args.download_name and info['downloaded']:
            print((
                "*" * 40 +
                "\nThe requested download '%s' has already been downloaded. "
                "To re-download this data, first run: \n\t%s\nin a shell "
                "and then re-run this command.\n" +
                "*" * 40) % (name, 'rm -rf ' + quote(
                    get_path(name, release=args.release))))
        if not info['downloaded'] and (name in args.download_name or default):
            items_to_fetch.add(name)

    mkdir_p(get_downloads_dir(args.release))

    qprint("Fetching %d/%d downloads from release %s" % (
        len(items_to_fetch), len(downloads), args.release))
    format_string = "%-40s  %-20s   %-20s  %-20s "
    qprint(format_string % (
        "DOWNLOAD NAME", "ALREADY DOWNLOADED?", "WILL DOWNLOAD NOW?", "URL"))

    for (item, info) in downloads.items():
        urls = (
            [info['metadata']["url"]]
            if "url" in info['metadata']
            else info['metadata']["part_urls"])
        url_description = urls[0]
        if len(urls) > 1:
            url_description += " + %d more parts" % (len(urls) - 1)

        qprint(format_string % (
            item,
            yes_no(info['downloaded']),
            yes_no(item in items_to_fetch),
            url_description))

    # TODO: may want to extract into somewhere temporary and then rename to
    # avoid making an incomplete extract if the process is killed.
    for item in items_to_fetch:
        metadata = downloads[item]['metadata']
        urls = (
            [metadata["url"]] if "url" in metadata else metadata["part_urls"])
        temp = NamedTemporaryFile(delete=False, suffix=".tar.bz2")
        try:
            for (url_num, url) in enumerate(urls):
                delete_downloaded = True
                if args.already_downloaded_dir:
                    filename = posixpath.basename(urlsplit(url).path)
                    downloaded_path = os.path.join(
                        args.already_downloaded_dir, filename)
                    delete_downloaded = False
                else:
                    qprint("Downloading [part %d/%d]: %s" % (
                        url_num + 1, len(urls), url))
                    (downloaded_path, _) = urlretrieve(
                        url,
                        temp.name if len(urls) == 1 else None,
                        reporthook=TqdmUpTo(
                            unit='B', unit_scale=True, miniters=1,
                            disable=args.quiet).update_to)
                    qprint("Downloaded to: %s" % quote(downloaded_path))

                if downloaded_path != temp.name:
                    qprint("Copying to: %s" % temp.name)
                    with open(downloaded_path, "rb") as fd:
                        copyfileobj(fd, temp, length=64*1024*1024)
                    if delete_downloaded:
                        os.remove(downloaded_path)

            temp.close()
            tar = tarfile.open(temp.name, 'r:bz2')
            members = tar.getmembers()
            names = [member.name for member in members]
            logging.debug("Extracting: %s" % names)
            bad_names = [
                member.name for member in members
                if suspicious_tar_member(member)
            ]
            if bad_names:
                raise RuntimeError(
                    "Archive has suspicious names: %s" % bad_names)
            result_dir = get_path(item, test_exists=False, release=args.release)
            os.mkdir(result_dir)

            for member in tqdm(members, desc='Extracting', disable=args.quiet):
                tar.extractall(path=result_dir, members=[member])
            tar.close()

            # Save URLs that were used for this download.
            with open(os.path.join(result_dir, "DOWNLOAD_INFO.csv"), "w", newline="") as fd:
                writer = csv.writer(fd)
                writer.writerow(["url"])
                writer.writerows([url] for url in urls)
            qprint("Extracted %d files to: %s" % (
                len(names), quote(result_dir)))
        finally:
            if not args.keep:
                os.remove(temp.name)


def _download_records(release, kind="all"):
    records = []
    for name, info in get_release_downloads(release).items():
        description = get_bundle_description(name)
        if kind != "all" and description["kind"] != kind:
            continue
        status = (
            "not installed" if not info["downloaded"] else
            "installed; source unknown" if info["up_to_date"] is None else
            "installed; source matches" if info["up_to_date"] else
            "installed; source differs")
        records.append(dict(
            name=name, release=release, **description,
            status=status, downloaded=info["downloaded"],
            source_matches=info["up_to_date"],
            path=os.path.abspath(get_path(name, test_exists=False, release=release)),
            urls=get_download_urls(info["metadata"]),
            default=info["metadata"].get("default", False)))
    return records


def _find_download(release, name):
    records = _download_records(release)
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
    return "32"


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
    """Describe distinct archive sources and installed catalogue directories."""
    versions = get_bundle_versions(name)
    installed = []
    custom = get_current_release() is None
    custom_status = None
    for version in versions:
        for release in version['releases']:
            if release not in releases:
                releases[release] = get_release_downloads(release)
            info = releases[release][name]
            if not info['downloaded']:
                continue
            marker = ("" if info['up_to_date'] is True else
                      "?" if info['up_to_date'] is None else "!")
            if custom:
                custom_status = marker
                if not marker:
                    return (versions[0]['releases'][0],
                            ', '.join(v['releases'][0] for v in versions[1:]) or '—',
                            'custom: ' + release)
            else:
                installed.append(release + marker)
    if custom and custom_status is not None:
        installed = ['custom' + custom_status]
    return (versions[0]['releases'][0],
            ', '.join(v['releases'][0] for v in versions[1:]) or '—',
            ', '.join(installed) or '—')


def _print_downloads(records):
    primary = [record for record in records if record['group'] == 'Prediction models']
    historical = [record for record in records if record['group'] != 'Prediction models']
    if primary:
        _heading('Prediction models — latest weights and available versions')
        releases = {}
        rows = [(record['name'] + (' (affinity)' if record['name'] == 'models_class1_pan' else ''),
                 *_model_versions(record['name'], releases))
                for record in primary]
        _print_table(('MODEL', 'LATEST', 'OTHER VERSIONS', 'INSTALLED'), rows,
                     colors={1: '36', 3: _status_color}, wrap_columns=(2, 3))
        if any(record['name'] == 'models_class1_presentation' for record in primary):
            print('The presentation bundle includes its own affinity and processing models.')
        print("Shared archive aliases are grouped under one version; 'releases NAME' lists all.")
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
    print("\nLocal status checks directories and recorded source URLs, not file integrity.")
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
    if args.download_name:
        record = _find_download(release, args.download_name)
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
        print("Status: " + record['status'])
        print("Directory: " + record['path'])
        print("\nFetch: " + fetch)
        print("Locate: mhcflurry downloads path %s --release %s" % (record['name'], release))
        if 'predict_command' in record:
            print("Predict: " + record['predict_command'])
        elif record['name'] == 'models_class1_pan':
            print("Predict: mhcflurry predict INPUT.csv --affinity-only --models " +
                  quote(os.path.join(record['path'], 'models.combined')))
        _heading("Versions / archive sources (shared URLs grouped)")
        for version in record['versions']:
            print("  " + ', '.join(version['releases']))
            for url in version['urls']:
                print("    " + url)
        print("\nSource match compares recorded URLs, not file integrity.")
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
        environment_overrides={key: os.environ.get(key) or None for key in ENVIRONMENT_VARIABLES},
        downloads=_download_records(release))
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


def path_subcommand(args):
    """Print a bundle directory or the download root for the selected release."""
    release = resolve_release(args.release)
    if args.download_name:
        _find_download(release, args.download_name)
        print(get_path(args.download_name, release=release))
    else:
        print(get_downloads_dir(release))


def url_subcommand(args):
    """Print every archive URL for a bundle in the selected release."""
    release = resolve_release(args.release)
    print("\n".join(_find_download(release, args.download_name)['urls']))
