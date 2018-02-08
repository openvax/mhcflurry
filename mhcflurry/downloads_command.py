'''
Download MHCflurry released datasets and trained models.

Examples

Fetch the default downloads:
    $ mhcflurry-downloads fetch

Fetch a specific download:
    $ mhcflurry-downloads fetch data_kim2014

Get the path to a download:
    $ mhcflurry-downloads path data_kim2014

Summarize available and fetched downloads:
    $ mhcflurry-downloads info
'''
from __future__ import (
    print_function,
    division,
    absolute_import,
)
import sys
import argparse
import logging
import os
from pipes import quote
import errno
import tarfile
from tempfile import mkstemp
from tqdm import tqdm
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

from .downloads import (
    get_current_release,
    get_current_release_downloads,
    get_downloads_dir,
    get_path,
    ENVIRONMENT_VARIABLES)

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

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

parser_info = subparsers.add_parser('info')

parser_path = subparsers.add_parser('path')
parser_path.add_argument(
    "download_name",
    nargs="?",
    default='')


def run(argv=sys.argv[1:]):
    args = parser.parse_args(argv)
    if not args.quiet:
        logging.basicConfig(level="INFO")
    if args.verbose:
        logging.basicConfig(level="DEBUG")

    command_functions = {
        "fetch": fetch_subcommand,
        "info": info_subcommand,
        "path": path_subcommand,
        None: lambda args: parser.print_help(),
    }
    command_functions[args.subparser_name](args)


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

    downloads = get_current_release_downloads()
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
                "*" * 40)
                % (name, 'rm -rf ' + quote(get_path(name))))
        if not info['downloaded'] and (name in args.download_name or default):
            items_to_fetch.add(name)

    mkdir_p(get_downloads_dir())

    qprint("Fetching %d/%d downloads from release %s" % (
        len(items_to_fetch), len(downloads), args.release))
    format_string = "%-40s  %-20s   %-20s  %-20s "
    qprint(format_string % (
        "DOWNLOAD NAME", "ALREADY DOWNLOADED?", "WILL DOWNLOAD NOW?", "URL"))

    for (item, info) in downloads.items():
        qprint(format_string % (
            item,
            yes_no(info['downloaded']),
            yes_no(item in items_to_fetch),
            info['metadata']["url"]))

    # TODO: may want to extract into somewhere temporary and then rename to
    # avoid making an incomplete extract if the process is killed.
    for item in items_to_fetch:
        metadata = downloads[item]['metadata']
        (temp_fd, target_path) = mkstemp()
        try:
            qprint("Downloading: %s" % metadata['url'])
            urlretrieve(
                metadata['url'],
                target_path,
                reporthook=TqdmUpTo(
                    unit='B', unit_scale=True, miniters=1).update_to)
            qprint("Downloaded to: %s" % quote(target_path))

            tar = tarfile.open(target_path, 'r:bz2')
            names = tar.getnames()
            logging.debug("Extracting: %s" % names)
            bad_names = [
                n for n in names
                if n.strip().startswith("/") or n.strip().startswith("..")
            ]
            if bad_names:
                raise RuntimeError(
                    "Archive has suspicious names: %s" % bad_names)
            result_dir = get_path(item, test_exists=False)
            os.mkdir(result_dir)

            for member in tqdm(tar.getmembers(), desc='Extracting'):
                tar.extractall(path=result_dir, members=[member])
            tar.close()
            qprint("Extracted %d files to: %s" % (
                len(names), quote(result_dir)))
        finally:
            if not args.keep:
                os.remove(target_path)


def info_subcommand(args):
    print("Environment variables")
    for variable in ENVIRONMENT_VARIABLES:
        value = os.environ.get(variable)
        if value:
            print('  %-35s = %s' % (variable, quote(value)))
        else:
            print("  %-35s [unset or empty]" % variable)

    print("")
    print("Configuration")

    def exists_string(path):
        return (
            "exists" if os.path.exists(path) else "does not exist")

    items = [
        ("current release", get_current_release(), ""),
        ("downloads dir",
            get_downloads_dir(),
            "[%s]" % exists_string(get_downloads_dir())),
    ]
    for (key, value, extra) in items:
        print("  %-35s = %-20s %s" % (key, quote(value), extra))

    print("")

    downloads = get_current_release_downloads()

    format_string = "%-40s  %-12s  %-20s "
    print(format_string % ("DOWNLOAD NAME", "DOWNLOADED?", "URL"))

    for (item, info) in downloads.items():
        print(format_string % (
            item,
            yes_no(info['downloaded']),
            info['metadata']["url"]))


def path_subcommand(args):
    print(get_path(args.download_name))
