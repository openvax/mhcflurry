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
Manage local downloaded data.
"""

import logging
import yaml
from os.path import join, exists, dirname, abspath
from os import environ
from shlex import quote
from importlib.resources import files
from collections import OrderedDict
from appdirs import user_data_dir

import csv

ENVIRONMENT_VARIABLES = [
    "MHCFLURRY_DATA_DIR",
    "MHCFLURRY_DOWNLOADS_CURRENT_RELEASE",
    "MHCFLURRY_DOWNLOADS_DIR",
    "MHCFLURRY_DEFAULT_CLASS1_MODELS",
    "MHCFLURRY_DEFAULT_CLASS1_PRESENTATION_MODELS_DIR",
    "MHCFLURRY_DEFAULT_CLASS1_PROCESSING_MODELS_DIR",
]

_DOWNLOADS_DIR = None
_CURRENT_RELEASE = None
_METADATA = None
_MHCFLURRY_DEFAULT_CLASS1_MODELS_DIR = environ.get(
    "MHCFLURRY_DEFAULT_CLASS1_MODELS")
_MHCFLURRY_DEFAULT_CLASS1_PRESENTATION_MODELS_DIR = environ.get(
    "MHCFLURRY_DEFAULT_CLASS1_PRESENTATION_MODELS_DIR")
_MHCFLURRY_DEFAULT_CLASS1_PROCESSING_MODELS_DIR = environ.get(
    "MHCFLURRY_DEFAULT_CLASS1_PROCESSING_MODELS_DIR")


def get_downloads_dir(release=None):
    """
    Return the download directory for a release, respecting custom overrides.
    """
    if release is None or _CURRENT_RELEASE is None:
        return _DOWNLOADS_DIR
    return join(dirname(_DOWNLOADS_DIR), release)


def get_current_release():
    """
    Return the current downloaded data release
    """
    return _CURRENT_RELEASE


def get_downloads_metadata():
    """
    Return the contents of downloads.yml as a dict
    """
    global _METADATA
    if _METADATA is None:
        _METADATA = yaml.safe_load(
            files("mhcflurry").joinpath("downloads.yml").read_text()
        )
    return _METADATA


def resolve_release(release=None):
    """Validate a catalogue identifier, defaulting to the configured release."""
    metadata = get_downloads_metadata()
    release = release or get_current_release() or metadata['current-release']
    if release not in metadata['releases']:
        raise ValueError(
            "Unknown download release %r. Run 'mhcflurry downloads releases' "
            "for valid identifiers (these are separate from package versions)."
            % release)
    return release


def get_download_urls(metadata):
    """Return ordered archive URLs, including every part of split downloads."""
    return [metadata['url']] if 'url' in metadata else metadata['part_urls']


def get_bundle_description(name):
    """Return the purpose and category of a named download."""
    try:
        return get_downloads_metadata()['bundles'][name]
    except KeyError:
        raise ValueError(
            "Unknown download %r. Run 'mhcflurry downloads list'." % name) from None


def get_bundle_versions(name):
    """Group catalogue entries for one bundle by identical archive URLs.

    URL equality identifies shared sources, not verified content equality.
    """
    get_bundle_description(name)
    versions = OrderedDict()
    for release, info in get_downloads_metadata()['releases'].items():
        for download in info['downloads']:
            if download['name'] == name:
                urls = tuple(get_download_urls(download))
                versions.setdefault(urls, []).append(release)
    return [dict(releases=releases, urls=list(urls))
            for urls, releases in versions.items()]


def get_model_release_dir(release):
    """Resolve explicitly selected presentation weights without model imports.

    Explicit selection overrides default-model environment variables. Custom,
    unversioned download roots must record matching source URLs so a release
    selector cannot silently load a different set of weights.
    """
    release = resolve_release(release)
    metadata = get_downloads_metadata()
    if (metadata['releases'][release]['compatibility-version'] !=
            metadata['current-compatibility-version']):
        raise ValueError("Download release %s uses an incompatible model format." % release)
    name = 'models_class1_presentation'
    available = get_release_downloads(release)
    if name not in available:
        raise ValueError(
            "Release %s has no presentation bundle. Run 'mhcflurry downloads "
            "releases models_class1_presentation'. For other predictors, use --models DIR."
            % release)
    info = available[name]
    path = get_path(name, 'models', release=release)
    if info['up_to_date'] is False or (
            get_current_release() is None and info['up_to_date'] is not True):
        raise ValueError(
            "Cannot confirm that %s contains release %s: recorded source URLs "
            "are missing or different. Use a versioned MHCFLURRY_DATA_DIR, "
            "or select your local weights explicitly with --models DIR."
            % (quote(path), release))
    return abspath(path)


def get_default_class1_models_dir(test_exists=True):
    """
    Return the absolute path to the default class1 models dir.

    If environment variable MHCFLURRY_DEFAULT_CLASS1_MODELS is set to an
    absolute path, return that path. If it's set to a relative path (i.e. does
    not start with /) then return that path taken to be relative to the mhcflurry
    downloads dir.

    If environment variable MHCFLURRY_DEFAULT_CLASS1_MODELS is NOT set,
    then return the pan-allele affinity path in the "models_class1_pan" download.

    Parameters
    ----------

    test_exists : boolean, optional
        Whether to raise an exception if the path does not exist

    Returns
    -------
    string : absolute path
    """
    if _MHCFLURRY_DEFAULT_CLASS1_MODELS_DIR:
        result = join(get_downloads_dir(), _MHCFLURRY_DEFAULT_CLASS1_MODELS_DIR)
        if test_exists and not exists(result):
            raise IOError("No such directory: %s" % result)
        return result
    return get_path(
        "models_class1_pan", "models.combined", test_exists=test_exists)


def get_default_class1_presentation_models_dir(test_exists=True):
    """
    Return the absolute path to the default class1 presentation models dir.

    See `get_default_class1_models_dir`.

    If environment variable MHCFLURRY_DEFAULT_CLASS1_PRESENTATION_MODELS_DIR is set
    to an absolute path, return that path. If it's set to a relative path (does
    not start with /) then return that path taken to be relative to the mhcflurry
    downloads dir.

    Parameters
    ----------

    test_exists : boolean, optional
        Whether to raise an exception if the path does not exist

    Returns
    -------
    string : absolute path
    """
    if _MHCFLURRY_DEFAULT_CLASS1_PRESENTATION_MODELS_DIR:
        result = join(
            get_downloads_dir(),
            _MHCFLURRY_DEFAULT_CLASS1_PRESENTATION_MODELS_DIR)
        if test_exists and not exists(result):
            raise IOError("No such directory: %s" % result)
        return result
    return get_path(
        "models_class1_presentation", "models", test_exists=test_exists)


def get_default_class1_processing_models_dir(test_exists=True):
    """
    Return the absolute path to the default class1 processing models dir.

    See `get_default_class1_models_dir`.

    If environment variable MHCFLURRY_DEFAULT_CLASS1_PROCESSING_MODELS_DIR is set
    to an absolute path, return that path. If it's set to a relative path (does
    not start with /) then return that path taken to be relative to the mhcflurry
    downloads dir.

    Parameters
    ----------

    test_exists : boolean, optional
        Whether to raise an exception if the path does not exist

    Returns
    -------
    string : absolute path
    """
    if _MHCFLURRY_DEFAULT_CLASS1_PROCESSING_MODELS_DIR:
        result = join(
            get_downloads_dir(),
            _MHCFLURRY_DEFAULT_CLASS1_PROCESSING_MODELS_DIR)
        if test_exists and not exists(result):
            raise IOError("No such directory: %s" % result)
        return result

    # Default to the 'with flanks' model variant.
    return get_path(
        "models_class1_processing", "models.selected.with_flanks", test_exists=test_exists)


def get_release_downloads(release):
    """
    Return a dict of all available downloads in a release.

    Parameters
    ----------
    release : string
        Release whose download metadata to return.

    Returns
    -------
    collections.OrderedDict
        Download names mapped to dictionaries with three entries:
        ``downloaded`` (bool), whether the local directory exists;
        ``metadata`` (dict), catalogue metadata such as URLs; and
        ``up_to_date`` (bool or None), whether the recorded download URLs
        match the catalogue, or None when unknown. This does not verify
        the integrity of installed files.
    """
    downloads = (
        get_downloads_metadata()
        ['releases']
        [resolve_release(release)]
        ['downloads'])

    def up_to_date(dir, urls):
        try:
            with open(join(dir, "DOWNLOAD_INFO.csv"), newline="") as fd:
                reader = csv.DictReader(fd)
                if not reader.fieldnames or "url" not in reader.fieldnames:
                    return None
                return [row["url"] for row in reader] == list(urls)
        except (OSError, csv.Error, UnicodeError):
            return None

    return OrderedDict(
        (download["name"], {
            'downloaded': exists(join(get_downloads_dir(release), download["name"])),
            'up_to_date': up_to_date(
                join(get_downloads_dir(release), download["name"]),
                get_download_urls(download)),
            'metadata': download,
        }) for download in downloads
    )


def get_current_release_downloads():
    """Return a dict of all available downloads in the current release."""
    return get_release_downloads(get_current_release())


def get_path(download_name, filename='', test_exists=True, release=None):
    """
    Get the local path to a file in a MHCflurry download

    Parameters
    ----------
    download_name : string

    filename : string
        Relative path within the download to the file of interest

    test_exists : boolean
        If True (default) throw an error telling the user how to download the
        data if the file does not exist

    release : string, optional
        Requested release; defaults to the configured current release.

    Returns
    -------
    string giving local absolute path
    """
    assert '/' not in download_name, "Invalid download: %s" % download_name
    path = join(get_downloads_dir(release), download_name, filename)
    if test_exists and not exists(path):
        raise RuntimeError(
            "Missing MHCflurry downloadable file: %s. "
            "To download this data, run:\n\tmhcflurry downloads fetch %s%s\n"
            "in a shell."
            % (quote(path), download_name,
               " --release " + quote(release) if release else ""))
    return path


def configure():
    """
    Setup various global variables based on environment variables.
    """
    global _DOWNLOADS_DIR
    global _CURRENT_RELEASE

    _CURRENT_RELEASE = None
    _DOWNLOADS_DIR = environ.get("MHCFLURRY_DOWNLOADS_DIR")
    if not _DOWNLOADS_DIR:
        metadata = get_downloads_metadata()
        _CURRENT_RELEASE = environ.get("MHCFLURRY_DOWNLOADS_CURRENT_RELEASE")
        if not _CURRENT_RELEASE:
            _CURRENT_RELEASE = metadata['current-release']

        current_release_compatability = (
            metadata["releases"].get(_CURRENT_RELEASE, {}).get("compatibility-version"))
        current_compatability = metadata["current-compatibility-version"]
        if (current_release_compatability is not None and
                current_release_compatability != current_compatability):
            logging.warning(
                "The specified downloads are not compatible with this version "
                "of the MHCflurry codebase. Downloads: release %s, "
                "compatability version: %d. Code compatability version: %d",
                _CURRENT_RELEASE,
                current_release_compatability,
                current_compatability)

        data_dir = environ.get("MHCFLURRY_DATA_DIR")
        if not data_dir:
            # increase the version every time we make a breaking change in
            # how the data is organized. For changes to e.g. just model
            # serialization, the downloads release numbers should be used.
            data_dir = user_data_dir("mhcflurry", version="4")
        _DOWNLOADS_DIR = join(data_dir, _CURRENT_RELEASE)

    logging.debug("Configured MHCFLURRY_DOWNLOADS_DIR: %s", _DOWNLOADS_DIR)


configure()
