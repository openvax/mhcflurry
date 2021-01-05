"""
Manage local downloaded data.
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
)
import logging
import yaml
from os.path import join, exists
from os import environ
from pipes import quote
from collections import OrderedDict
from appdirs import user_data_dir
from pkg_resources import resource_string

import pandas

ENVIRONMENT_VARIABLES = [
    "MHCFLURRY_DATA_DIR",
    "MHCFLURRY_DOWNLOADS_CURRENT_RELEASE",
    "MHCFLURRY_DOWNLOADS_DIR",
    "MHCFLURRY_DEFAULT_CLASS1_MODELS"
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


def get_downloads_dir():
    """
    Return the path to local downloaded data
    """
    return _DOWNLOADS_DIR


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
        _METADATA = yaml.safe_load(resource_string(__name__, "downloads.yml"))
    return _METADATA


def get_default_class1_models_dir(test_exists=True):
    """
    Return the absolute path to the default class1 models dir.

    If environment variable MHCFLURRY_DEFAULT_CLASS1_MODELS is set to an
    absolute path, return that path. If it's set to a relative path (i.e. does
    not start with /) then return that path taken to be relative to the mhcflurry
    downloads dir.

    If environment variable MHCFLURRY_DEFAULT_CLASS1_MODELS is NOT set,
    then return the path to downloaded models in the "models_class1" download.

    Parameters
    ----------

    test_exists : boolean, optional
        Whether to raise an exception of the path does not exist

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

    If environment variable MHCFLURRY_DEFAULT_CLASS1_PRESENTATION_MODELS is set
    to an absolute path, return that path. If it's set to a relative path (does
    not start with /) then return that path taken to be relative to the mhcflurry
    downloads dir.

    Parameters
    ----------

    test_exists : boolean, optional
        Whether to raise an exception of the path does not exist

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

    If environment variable MHCFLURRY_DEFAULT_CLASS1_PROCESSING_MODELS is set
    to an absolute path, return that path. If it's set to a relative path (does
    not start with /) then return that path taken to be relative to the mhcflurry
    downloads dir.

    Parameters
    ----------

    test_exists : boolean, optional
        Whether to raise an exception of the path does not exist

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


def get_current_release_downloads():
    """
    Return a dict of all available downloads in the current release.

    The dict keys are the names of the downloads. The values are a dict
    with two entries:

    downloaded : bool
        Whether the download is currently available locally

    metadata : dict
        Info about the download from downloads.yml such as URL

    up_to_date : bool or None
        Whether the download URL(s) match what was used to download the current
        data. This is None if it cannot be determined.
    """
    downloads = (
        get_downloads_metadata()
        ['releases']
        [get_current_release()]
        ['downloads'])

    def up_to_date(dir, urls):
        try:
            df = pandas.read_csv(join(dir, "DOWNLOAD_INFO.csv"))
            return list(df.url) == list(urls)
        except IOError:
            return None

    return OrderedDict(
        (download["name"], {
            'downloaded': exists(join(get_downloads_dir(), download["name"])),
            'up_to_date': up_to_date(
                join(get_downloads_dir(), download["name"]),
                [download['url']] if 'url' in download else download['part_urls']),
            'metadata': download,
        }) for download in downloads
    )


def get_path(download_name, filename='', test_exists=True):
    """
    Get the local path to a file in a MHCflurry download

    Parameters
    -----------
    download_name : string

    filename : string
        Relative path within the download to the file of interest

    test_exists : boolean
        If True (default) throw an error telling the user how to download the
        data if the file does not exist

    Returns
    -----------
    string giving local absolute path
    """
    assert '/' not in download_name, "Invalid download: %s" % download_name
    path = join(get_downloads_dir(), download_name, filename)
    if test_exists and not exists(path):
        raise RuntimeError(
            "Missing MHCflurry downloadable file: %s. "
            "To download this data, run:\n\tmhcflurry-downloads fetch %s\n"
            "in a shell."
            % (quote(path), download_name))
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
            metadata["releases"][_CURRENT_RELEASE]["compatibility-version"])
        current_compatability = metadata["current-compatibility-version"]
        if current_release_compatability != current_compatability:
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
