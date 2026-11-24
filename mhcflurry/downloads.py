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

Importing this module is cheap: the download catalogue (``downloads.yml``)
is read, and the download directory configured, the first time a function
here needs them.
"""

import csv
import io
import logging
import os
import posixpath
from collections import OrderedDict
from os import environ
from os.path import abspath, dirname, exists, isabs, join
from shlex import quote
from urllib.parse import urlsplit

from platformdirs import user_data_dir

MANAGED_ARCHIVES_DIRNAME = ".datacache"

ENVIRONMENT_VARIABLES = [
    "MHCFLURRY_DATA_DIR",
    "OPENVAX_DATA_CACHE",
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
    if _DOWNLOADS_DIR is None:
        configure()
    if release is None or _CURRENT_RELEASE is None:
        return _DOWNLOADS_DIR
    return join(dirname(_DOWNLOADS_DIR), release)


def get_managed_store_dir(download_name, release=None):
    """
    Return the ``datacache`` archive store directory for one bundle.

    A managed store is not itself the extracted tree; call :func:`get_path`
    or :func:`inspect_download` to obtain the active generation directory.
    """
    assert "/" not in download_name, "Invalid download: %s" % download_name
    return join(get_downloads_dir(release), MANAGED_ARCHIVES_DIRNAME, download_name)


def get_current_release():
    """
    Return the current downloaded data release
    """
    if _DOWNLOADS_DIR is None:
        configure()
    return _CURRENT_RELEASE


def get_downloads_metadata():
    """
    Return the contents of downloads.yml as a dict
    """
    global _METADATA
    if _METADATA is None:
        # Imported here, not at module level: reading the catalogue costs
        # tens of milliseconds, which shell completion should not pay.
        import yaml
        from importlib.resources import files
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


def get_download_integrity(metadata):
    """Return ``(expected_sha256, expected_size)`` from catalogue metadata.

    Returns ``(None, None)`` for historical unpinned entries. Raises
    ``ValueError`` if only one of ``sha256`` or ``size`` is supplied.
    """
    sha256 = metadata.get("sha256")
    size = metadata.get("size")
    if (sha256 is None) != (size is None):
        raise ValueError(
            "Catalogue entry %r must specify both 'sha256' and 'size' together."
            % (metadata.get("name"),)
        )
    return sha256, size


def _download_info_csv_bytes(urls):
    """Serialize the ordered ``DOWNLOAD_INFO.csv`` consumer receipt."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["url"])
    for url in urls:
        writer.writerow([url])
    return buf.getvalue().encode("utf-8")


def _build_archive_sources(metadata, already_downloaded_dir=None):
    """Build ordered ``datacache.install_archive`` source descriptors."""
    urls = get_download_urls(metadata)
    if not already_downloaded_dir:
        return list(urls)
    sources = []
    for url in urls:
        filename = posixpath.basename(urlsplit(url).path)
        sources.append({
            "url": url,
            "path": join(already_downloaded_dir, filename),
        })
    return sources


def _read_legacy_download_info_urls(directory):
    """Read recorded URLs from a legacy ``DOWNLOAD_INFO.csv`` if present."""
    try:
        with open(join(directory, "DOWNLOAD_INFO.csv"), newline="") as fd:
            reader = csv.DictReader(fd)
            if not reader.fieldnames or "url" not in reader.fieldnames:
                return None
            return [row["url"] for row in reader]
    except (OSError, csv.Error, UnicodeError):
        return None


def _get_catalogue_entry(download_name, release):
    """Return catalogue metadata for ``download_name`` in ``release``, or None."""
    if not release:
        return None
    releases = get_downloads_metadata().get("releases", {})
    release_info = releases.get(release)
    if not release_info:
        return None
    for entry in release_info.get("downloads", ()):
        if entry["name"] == download_name:
            return entry
    return None


def inspect_download(download_name, release=None, *, metadata=None, verify_files=False):
    """Inspect one bundle offline without writes, locks, or repair.

    Checks the ``datacache``-managed archive store first and falls back to any
    existing legacy directory under the release root. Legacy directories are
    never migrated, modified, or claimed.

    Parameters
    ----------
    download_name : str
        Bundle name in the download catalogue.
    release : str, optional
        Catalogue release identifier. Defaults to the configured release.
    metadata : dict, optional
        Catalogue metadata dict for this bundle. Looked up automatically when
        omitted and ``release`` is known.
    verify_files : bool, optional
        When False (default), perform a fast manifest and source-identity check
        without hashing extracted files (``verified`` is always False). When
        True, hash every extracted file in a managed store against its manifest
        and trusted catalogue checksums.

    Returns
    -------
    dict
        Inspection summary with keys ``downloaded``, ``up_to_date``, ``path``,
        ``layout``, ``install_status``, ``verified``, ``trusted_checksum``,
        ``source_urls``, ``fetched_at``, ``archive_size``, ``recorded_sha256``,
        and ``error``.
    """
    assert "/" not in download_name, "Invalid download: %s" % download_name
    if metadata is None:
        effective_release = release or get_current_release()
        metadata = _get_catalogue_entry(download_name, effective_release)

    expected_urls = get_download_urls(metadata) if metadata is not None else None
    expected_sha256, expected_size = (
        get_download_integrity(metadata) if metadata is not None else (None, None)
    )
    trusted_checksum = expected_sha256 is not None and expected_size is not None

    store_dir = get_managed_store_dir(download_name, release=release)
    legacy_dir = join(get_downloads_dir(release), download_name)

    if exists(store_dir):
        import datacache

        extra_files = (
            {"DOWNLOAD_INFO.csv": _download_info_csv_bytes(expected_urls)}
            if expected_urls is not None
            else None
        )
        matched = None
        if expected_urls is not None:
            matched = datacache.inspect_archive(
                store_dir,
                sources=expected_urls,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
                extra_files=extra_files,
                verify_files=verify_files,
            )
            if matched.status == "available":
                return {
                    "downloaded": True,
                    "up_to_date": True,
                    "path": matched.bundle,
                    "layout": "datacache",
                    "install_status": "available",
                    "verified": bool(matched.verified),
                    "trusted_checksum": trusted_checksum,
                    "source_urls": list(matched.source_urls),
                    "fetched_at": matched.fetched_at,
                    "archive_size": matched.archive_size,
                    "recorded_sha256": matched.recorded_sha256,
                    "error": None,
                }

        receipt = datacache.inspect_archive(store_dir, verify_files=verify_files)
        if receipt.status == "available":
            up_to_date = None if expected_urls is None else False
            return {
                "downloaded": True,
                "up_to_date": up_to_date,
                "path": receipt.bundle,
                "layout": "datacache",
                "install_status": "available",
                "verified": False,
                "trusted_checksum": trusted_checksum,
                "source_urls": list(receipt.source_urls),
                "fetched_at": receipt.fetched_at,
                "archive_size": receipt.archive_size,
                "recorded_sha256": receipt.recorded_sha256,
                "error": (
                    str(matched.error)
                    if matched is not None and matched.error is not None
                    else None
                ),
            }
        if receipt.status != "missing":
            return {
                "downloaded": False,
                "up_to_date": None,
                "path": legacy_dir,
                "layout": "datacache",
                "install_status": receipt.status,
                "verified": False,
                "trusted_checksum": trusted_checksum,
                "source_urls": list(receipt.source_urls),
                "fetched_at": receipt.fetched_at,
                "archive_size": receipt.archive_size,
                "recorded_sha256": receipt.recorded_sha256,
                "error": str(receipt.error) if receipt.error is not None else None,
            }

    if exists(legacy_dir):
        recorded_urls = _read_legacy_download_info_urls(legacy_dir)
        up_to_date = (
            None
            if expected_urls is None or recorded_urls is None
            else recorded_urls == list(expected_urls)
        )
        return {
            "downloaded": True,
            "up_to_date": up_to_date,
            "path": legacy_dir,
            "layout": "legacy",
            "install_status": "legacy",
            "verified": False,
            "trusted_checksum": trusted_checksum,
            "source_urls": recorded_urls,
            "fetched_at": None,
            "archive_size": None,
            "recorded_sha256": None,
            "error": None,
        }

    return {
        "downloaded": False,
        "up_to_date": None,
        "path": legacy_dir,
        "layout": None,
        "install_status": "missing",
        "verified": False,
        "trusted_checksum": trusted_checksum,
        "source_urls": None,
        "fetched_at": None,
        "archive_size": None,
        "recorded_sha256": None,
        "error": None,
    }


def install_download(
    download_name,
    release=None,
    *,
    metadata=None,
    already_downloaded_dir=None,
    force=False,
    offline=None,
    show_progress="auto",
    timeout=None,
    max_retries=None,
):
    """Install or refresh one download bundle through ``datacache.install_archive``.

    Parameters
    ----------
    download_name : str
        Bundle name in the download catalogue.
    release : str, optional
        Catalogue release identifier. Defaults to the configured release.
    metadata : dict, optional
        Catalogue entry dict. Looked up from ``downloads.yml`` when omitted.
    already_downloaded_dir : str, optional
        Local directory containing pre-downloaded archive files (retaining
        catalogue URLs as logical source identities).
    force : bool, optional
        Install a fresh generation even if a valid or invalid managed store
        already exists. Failed refreshes leave previous generations usable.
    offline : bool or None, optional
        Explicit offline override forwarded to ``datacache`` (``None`` honors
        ``OPENVAX_OFFLINE`` / ``DATACACHE_OFFLINE``).
    show_progress : bool or str, optional
        Progress bar policy (``"auto"``, ``True``, or ``False``).
    timeout : float or None, optional
        Optional per-request timeout in seconds forwarded to ``datacache``.
    max_retries : int or None, optional
        Optional retry count forwarded to ``datacache``.

    Returns
    -------
    pathlib.Path
        Directory path of the published extracted bundle generation.
    """
    import datacache

    if metadata is None:
        resolved_release = resolve_release(release)
        metadata = _get_catalogue_entry(download_name, resolved_release)
        if metadata is None:
            raise ValueError(
                "Unknown download %r in release %s." % (download_name, resolved_release)
            )
    else:
        resolved_release = release

    urls = get_download_urls(metadata)
    expected_sha256, expected_size = get_download_integrity(metadata)
    has_trusted_checksum = expected_sha256 is not None and expected_size is not None
    sources = _build_archive_sources(
        metadata, already_downloaded_dir=already_downloaded_dir
    )
    extra_files = {"DOWNLOAD_INFO.csv": _download_info_csv_bytes(urls)}
    store_dir = get_managed_store_dir(download_name, release=resolved_release)

    progress_enabled = (
        datacache.should_show_progress(show_progress)
        if isinstance(show_progress, str)
        else bool(show_progress)
    )
    download_options = {
        "show_progress": progress_enabled,
        "offline": offline,
    }
    # Only enable resumable transfers for single-archive entries with trusted
    # integrity metadata; never enable resume on unpinned or unpinned split parts.
    if has_trusted_checksum and len(urls) == 1 and not already_downloaded_dir:
        download_options["resume"] = "auto"
    if timeout is not None:
        download_options["timeout"] = timeout
    if max_retries is not None:
        download_options["max_retries"] = max_retries

    return datacache.install_archive(
        store_dir,
        sources,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        extra_files=extra_files,
        force=force,
        verified=has_trusted_checksum,
        download_options=download_options,
        portable_names=(os.name == "nt"),
    )


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


def _resolve_default_override_path(override_value, test_exists=True):
    """Resolve an environment override path (absolute or relative to downloads)."""
    if isabs(override_value):
        result = override_value
    else:
        result = join(get_downloads_dir(), override_value)
        if not exists(result):
            # Support relative overrides like ``models_class1_pan/models.combined``
            # against datacache-managed archive stores as well as legacy dirs.
            normalized = override_value.replace("\\", "/").lstrip("/")
            parts = normalized.split("/", 1)
            bundle_name = parts[0]
            subpath = parts[1] if len(parts) > 1 else ""
            if bundle_name and exists(get_managed_store_dir(bundle_name)):
                candidate = get_path(bundle_name, subpath, test_exists=False)
                if exists(candidate):
                    result = candidate
    if test_exists and not exists(result):
        raise IOError("No such directory: %s" % result)
    return result


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
        return _resolve_default_override_path(
            _MHCFLURRY_DEFAULT_CLASS1_MODELS_DIR, test_exists=test_exists
        )
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
        return _resolve_default_override_path(
            _MHCFLURRY_DEFAULT_CLASS1_PRESENTATION_MODELS_DIR,
            test_exists=test_exists,
        )
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
        return _resolve_default_override_path(
            _MHCFLURRY_DEFAULT_CLASS1_PROCESSING_MODELS_DIR,
            test_exists=test_exists,
        )

    # Default to the 'with flanks' model variant.
    return get_path(
        "models_class1_processing", "models.selected.with_flanks", test_exists=test_exists)


def get_release_downloads(release, verify_files=False):
    """
    Return a dict of all available downloads in a release.

    Parameters
    ----------
    release : string
        Release whose download metadata to return.
    verify_files : bool, optional
        When False (default), perform fast offline manifest inspection without
        hashing extracted files. When True, hash extracted files in managed
        stores against their manifests and trusted checksums.

    Returns
    -------
    collections.OrderedDict
        Download names mapped to dictionaries containing ``downloaded`` (bool),
        ``up_to_date`` (bool or None), ``metadata`` (dict), and provenance
        fields (``path``, ``layout``, ``install_status``, ``verified``,
        ``trusted_checksum``, ``source_urls``, ``fetched_at``, ``archive_size``,
        ``recorded_sha256``, ``error``).
    """
    resolved = resolve_release(release)
    downloads = get_downloads_metadata()['releases'][resolved]['downloads']

    result = OrderedDict()
    for download in downloads:
        inspection = inspect_download(
            download["name"],
            release=resolved,
            metadata=download,
            verify_files=verify_files,
        )
        inspection["metadata"] = download
        result[download["name"]] = inspection
    return result


def get_current_release_downloads():
    """Return a dict of all available downloads in the current release."""
    return get_release_downloads(get_current_release())


def _resolve_bundle_root(download_name, release=None):
    """Return the extracted tree root for ``download_name`` (managed or legacy)."""
    store_dir = get_managed_store_dir(download_name, release=release)
    if exists(store_dir):
        import datacache

        inspection = datacache.inspect_archive(store_dir, verify_files=False)
        if inspection.status == "available" and inspection.bundle:
            return inspection.bundle
    return join(get_downloads_dir(release), download_name)


def get_path(download_name, filename='', test_exists=True, release=None):
    """
    Get the local path to a file in a MHCflurry download.

    Checks ``datacache``-managed archive stores first and falls back to any
    legacy directory under the configured release root.

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
    base = _resolve_bundle_root(download_name, release=release)
    path = join(base, filename) if filename else base
    if test_exists and not exists(path):
        raise RuntimeError(
            "Missing MHCflurry downloadable file: %s. "
            "To download this data, run:\n\tmhcflurry downloads fetch %s%s\n"
            "in a shell."
            % (quote(path), download_name,
               " --release " + quote(release) if release else ""))
    return path


def resolve_data_location():
    """Resolve the base MHCflurry data directory and explain why (read-only).

    Precedence: ``MHCFLURRY_DATA_DIR`` -> ``OPENVAX_DATA_CACHE/mhcflurry`` ->
    populated shared platform cache -> populated legacy
    ``user_data_dir("mhcflurry", version="4")`` -> default shared platform cache.
    """
    import datacache

    # Increase the legacy version only on breaking layout changes; historical
    # installs under platformdirs version="4" are preserved automatically.
    return datacache.resolve_cache_dir(
        "mhcflurry",
        "MHCFLURRY_DATA_DIR",
        legacy=(user_data_dir("mhcflurry", version="4"),),
    )


def configure():
    """
    Setup various global variables based on environment variables.

    Runs automatically the first time the download directory or current
    release is needed. Call it again after changing
    ``MHCFLURRY_DOWNLOADS_DIR``, ``MHCFLURRY_DOWNLOADS_CURRENT_RELEASE``,
    ``MHCFLURRY_DATA_DIR``, or ``OPENVAX_DATA_CACHE``.
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

        data_dir = resolve_data_location().path
        _DOWNLOADS_DIR = join(data_dir, _CURRENT_RELEASE)

    logging.debug("Configured MHCFLURRY_DOWNLOADS_DIR: %s", _DOWNLOADS_DIR)
