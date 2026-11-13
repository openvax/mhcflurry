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

"""Check that every archive URL in the downloads catalogue resolves.

Sends one HEAD request per distinct URL in ``mhcflurry/downloads.yml`` and
exits non-zero if any fails. Run it before publishing a new release stanza;
it needs network access, so it is not part of the test suite.
"""

import argparse
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from mhcflurry.downloads import get_download_urls, get_downloads_metadata


def catalogue_urls(metadata):
    """
    Map each distinct archive URL to the catalogue entries that use it.

    Parameters
    ----------
    metadata : dict
        Parsed ``downloads.yml``.

    Returns
    -------
    dict of str -> list of (str, str)
        URL to ``(release, download name)`` pairs, in catalogue order.
    """
    result = {}
    for release, info in metadata["releases"].items():
        for download in info["downloads"]:
            for url in get_download_urls(download):
                result.setdefault(url, []).append((release, download["name"]))
    return result


def url_error(url, timeout):
    """
    Return None if a HEAD request for the URL succeeds, else the error text.

    Parameters
    ----------
    url : str
    timeout : float
        Seconds to wait for the server.

    Returns
    -------
    str or None
    """
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return None
    except urllib.error.HTTPError as error:
        return "HTTP %d" % error.code
    except (urllib.error.URLError, OSError) as error:
        return str(error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        action="append",
        help="Check only this release stanza. May be given more than once.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)

    urls = catalogue_urls(get_downloads_metadata())
    if args.release:
        urls = {
            url: users for url, users in urls.items()
            if any(release in args.release for release, _ in users)
        }
        if not urls:
            parser.error("No downloads in release(s): %s" % ", ".join(args.release))

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        errors = list(executor.map(lambda url: url_error(url, args.timeout), urls))

    failures = 0
    for (url, users), error in zip(urls.items(), errors):
        if error is None:
            continue
        failures += 1
        print("%s: %s (%s)" % (
            error, url,
            ", ".join("%s %s" % user for user in users)))
    print("Checked %d URLs: %d failed" % (len(urls), failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
