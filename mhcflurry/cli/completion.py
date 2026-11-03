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

"""Shell tab completion (argcomplete) for the command-line entry points.

Each Tab press runs the command again, up to its ``argcomplete.autocomplete``
call, in a process whose environment carries the line being completed. Only
argument parsing happens in that process, so anything a parser computes just
for its help text can be skipped there; see :func:`is_completing`.
"""

import os

import argcomplete


def is_completing(environ=None):
    """Return whether this process is answering a shell completion request.

    Parameters
    ----------
    environ : mapping, optional
        Environment to inspect. Defaults to ``os.environ``.

    Returns
    -------
    bool
        True when argcomplete's ``_ARGCOMPLETE`` variable is set.
    """
    environ = os.environ if environ is None else environ
    return "_ARGCOMPLETE" in environ


def completing_subcommand(subcommands, environ=None):
    """Return the subcommand already typed on the line being completed.

    Parameters
    ----------
    subcommands : collection of str
        Subcommand names the program accepts.
    environ : mapping, optional
        Environment holding argcomplete's ``COMP_LINE``, ``COMP_POINT`` and
        ``_ARGCOMPLETE`` variables. Defaults to ``os.environ``.

    Returns
    -------
    str or None
        The word after the program name, when it is one of ``subcommands``
        and the cursor has moved past it. None otherwise, including when
        this process is not completing.
    """
    environ = os.environ if environ is None else environ
    if not is_completing(environ):
        return None
    comp_line = environ.get("COMP_LINE", "")
    comp_point = int(environ.get("COMP_POINT", len(comp_line)))
    # The words before the one under the cursor. As argcomplete does,
    # ``_ARGCOMPLETE`` gives where the program name sits: 1 for ``prog``,
    # 2 for ``python prog`` and 3 for ``python -m module``.
    words = argcomplete.split_line(comp_line, comp_point)[3]
    words = words[int(environ["_ARGCOMPLETE"]) - 1:]
    if len(words) > 1 and words[1] in subcommands:
        return words[1]
    return None
