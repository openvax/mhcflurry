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

"""Readable argparse help with optional terminal styling."""

import argparse
import os
import re
import shutil
import sys


_OPTION = re.compile(r"(?<![\w-])(--[a-zA-Z][\w-]*|-[a-zA-Z])\b")


def style_help(text, stream):
    """Style headings and flags only when writing to a color-capable terminal."""
    if (
            not getattr(stream, "isatty", lambda: False)()
            or os.environ.get("NO_COLOR")
            or os.environ.get("TERM") == "dumb"):
        return text

    lines = []
    for line in text.splitlines(keepends=True):
        if line.rstrip().endswith(":") and not line[0].isspace():
            line = "\033[1;36m" + line.rstrip("\n") + "\033[0m\n"
        else:
            line = _OPTION.sub(lambda match: "\033[36m" + match[0] + "\033[0m", line)
            if line.startswith("usage:"):
                line = line.replace("usage:", "\033[1musage:\033[0m", 1)
        lines.append(line)
    return "".join(lines)


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Limit reading width and separate options without adding table borders."""

    def __init__(self, prog):
        width = max(40, min(100, shutil.get_terminal_size().columns - 2))
        super().__init__(prog, max_help_position=32, width=width)

    def _split_lines(self, text, width):
        # RST literals are useful in generated docs but noisy in terminal help.
        return super()._split_lines(text.replace("``", ""), width)

    def _format_action(self, action):
        text = super()._format_action(action)
        return text + "\n" if text else text


class HelpArgumentParser(argparse.ArgumentParser):
    """Keep format_help() plain; add color only when printing to a terminal."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", HelpFormatter)
        if sys.version_info >= (3, 14):
            # Apply the same styling on all supported Python versions.
            kwargs["color"] = False
        super().__init__(*args, **kwargs)

    def print_help(self, file=None):
        stream = sys.stdout if file is None else file
        self._print_message(style_help(self.format_help(), stream), stream)
