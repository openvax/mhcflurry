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

"""Terminal help remains readable without contaminating machine output."""

import io
import re

import pytest

from mhcflurry.cli import main, predict_command, predict_scan_command
from mhcflurry.cli.help import style_help


class Terminal(io.StringIO):
    def isatty(self):
        return True


@pytest.mark.parametrize("module", [predict_command, predict_scan_command])
@pytest.mark.parametrize("columns", [80, 240])
def test_prediction_help_is_compact_and_lists_every_flag(monkeypatch, module, columns):
    monkeypatch.setenv("COLUMNS", str(columns))
    text = module.parser.format_help()
    assert len(text.splitlines()[0]) < 80
    assert "[OPTIONS]" in text.splitlines()[0]
    assert "Examples:" in text
    assert text.index("Input") < text.index("Help:")
    assert "\x1b[" not in text
    assert "``" not in text
    assert max(map(len, text.splitlines())) <= min(columns, 100)
    for action in module.parser._actions:
        for flag in action.option_strings:
            assert flag in text


@pytest.mark.parametrize("module", [predict_command, predict_scan_command])
def test_printed_help_color_depends_on_destination(monkeypatch, module):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    plain = io.StringIO()
    terminal = Terminal()
    module.parser.print_help(plain)
    module.parser.print_help(terminal)
    assert "\x1b[" not in plain.getvalue()
    assert "\x1b[36m--alleles\x1b[0m" in terminal.getvalue()
    assert re.sub(r"\x1b\[[0-9;]*m", "", terminal.getvalue()) == plain.getvalue()
    # Programmatic consumers (including generated docs) never get ANSI escapes.
    assert module.parser.format_help() == plain.getvalue()


@pytest.mark.parametrize("environment", [{"NO_COLOR": "1"}, {"TERM": "dumb"}])
def test_help_respects_disabled_color(monkeypatch, environment):
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    text = predict_command.parser.format_help()
    assert style_help(text, Terminal()) == text


@pytest.mark.parametrize("command", ["predict", "predict-scan"])
@pytest.mark.parametrize("arguments,status", [([], 1), (["--help"], 0)])
def test_bare_and_explicit_help_preserve_exit_behavior(capsys, command, arguments, status):
    with pytest.raises(SystemExit) as exc:
        main.main([command, *arguments])
    assert exc.value.code == status
    captured = capsys.readouterr()
    assert captured.out.startswith("usage: mhcflurry " + command + " [OPTIONS]")
    assert "\x1b[" not in captured.out
    assert captured.err == ""


def test_argument_errors_still_go_to_stderr(capsys):
    with pytest.raises(SystemExit) as exc:
        main.main(["predict", "--not-an-option"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unrecognized arguments" in captured.err
    assert "\x1b[" not in captured.err
