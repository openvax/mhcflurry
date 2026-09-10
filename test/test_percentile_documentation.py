"""Keep the shared percentile guide's executable examples tied to the APIs."""

import doctest
import importlib.util
from pathlib import Path
import re
import shlex
from types import SimpleNamespace

import pytest

from mhcflurry.cli.calibrate_percentile_ranks_command import parser


GUIDE = Path(__file__).resolve().parents[1] / "docs" / "shared_percent_rank_transforms.md"


def test_shared_percentile_guide_doctests():
    source = GUIDE.read_text()
    examples = re.findall(r"```\{doctest\}\n(.*?)```", source, re.DOTALL)
    assert examples, "The shared guide must contain executable API examples"
    runner = doctest.DocTestRunner()
    for index, text in enumerate(examples):
        example = doctest.DocTestParser().get_doctest(
            text, {}, "shared-percentiles-%d" % index, str(GUIDE), 0)
        runner.run(example)
    results = runner.summarize()
    assert results.attempted >= 10
    assert results.failed == 0


def test_shared_percentile_cli_templates_match_parser():
    source = GUIDE.read_text()
    blocks = re.findall(r"```shell\n(.*?)```", source, re.DOTALL)
    kinds = set()
    for block in blocks:
        for command in block.strip().split("\n\n"):
            words = shlex.split(command.replace("\\\n", ""))
            assert words[:2] == ["mhcflurry", "calibrate-percentile-ranks"]
            args = parser.parse_args(words[2:])
            kinds.add(args.predictor_kind)
            assert args.percentile_method == "compact"
            assert args.max_percentile_knots == 128
            assert args.num_jobs == 0
            if args.predictor_kind == "class1_processing":
                assert args.processing_reference_data
    assert kinds == {"class1_affinity", "class1_processing", "class1_presentation"}


def load_api_coverage_extension():
    path = GUIDE.parent / "api_coverage.py"
    spec = importlib.util.spec_from_file_location("api_coverage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("complete", [True, False])
def test_documentation_build_requires_actual_api_objects(complete):
    extension = load_api_coverage_extension()
    names = extension.REQUIRED_API_OBJECTS if complete else ()
    domain = SimpleNamespace(get_objects=lambda: [(name,) for name in names])
    app = SimpleNamespace(env=SimpleNamespace(get_domain=lambda name: domain))
    if complete:
        extension.check_public_api(app, None)
    else:
        with pytest.raises(RuntimeError, match="Public API documentation is missing"):
            extension.check_public_api(app, None)
    # Preserve an original builder failure, rather than masking it with this check.
    extension.check_public_api(app, RuntimeError("earlier failure"))


def test_api_reference_parses_generated_rst_and_covers_subpackages():
    source = (GUIDE.parent / "api.md").read_text()
    assert "```{eval-rst}\n.. include:: _build/mhcflurry.rst" in source
    for name in ("affinity", "cli", "parallelism"):
        wrapper = GUIDE.parent / ("mhcflurry.%s.rst" % name)
        assert ".. include:: _build/mhcflurry.%s.rst" % name in wrapper.read_text()
