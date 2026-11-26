#!/usr/bin/env python3
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

"""Classify changed files to decide whether CI needs the full ML test matrix.

Issue #471: Documentation-only and container-only PRs (including the required
``mhcflurry/version.py`` bump and release notes) should not block for ~45
minutes re-running the full 4-Python ML model test matrix when no runtime,
training, catalogue, dependency, test, or CI-gate file changed.

Fail-safe design:
- Non-``pull_request`` events (e.g., ``push`` to ``master``) always run ``full``.
- Empty or unresolvable diffs always run ``full``.
- Any path not explicitly allowlisted as docs/container/release-notes/version
  triggers ``full``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


LIGHT_EXACT_PATHS = frozenset({
    "Dockerfile",
    ".dockerignore",
    "mhcflurry/version.py",
    ".github/workflows/docs.yml",
    ".github/workflows/docker.yml",
})

LIGHT_PREFIXES = (
    "docs/",
    "release-notes/",
    "docker/",
)


def _normalize_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def is_light_path(path: str) -> bool:
    """Return True iff ``path`` cannot affect Python package or ML test outcomes."""
    norm = _normalize_path(path)
    if not norm:
        return False
    if norm in LIGHT_EXACT_PATHS:
        return True
    if any(norm.startswith(prefix) for prefix in LIGHT_PREFIXES):
        return True
    # Top-level Markdown files (README.md, CONTRIBUTING.md, AGENTS.md, etc.)
    if "/" not in norm and norm.endswith(".md"):
        return True
    return False


def classify_changed_files(
    changed_files: list[str] | tuple[str, ...],
    *,
    event_name: str = "pull_request",
) -> dict:
    """Classify a change set as ``light`` or ``full`` with an audit trail."""
    normalized = sorted({_normalize_path(p) for p in changed_files if p.strip()})

    if event_name != "pull_request":
        return {
            "mode": "full",
            "reason": (
                f"event '{event_name}' always runs the full ML matrix"
            ),
            "changed_files": normalized,
            "full_trigger_files": normalized,
        }

    if not normalized:
        return {
            "mode": "full",
            "reason": "no changed files detected; failing safe to full ML matrix",
            "changed_files": [],
            "full_trigger_files": [],
        }

    full_triggers = [p for p in normalized if not is_light_path(p)]
    if full_triggers:
        preview = ", ".join(full_triggers[:5])
        if len(full_triggers) > 5:
            preview += f" (+{len(full_triggers) - 5} more)"
        return {
            "mode": "full",
            "reason": (
                f"{len(full_triggers)} non-docs/container file(s) changed: {preview}"
            ),
            "changed_files": normalized,
            "full_trigger_files": full_triggers,
        }

    return {
        "mode": "light",
        "reason": (
            f"all {len(normalized)} changed file(s) are docs/container/"
            "release-notes/version-only"
        ),
        "changed_files": normalized,
        "full_trigger_files": [],
    }


def git_changed_files(base_sha: str, head_sha: str) -> list[str]:
    """Return paths changed between ``base_sha`` and ``head_sha`` via git diff."""
    proc = subprocess.run(
        ["git", "diff", "--name-only", f"{base_sha}...{head_sha}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def format_markdown_summary(result: dict) -> str:
    """Render a concise Markdown summary suitable for ``$GITHUB_STEP_SUMMARY``."""
    lines = [
        "### CI Change Classification (`#471`)",
        "",
        f"- **Mode:** `{result['mode']}`",
        f"- **Reason:** {result['reason']}",
        f"- **Changed files ({len(result['changed_files'])}):**",
    ]
    for path in result["changed_files"][:30]:
        tag = "full-trigger" if path in result["full_trigger_files"] else "light"
        lines.append(f"  - `{path}` ({tag})")
    if len(result["changed_files"]) > 30:
        lines.append(f"  - *(+{len(result['changed_files']) - 30} more)*")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify CI changes as light (docs/container) or full (ML matrix)."
    )
    parser.add_argument("--event-name", default=os.environ.get("GITHUB_EVENT_NAME", "pull_request"))
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--head-sha", default="HEAD")
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument("--github-output", default=None)
    parser.add_argument("--summary-file", default=None)
    args = parser.parse_args(argv)

    if args.files is not None:
        changed = list(args.files)
    elif args.event_name == "pull_request" and args.base_sha:
        try:
            changed = git_changed_files(args.base_sha, args.head_sha or "HEAD")
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            detail = f"{exc}: {stderr}" if stderr else str(exc)
            print(
                f"git diff failed ({detail}); failing safe to full ML matrix",
                file=sys.stderr,
            )
            changed = []
    else:
        changed = []

    result = classify_changed_files(changed, event_name=args.event_name)
    print(f"CI mode={result['mode']}: {result['reason']}")

    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as fd:
            fd.write(f"mode={result['mode']}\n")
            fd.write(f"reason={result['reason']}\n")

    if args.summary_file:
        with open(args.summary_file, "a", encoding="utf-8") as fd:
            fd.write(format_markdown_summary(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
