# AGENTS.md — mhcflurry

Guide for coding agents working in this repo. Read this before touching code.

---

## Golden Rules

1. **Never commit to `main`.** Always `git checkout -b <feature-branch>` before editing. Land via PR.
2. **Every PR bumps the version.** Even doc-only PRs — at minimum a patch bump in the package's `__init__.py` / `_version.py`.
3. **"Done" means merged AND released** — never stop at merge. mhcflurry doesn't (yet) have `deploy.sh`; follow the release recipe in `CONTRIBUTING.md` / `NOTES.md` and push the tag so PyPI gets the new version. Skipping release = task not done.
4. **File problems as issues, don't silently work around them.** If you hit a bug here or in a sibling openvax/pirl-unc repo, open a GitHub issue on the correct repo and link it from the PR.
5. **After a PR ships, look for the next block of work.** Read open issues across the relevant openvax repos, group by dependency + urgency. Prefer *foundational* changes that unblock multiple downstream improvements; otherwise chain the smallest independent improvements.

---

## Repo Shape (read before scripting)

Unlike its siblings, mhcflurry does **not** have `test.sh`, `deploy.sh`, or `format.sh`. It has:

- `develop.sh` — **source** this (`source develop.sh`) to create/activate `.venv` and editable-install. Do not `./develop.sh` (its venv activation won't persist).
- `lint.sh` — `ruff check mhcflurry/ test/` (note: tests live in `test/`, singular).
- `setup.py` — packaging.
- No `pyproject.toml`.

If you want to add `test.sh` / `deploy.sh` / `format.sh` to match the other openvax repos, that's a welcome foundational PR — discuss with Alex first.

## Before Completing Any Task

Before telling the user a change is "complete":

1. **`./lint.sh`** — must pass (ruff check)
2. **Run tests**: `pytest test/` (no `test.sh` wrapper). For the slow ML suite you may need downloaded models — see `docker/` or `test-environment.yml`.
3. For a PR: **CI must be green on GitHub**, then merge, then release (see Golden Rule 3).

## Code Style

- Python 3.9+
- Lint: ruff (concise output)
- Docstrings: numpy style
- Bugfixes include a regression test where feasible
- mhcflurry is a trained ML model system — be extremely cautious about changes that could alter predictions without a clear reason. Prediction-affecting changes need empirical validation, not just green tests.

---

## Workflow Orchestration

### 1. Upfront Planning
- For any non-trivial task (3+ steps or architectural): write a short spec first. If something goes sideways, STOP and re-plan — don't keep pushing.

### 2. Verification Before Done
- Never claim complete without proof: tests green, CI green, release tagged.
- For model or training changes: include before/after metrics on a held-out set.

### 3. Autonomous Bug Fixing
- Given a bug report: just fix it. Point at logs/errors/failing tests and resolve them without hand-holding.

### 4. Demand Elegance (Balanced)
- For non-trivial changes pause and ask "is there a more elegant way?" — skip for trivial fixes.
- Treat workarounds as bugs, not new abstractions. Rip out legacy paths decisively rather than accumulating special cases.

### 5. Issue Triage After Each Ship
- Close superseded/outdated issues as you notice them.
- New problems mid-task → file as issues (on the right repo, even if it's not this one), don't bury.

---

## Core Principles

- **Simplicity first.** Minimal diffs, minimal abstractions.
- **No laziness.** Find root causes; no temporary fixes, no empty-category fudges.
- **Minimal blast radius.** Touch only what the task requires.

## Scientific Domain Knowledge

- If a change touches immunology/genomics semantics, check primary sources (papers, UniProt, GenBank) before edits.
- If the code expresses a scientific model at odds with your understanding, flag it — don't silently "fix" it into something wrong.
- Use `mhcgnomes` for MHC allele parsing. Never `startswith("HLA-")` or other string hacks — alleles aren't always human.
