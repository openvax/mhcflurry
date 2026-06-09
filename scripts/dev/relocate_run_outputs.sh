#!/usr/bin/env bash
# Relocate run-output directories outside the repo so they aren't
# rsync'd to remote training boxes on every launch.
#
# Background: ``runplz``'s rsync_up exclude list is hardcoded
# (.git, .venv, __pycache__, *.egg-info, build, dist, out, secrets).
# Project-specific output dirs like ``brev_runs/`` (~15 GB) and
# ``results/`` ride along on every launch unless they're moved
# outside the rsync source tree. ``rsync -a`` ships symlinks as
# symlinks (not the dereferenced target), so a relocated-and-
# symlinked dir uploads as ~zero bytes.
#
# Usage:
#   bash scripts/dev/relocate_run_outputs.sh        # dry-run (default)
#   bash scripts/dev/relocate_run_outputs.sh --apply
#
# After running with --apply:
#   * brev_runs/  → ~/mhcflurry-brev-runs/, symlinked back
#   * results/    → ~/mhcflurry-results/,   symlinked back
# All existing tooling continues to read brev_runs/<run>/ paths
# unchanged via the symlink.
#
# Idempotent: if a path is already a symlink, this script no-ops.
# Safe-by-default: the move is a hardlink-friendly ``mv`` between
# files on the same filesystem; if the target exists we abort
# rather than clobber.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APPLY=0
if [ "${1:-}" = "--apply" ]; then
    APPLY=1
fi

relocate_one() {
    local rel="$1"
    local target="$2"
    local src="$REPO_ROOT/$rel"

    if [ -L "$src" ]; then
        printf "  skip %s (already a symlink → %s)\n" "$rel" "$(readlink "$src")"
        return 0
    fi
    if [ ! -d "$src" ]; then
        printf "  skip %s (not present)\n" "$rel"
        return 0
    fi
    if [ -e "$target" ]; then
        printf "  ABORT %s: relocation target %s already exists; "\
"merge manually before re-running\n" "$rel" "$target" >&2
        return 1
    fi

    if [ "$APPLY" = "1" ]; then
        mv "$src" "$target"
        ln -s "$target" "$src"
        printf "  moved %s → %s and symlinked back\n" "$rel" "$target"
    else
        local size
        size="$(du -sh "$src" 2>/dev/null | cut -f1)"
        printf "  would move %s (%s) → %s and symlink back\n" "$rel" "$size" "$target"
    fi
}

mode_label="DRY RUN"
if [ "$APPLY" = "1" ]; then
    mode_label="APPLY"
fi
echo "[$mode_label] Relocating run-output dirs outside $REPO_ROOT"
relocate_one "brev_runs" "$HOME/mhcflurry-brev-runs"
relocate_one "results"   "$HOME/mhcflurry-results"

if [ "$APPLY" != "1" ]; then
    echo
    echo "Re-run with --apply to actually move and symlink."
fi
