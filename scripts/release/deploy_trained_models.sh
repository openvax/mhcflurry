#!/usr/bin/env bash
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

set -euo pipefail

usage() {
    cat <<'EOF'
Package trained MHCflurry release models and upload them to GitHub.

Usage:
  scripts/release/deploy_trained_models.sh \
      --run-dir /path/to/release-run \
      --release 2.3.0 \
      --github-release 2.3.0 \
      [--date YYYYMMDD] \
      [--assets-dir /path/to/assets] \
      [--dry-run | --draft | --publish | --mode MODE]

Modes:
  --dry-run   Validate paths and print planned assets. This is the default.
  --draft     Build assets and upload them to a draft GitHub release, creating
              the draft if needed.
  --publish   Build assets and upload them to an existing GitHub release. This
              script does not publish a release because publishing also triggers
              package-release workflows.

Expected run layout:
  <run-dir>/affinity/models.combined
  <run-dir>/processing/models.selected.no_flank
  <run-dir>/processing/models.selected.short_flanks
  <run-dir>/presentation/models

The script writes SHA256SUMS and a downloads.yml snippet beside the assets.
Commit the downloads.yml update after the GitHub assets are uploaded and their
final URLs are known.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

warn() {
    echo "WARNING: $*" >&2
}

note() {
    echo "$*" >&2
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_dir() {
    [ -d "$1" ] || die "Missing directory: $1"
}

require_file() {
    [ -f "$1" ] || die "Missing file: $1"
}

require_one_file() {
    local label=$1
    shift
    local candidate
    for candidate in "$@"; do
        if [ -f "$candidate" ]; then
            return 0
        fi
    done
    die "Missing ${label}; tried: $*"
}

checksum_assets() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$@"
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$@"
    else
        die "Required command not found: sha256sum or shasum"
    fi
}

quote_cmd() {
    local arg
    printf '+'
    for arg in "$@"; do
        printf ' %q' "$arg"
    done
    printf '\n'
}

RUN_DIR=
RELEASE=
GITHUB_RELEASE=
ASSETS_DIR=
ASSET_DATE=$(date -u +%Y%m%d)
MODE=dry-run

while [ $# -gt 0 ]; do
    case "$1" in
        --run-dir)
            RUN_DIR=$2
            shift 2
            ;;
        --release)
            RELEASE=$2
            shift 2
            ;;
        --github-release)
            GITHUB_RELEASE=$2
            shift 2
            ;;
        --assets-dir)
            ASSETS_DIR=$2
            shift 2
            ;;
        --date)
            ASSET_DATE=$2
            shift 2
            ;;
        --dry-run)
            MODE=dry-run
            shift
            ;;
        --draft)
            MODE=draft
            shift
            ;;
        --publish)
            MODE=publish
            shift
            ;;
        --mode)
            case "$2" in
                dry-run|draft|publish)
                    MODE=$2
                    ;;
                *)
                    die "--mode must be one of: dry-run, draft, publish"
                    ;;
            esac
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

[ -n "$RUN_DIR" ] || die "--run-dir is required"
[ -n "$RELEASE" ] || die "--release is required"
if [ -z "$GITHUB_RELEASE" ]; then
    GITHUB_RELEASE=$RELEASE
fi
if [ -z "$ASSETS_DIR" ]; then
    ASSETS_DIR="$RUN_DIR/release-assets"
fi

require_dir "$RUN_DIR"
RUN_DIR=$(cd "$RUN_DIR" && pwd)
case "$ASSETS_DIR" in
    /*) ;;
    *) ASSETS_DIR="$(pwd)/$ASSETS_DIR" ;;
esac
ASSETS_DIR=${ASSETS_DIR%/}

AFFINITY_DIR="$RUN_DIR/affinity"
PROCESSING_DIR="$RUN_DIR/processing"
PRESENTATION_DIR="$RUN_DIR/presentation"
AFFINITY_MODELS="$AFFINITY_DIR/models.combined"
PROCESSING_NO_FLANK="$PROCESSING_DIR/models.selected.no_flank"
PROCESSING_SHORT_FLANKS="$PROCESSING_DIR/models.selected.short_flanks"
PROCESSING_WITH_FLANKS="$PROCESSING_DIR/models.selected.with_flanks"
PRESENTATION_MODELS="$PRESENTATION_DIR/models"

PAN_ASSET="models_class1_pan.selected.${ASSET_DATE}.tar.bz2"
PROCESSING_ASSET="models_class1_processing.selected.${ASSET_DATE}.tar.bz2"
PRESENTATION_ASSET="models_class1_presentation.${ASSET_DATE}.tar.bz2"
SHA_FILE="SHA256SUMS"
SNIPPET_FILE="downloads.${RELEASE}.snippet.yml"

require_command tar
if [ "$MODE" != "dry-run" ]; then
    require_command gh
fi

require_dir "$AFFINITY_MODELS"
require_file "$AFFINITY_MODELS/manifest.csv"
require_one_file "affinity percent ranks" \
    "$AFFINITY_MODELS/percent_ranks.csv" \
    "$AFFINITY_MODELS/percent_ranks.csv.bz2"

require_dir "$PROCESSING_NO_FLANK"
require_file "$PROCESSING_NO_FLANK/manifest.csv"
require_dir "$PROCESSING_SHORT_FLANKS"
require_file "$PROCESSING_SHORT_FLANKS/manifest.csv"
if [ ! -d "$PROCESSING_WITH_FLANKS" ]; then
    warn "Processing models.selected.with_flanks is absent."
    warn "The archive will contain no_flank and short_flanks only."
fi

require_dir "$PRESENTATION_MODELS"
require_file "$PRESENTATION_MODELS/weights.csv"
require_one_file "presentation percent ranks" \
    "$PRESENTATION_MODELS/percent_ranks.csv" \
    "$PRESENTATION_MODELS/percent_ranks.csv.bz2"

note "Release:          $RELEASE"
note "GitHub release:   $GITHUB_RELEASE"
note "Run directory:    $RUN_DIR"
note "Assets directory: $ASSETS_DIR"
note "Mode:             $MODE"
note ""
note "Assets:"
note "  $PAN_ASSET"
note "  $PROCESSING_ASSET"
note "  $PRESENTATION_ASSET"

if [ "$MODE" = "dry-run" ]; then
    note ""
    note "Dry run only. Commands that would run:"
    quote_cmd mkdir -p "$ASSETS_DIR"
    quote_cmd tar -C "$AFFINITY_DIR" -cjf "$ASSETS_DIR/$PAN_ASSET" models.combined
    if [ -d "$PROCESSING_WITH_FLANKS" ]; then
        quote_cmd tar -C "$PROCESSING_DIR" -cjf "$ASSETS_DIR/$PROCESSING_ASSET" \
            models.selected.no_flank models.selected.short_flanks \
            models.selected.with_flanks
    else
        quote_cmd tar -C "$PROCESSING_DIR" -cjf "$ASSETS_DIR/$PROCESSING_ASSET" \
            models.selected.no_flank models.selected.short_flanks
    fi
    quote_cmd tar -C "$PRESENTATION_DIR" -cjf "$ASSETS_DIR/$PRESENTATION_ASSET" models
    quote_cmd gh release upload "$GITHUB_RELEASE" \
        "$ASSETS_DIR/$PAN_ASSET" \
        "$ASSETS_DIR/$PROCESSING_ASSET" \
        "$ASSETS_DIR/$PRESENTATION_ASSET" \
        "$ASSETS_DIR/$SHA_FILE" \
        --clobber
    exit 0
fi

mkdir -p "$ASSETS_DIR"
tar -C "$AFFINITY_DIR" -cjf "$ASSETS_DIR/$PAN_ASSET" models.combined
if [ -d "$PROCESSING_WITH_FLANKS" ]; then
    tar -C "$PROCESSING_DIR" -cjf "$ASSETS_DIR/$PROCESSING_ASSET" \
        models.selected.no_flank models.selected.short_flanks \
        models.selected.with_flanks
else
    tar -C "$PROCESSING_DIR" -cjf "$ASSETS_DIR/$PROCESSING_ASSET" \
        models.selected.no_flank models.selected.short_flanks
fi
tar -C "$PRESENTATION_DIR" -cjf "$ASSETS_DIR/$PRESENTATION_ASSET" models

(
    cd "$ASSETS_DIR"
    checksum_assets "$PAN_ASSET" "$PROCESSING_ASSET" "$PRESENTATION_ASSET" \
        > "$SHA_FILE"
)

cat > "$ASSETS_DIR/$SNIPPET_FILE" <<EOF
  ${RELEASE}:
    compatibility-version: 2
    downloads:
      - name: models_class1_pan
        url: https://github.com/openvax/mhcflurry/releases/download/${GITHUB_RELEASE}/${PAN_ASSET}
        default: false

      - name: models_class1_processing
        url: https://github.com/openvax/mhcflurry/releases/download/${GITHUB_RELEASE}/${PROCESSING_ASSET}
        default: false

      - name: models_class1_presentation
        url: https://github.com/openvax/mhcflurry/releases/download/${GITHUB_RELEASE}/${PRESENTATION_ASSET}
        default: true
EOF

if [ "$MODE" = "draft" ]; then
    if ! gh release view "$GITHUB_RELEASE" >/dev/null 2>&1; then
        gh release create "$GITHUB_RELEASE" --draft --title "MHCflurry $RELEASE"
    fi
elif [ "$MODE" = "publish" ]; then
    gh release view "$GITHUB_RELEASE" >/dev/null 2>&1 || \
        die "--publish requires an existing GitHub release: $GITHUB_RELEASE"
else
    die "Unsupported mode: $MODE"
fi

gh release upload "$GITHUB_RELEASE" \
    "$ASSETS_DIR/$PAN_ASSET" \
    "$ASSETS_DIR/$PROCESSING_ASSET" \
    "$ASSETS_DIR/$PRESENTATION_ASSET" \
    "$ASSETS_DIR/$SHA_FILE" \
    --clobber

note ""
note "Wrote:"
note "  $ASSETS_DIR/$SHA_FILE"
note "  $ASSETS_DIR/$SNIPPET_FILE"
