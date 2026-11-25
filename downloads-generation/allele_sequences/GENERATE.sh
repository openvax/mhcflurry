#!/bin/bash
#
# Create MHCflurry Class I pseudosequence tables.
#
# Default backend (MHCFLURRY_ALLELE_SEQUENCES_BACKEND=mhcseqs):
#   Builds 39-aa and 34-aa pseudosequences via ``mhcflurry pseudosequences
#   build`` backed by ``mhcseqs``, locking every allele present in the baseline
#   table bit-for-bit verbatim (``--lock-existing``) and verifying 0 diffs on
#   every allele in ``curated_training_data.csv.bz2`` before packaging.
#   Requires: mhcseqs (pip install 'mhcflurry[mhcseqs]')
#
# Legacy backend (MHCFLURRY_ALLELE_SEQUENCES_BACKEND=legacy):
#   Re-runs the original IPD-IMGT/HLA + IPD-MHC + UniProt ``wget`` +
#   ``clustalo`` + ``make_allele_sequences.py`` alignment pipeline.
#   Requires: clustalo, wget
#
set -e
set -x

DOWNLOAD_NAME=allele_sequences
BACKEND="${MHCFLURRY_ALLELE_SEQUENCES_BACKEND:-mhcseqs}"
SCRATCH_DIR=${TMPDIR-/tmp}/mhcflurry-downloads-generation
SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR=$(dirname "$SCRIPT_ABSOLUTE_PATH")
export PYTHONUNBUFFERED=1

pseudosequence_lookup() {
    python -c 'from mhcflurry.pseudosequences import main; main()' "$@"
}

mkdir -p "$SCRATCH_DIR"
rm -rf "$SCRATCH_DIR/$DOWNLOAD_NAME"
mkdir "$SCRATCH_DIR/$DOWNLOAD_NAME"

# Send stdout and stderr to a logfile included with the archive.
exec >  >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt")
exec 2> >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt" >&2)

# Log some environment info
date
echo "Backend: $BACKEND"
pip freeze
git status || true

cd "$SCRATCH_DIR/$DOWNLOAD_NAME"
cp "$SCRIPT_DIR/make_allele_sequences.py" .
cp "$SCRIPT_DIR/select_alleles_to_disambiguate.py" .
cp "$SCRIPT_DIR/filter_sequences.py" .
cp "$SCRIPT_DIR/verify_trained_allele_parity.py" .

NETMHCPAN_PSEUDOSEQUENCES="$(pseudosequence_lookup filename --length 34)"
MHCFLURRY_PSEUDOSEQUENCES="$(pseudosequence_lookup filename --length 39)"
LEGACY_ALLELE_SEQUENCES="$(pseudosequence_lookup legacy allele_sequences)"
LEGACY_CLASS1_PSEUDOSEQUENCES="$(pseudosequence_lookup legacy class1_pseudosequences)"
LEGACY_NO_DIFFERENTIATION="$(pseudosequence_lookup legacy no_differentiation)"

cp "$SCRIPT_DIR/$NETMHCPAN_PSEUDOSEQUENCES" .
# Compatibility alias used by older generation scripts and model artifacts.
cp "$SCRIPT_DIR/$NETMHCPAN_PSEUDOSEQUENCES" "$LEGACY_CLASS1_PSEUDOSEQUENCES"
cp "$SCRIPT_ABSOLUTE_PATH" .

TRAINING_DATA="${MHCFLURRY_TRAINING_DATA:-$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2}"

if [ "$BACKEND" = "mhcseqs" ]; then
    python -c 'import mhcseqs, mhcgnomes; print("mhcseqs:", mhcseqs.__version__, "mhcgnomes:", mhcgnomes.__version__)'

    # Resolve baseline 39-aa allele_sequences.csv to lock previously trained
    # alleles verbatim so no trained allele's pseudosequence can ever drift.
    BASELINE_CSV="${MHCFLURRY_BASELINE_ALLELE_SEQUENCES:-}"
    if [ -z "$BASELINE_CSV" ]; then
        if BASELINE_DIR="$(mhcflurry-downloads path allele_sequences 2>/dev/null)" && [ -f "$BASELINE_DIR/$LEGACY_ALLELE_SEQUENCES" ]; then
            BASELINE_CSV="$BASELINE_DIR/$LEGACY_ALLELE_SEQUENCES"
        elif BASELINE_DIR="$(mhcflurry-downloads path models_class1_pan 2>/dev/null)" && [ -f "$BASELINE_DIR/models/$LEGACY_ALLELE_SEQUENCES" ]; then
            BASELINE_CSV="$BASELINE_DIR/models/$LEGACY_ALLELE_SEQUENCES"
        fi
    fi

    BUILD_39_ARGS=(--length 39 --out "$MHCFLURRY_PSEUDOSEQUENCES")
    if [ -n "${MHCFLURRY_MHCSEQS_CSV:-}" ]; then
        BUILD_39_ARGS+=(--mhcseqs-csv "$MHCFLURRY_MHCSEQS_CSV")
    fi

    if [ -n "$BASELINE_CSV" ] && [ "$BASELINE_CSV" != "none" ]; then
        if [ ! -f "$BASELINE_CSV" ]; then
            echo "ERROR: MHCFLURRY_BASELINE_ALLELE_SEQUENCES does not exist: $BASELINE_CSV" >&2
            exit 1
        fi
        BUILD_39_ARGS+=(--existing-csv "$BASELINE_CSV" --lock-existing)
    elif [ "${MHCFLURRY_ALLOW_UNLOCKED_BUILD:-0}" != "1" ]; then
        echo "ERROR: No baseline allele_sequences.csv found to lock trained alleles." >&2
        echo "Fetch the baseline bundle (mhcflurry downloads fetch allele_sequences)," >&2
        echo "set MHCFLURRY_BASELINE_ALLELE_SEQUENCES=/path/to/allele_sequences.csv," >&2
        echo "or explicitly opt in with MHCFLURRY_ALLOW_UNLOCKED_BUILD=1." >&2
        exit 1
    fi

    time pseudosequence_lookup build "${BUILD_39_ARGS[@]}"
    cp "$MHCFLURRY_PSEUDOSEQUENCES" "$LEGACY_ALLELE_SEQUENCES"

    # Also build the 34-aa NetMHCpan-compatible table (locking the checked-in
    # 34-aa seed table verbatim and extending with mhcseqs).
    BUILD_34_ARGS=(
        --length 34
        --existing-csv "$SCRIPT_DIR/$NETMHCPAN_PSEUDOSEQUENCES"
        --lock-existing
        --out "$LEGACY_NO_DIFFERENTIATION"
    )
    if [ -n "${MHCFLURRY_MHCSEQS_CSV:-}" ]; then
        BUILD_34_ARGS+=(--mhcseqs-csv "$MHCFLURRY_MHCSEQS_CSV")
    fi
    time pseudosequence_lookup build "${BUILD_34_ARGS[@]}"

    # Hard verification gate: assert 0 missing alleles and 0 character diffs on
    # every trained allele in curated_training_data.csv.bz2.
    if [ -n "$BASELINE_CSV" ] && [ "$BASELINE_CSV" != "none" ]; then
        VERIFY_ARGS=(
            --baseline "$BASELINE_CSV"
            --generated "$MHCFLURRY_PSEUDOSEQUENCES"
            --length 39
        )
        if [ -f "$TRAINING_DATA" ]; then
            VERIFY_ARGS+=(--training-data "$TRAINING_DATA")
        fi
        python verify_trained_allele_parity.py "${VERIFY_ARGS[@]}"
    fi

elif [ "$BACKEND" = "legacy" ]; then
    which clustalo
    clustalo --version

    # Generate pseudosequences.
    # Training data is used to decide which additional positions to include in the
    # pseudosequences to differentiate alleles that have identical traditional
    # pseudosequences but have associated training data
    python select_alleles_to_disambiguate.py \
        "$TRAINING_DATA" \
        --min-count 1000 \
        --out training_data.alleles.txt

    # Human
    wget -q ftp://ftp.ebi.ac.uk/pub/databases/ipd/imgt/hla/fasta/A_prot.fasta
    wget -q ftp://ftp.ebi.ac.uk/pub/databases/ipd/imgt/hla/fasta/B_prot.fasta
    wget -q ftp://ftp.ebi.ac.uk/pub/databases/ipd/imgt/hla/fasta/C_prot.fasta
    wget -q ftp://ftp.ebi.ac.uk/pub/databases/ipd/imgt/hla/fasta/E_prot.fasta
    wget -q ftp://ftp.ebi.ac.uk/pub/databases/ipd/imgt/hla/fasta/F_prot.fasta
    wget -q ftp://ftp.ebi.ac.uk/pub/databases/ipd/imgt/hla/fasta/G_prot.fasta

    # Mouse
    wget -q https://www.uniprot.org/uniprot/P01899.fasta  # H-2 Db
    wget -q https://www.uniprot.org/uniprot/P01900.fasta  # H-2 Dd
    wget -q https://www.uniprot.org/uniprot/P14427.fasta  # H-2 Dp
    wget -q https://www.uniprot.org/uniprot/P14426.fasta  # H-2 Dk
    wget -q https://www.uniprot.org/uniprot/Q31145.fasta  # H-2 Dq

    wget -q https://www.uniprot.org/uniprot/P01901.fasta  # H-2 Kb
    wget -q https://www.uniprot.org/uniprot/P01902.fasta  # H-2 Kd
    wget -q https://www.uniprot.org/uniprot/P04223.fasta  # H-2 Kk
    wget -q https://www.uniprot.org/uniprot/P14428.fasta  # H-2 Kq

    wget -q https://www.uniprot.org/uniprot/P01897.fasta  # H-2 Ld
    wget -q https://www.uniprot.org/uniprot/Q31151.fasta  # H-2 Lq

    # Various
    wget -q ftp://ftp.ebi.ac.uk/pub/databases/ipd/mhc/MHC_prot.fasta

    python filter_sequences.py ./*.fasta --out class1.fasta

    time clustalo -i class1.fasta -o class1.aligned.fasta

    time python make_allele_sequences.py \
        class1.aligned.fasta \
        --recapitulate-sequences "$NETMHCPAN_PSEUDOSEQUENCES" \
        --differentiate-alleles training_data.alleles.txt \
        --out-csv "$MHCFLURRY_PSEUDOSEQUENCES"
    cp "$MHCFLURRY_PSEUDOSEQUENCES" "$LEGACY_ALLELE_SEQUENCES"

    time python make_allele_sequences.py \
        class1.aligned.fasta \
        --recapitulate-sequences "$NETMHCPAN_PSEUDOSEQUENCES" \
        --out-csv "$LEGACY_NO_DIFFERENTIATION"

    # Cleanup
    gzip -f class1.fasta
    gzip -f class1.aligned.fasta
    rm ./*.fasta
else
    echo "ERROR: Unknown MHCFLURRY_ALLELE_SEQUENCES_BACKEND=$BACKEND (expected 'mhcseqs' or 'legacy')" >&2
    exit 1
fi

cp "$SCRIPT_ABSOLUTE_PATH" .
bzip2 LOG.txt
RESULT="$SCRATCH_DIR/${DOWNLOAD_NAME}.$(date +%Y%m%d).tar.bz2"
tar -cjf "$RESULT" ./*
echo "Created archive: $RESULT"
