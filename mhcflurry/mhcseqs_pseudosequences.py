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

"""Class I MHC binding-groove pseudosequence extraction backed by ``mhcseqs``.

Extracts fixed-length MHCflurry 39-aa (or NetMHCpan 34-aa) Class I
pseudosequences from arbitrary MHC Class I heavy-chain protein sequences or
from an ``mhcseqs`` sequence table, strictly generalizing the alignment-based
pseudosequence pipeline without requiring external ``clustalo`` or Biopython
installations.

Compatibility with existing MHCflurry 2.x weights is preserved by pinning the
exact 39 mature Class I positions (1-based on the canonical ``HLA-A*02:01``
α1/α2 reference domains):

* 22 α1 (``groove1``) positions:
  ``7, 9, 16, 19, 21, 24, 45, 47, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84``
* 17 α2 (``groove2``) positions:
  ``91, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171``

The 34-aa NetMHCpan pseudosequence is the exact subset obtained by omitting
the 5 MHCflurry disambiguation positions (``16, 19, 21, 47, 91``).
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Mapping, Sequence


# Canonical HLA-A*02:01 mature α1 (90 aa, mature 1..90) and α2 (93 aa, mature
# 91..183) binding-groove reference domains.
HLA_A0201_GROOVE1 = (
    "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEA"
)
HLA_A0201_GROOVE2 = (
    "GSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTD"
)

# 1-based mature Class I coordinates on HLA-A*02:01 (α1 = 1..90, α2 = 91..183).
MHCFLURRY_39AA_MATURE_POSITIONS: tuple[int, ...] = (
    7, 9, 16, 19, 21, 24, 45, 47, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77,
    80, 81, 84,
    91, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163,
    167, 171,
)

# The 5 extra positions added by MHCflurry to disambiguate training alleles
# with identical NetMHCpan 34-aa pseudosequences.
MHCFLURRY_EXTRA_MATURE_POSITIONS: frozenset[int] = frozenset({16, 19, 21, 47, 91})

NETMHCPAN_34AA_MATURE_POSITIONS: tuple[int, ...] = tuple(
    pos for pos in MHCFLURRY_39AA_MATURE_POSITIONS
    if pos not in MHCFLURRY_EXTRA_MATURE_POSITIONS
)

PSEUDOSEQUENCE_MATURE_POSITIONS: dict[int, tuple[int, ...]] = {
    34: NETMHCPAN_34AA_MATURE_POSITIONS,
    39: MHCFLURRY_39AA_MATURE_POSITIONS,
}

# Precomputed 0-based indices into groove1 (len 90) and groove2 (len 93).
_GROOVE_INDICES_BY_LENGTH: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {
    length: (
        tuple(p - 1 for p in positions if p <= len(HLA_A0201_GROOVE1)),
        tuple(p - len(HLA_A0201_GROOVE1) - 1 for p in positions if p > len(HLA_A0201_GROOVE1)),
    )
    for length, positions in PSEUDOSEQUENCE_MATURE_POSITIONS.items()
}

_STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_NON_GROOVE_STATUSES = frozenset({
    "non_groove",
    "missing_groove",
    "too_short",
    "invalid_boundaries",
    "chain_inference_failed",
})
_FRAGMENT_STATUSES = frozenset({
    "short",
    "alpha1_only",
    "alpha2_only",
    "fragment_fallback",
})

# Lazily populated BLOSUM62 scoring tables to keep module import fast.
_BLOSUM_CACHE = None

# Default Gotoh penalties used by the standalone fallback aligner.
_DEFAULT_GAP_OPEN = -10
_DEFAULT_GAP_EXTEND = -1


def _get_mhcseqs_imgt_aligner():
    """Return ``mhcseqs.align_domain_to_imgt`` when mhcseqs >= 2.9.0 is installed."""
    try:
        import mhcseqs
    except ImportError:
        return None
    return getattr(mhcseqs, "align_domain_to_imgt", None)


def _get_blosum_tables():
    """Return ``(aa_to_idx, x_idx, score_matrix)`` backed by BLOSUM62."""
    global _BLOSUM_CACHE
    if _BLOSUM_CACHE is None:
        from . import amino_acid

        matrix = amino_acid.BLOSUM62_MATRIX
        aa_to_idx = {aa: idx for idx, aa in enumerate(matrix.index)}
        x_idx = aa_to_idx["X"]
        score_matrix = matrix.values.astype("int32").tolist()
        _BLOSUM_CACHE = (aa_to_idx, x_idx, score_matrix)
    return _BLOSUM_CACHE


def _sanitize_residue(char: str) -> str:
    """Normalize a single extracted residue to standard 20-AA alphabet or X."""
    c = char.upper()
    return c if c in _STANDARD_AMINO_ACIDS else "X"


def _resolve_groove_indices(length: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(g1_indices, g2_indices)`` for a supported pseudosequence length."""
    try:
        return _GROOVE_INDICES_BY_LENGTH[int(length)]
    except (KeyError, TypeError, ValueError):
        supported = ", ".join(str(k) for k in sorted(_GROOVE_INDICES_BY_LENGTH))
        raise ValueError(
            "Unsupported pseudosequence length %r. Supported lengths: %s."
            % (length, supported)
        ) from None


def _align_domain_to_reference(
    seq: str,
    ref: str,
    positions: Sequence[int],
    *,
    gap_open: int = -10,
    gap_extend: int = -1,
    free_end_gaps: bool = True,
) -> str:
    """Semi-global affine-gap Gotoh alignment of ``seq`` onto ``ref``.

    Parameters
    ----------
    seq : str
        Query groove domain sequence (α1 or α2).
    ref : str
        Canonical reference groove domain sequence (``HLA_A0201_GROOVE1`` or
        ``HLA_A0201_GROOVE2``).
    positions : sequence of int
        0-based reference coordinates to extract.
    gap_open : int, optional
        Affine gap-opening score (negative). Default: -10.
    gap_extend : int, optional
        Affine gap-extension score (negative). Default: -1.
    free_end_gaps : bool, optional
        If True (default), terminal gaps at either end of ``seq`` or ``ref``
        are unpenalized so partial exon-2/exon-3 fragments align accurately.

    Returns
    -------
    str
        Extracted residues at ``positions`` (``"X"`` where gapped or missing).
    """
    n = len(ref)
    m = len(seq)
    if m == 0:
        return "X" * len(positions)

    aa_to_idx, x_idx, score_mat = _get_blosum_tables()
    ref_idx = [aa_to_idx.get(c, x_idx) for c in ref]
    seq_upper = seq.upper()
    seq_idx = [aa_to_idx.get(c, x_idx) for c in seq_upper]

    neg_inf = -(10**9)
    m_dp = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    ix_dp = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    iy_dp = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    # Traceback codes: 0 = match/mismatch (M), 1 = gap in seq (Ix, up),
    # 2 = gap in ref (Iy, left).
    tb_m = [bytearray(m + 1) for _ in range(n + 1)]
    tb_ix = [bytearray(m + 1) for _ in range(n + 1)]
    tb_iy = [bytearray(m + 1) for _ in range(n + 1)]

    m_dp[0][0] = 0
    for i in range(1, n + 1):
        ix_dp[i][0] = 0 if free_end_gaps else (gap_open + (i - 1) * gap_extend)
        tb_ix[i][0] = 1 if i > 1 else 0
    for j in range(1, m + 1):
        iy_dp[0][j] = 0 if free_end_gaps else (gap_open + (j - 1) * gap_extend)
        tb_iy[0][j] = 2 if j > 1 else 0

    for i in range(1, n + 1):
        row_s = score_mat[ref_idx[i - 1]]
        m_prev, ix_prev, iy_prev = m_dp[i - 1], ix_dp[i - 1], iy_dp[i - 1]
        m_cur, ix_cur, iy_cur = m_dp[i], ix_dp[i], iy_dp[i]
        tbm_i, tbix_i, tbiy_i = tb_m[i], tb_ix[i], tb_iy[i]
        lo_i = 0 if (free_end_gaps and i == n) else gap_open
        le_i = 0 if (free_end_gaps and i == n) else gap_extend
        for j in range(1, m + 1):
            s = row_s[seq_idx[j - 1]]
            v0 = m_prev[j - 1]
            v1 = ix_prev[j - 1]
            v2 = iy_prev[j - 1]
            if v0 >= v1 and v0 >= v2:
                m_cur[j] = v0 + s
                tbm_i[j] = 0
            elif v1 >= v2:
                m_cur[j] = v1 + s
                tbm_i[j] = 1
            else:
                m_cur[j] = v2 + s
                tbm_i[j] = 2

            if free_end_gaps and j == m:
                u0 = m_prev[j]
                u1 = ix_prev[j]
            else:
                u0 = m_prev[j] + gap_open
                u1 = ix_prev[j] + gap_extend
            if u0 >= u1:
                ix_cur[j] = u0
                tbix_i[j] = 0
            else:
                ix_cur[j] = u1
                tbix_i[j] = 1

            w0 = m_cur[j - 1] + lo_i
            w2 = iy_cur[j - 1] + le_i
            if w0 >= w2:
                iy_cur[j] = w0
                tbiy_i[j] = 0
            else:
                iy_cur[j] = w2
                tbiy_i[j] = 2

    e0, e1, e2 = m_dp[n][m], ix_dp[n][m], iy_dp[n][m]
    if e0 >= e1 and e0 >= e2:
        state = 0
    elif e1 >= e2:
        state = 1
    else:
        state = 2

    i, j = n, m
    ref_to_char = ["X"] * n
    while i > 0 or j > 0:
        if state == 0:
            prev = tb_m[i][j]
            ref_to_char[i - 1] = _sanitize_residue(seq_upper[j - 1])
            i -= 1
            j -= 1
            state = prev
        elif state == 1:
            prev = tb_ix[i][j]
            ref_to_char[i - 1] = "X"
            i -= 1
            state = prev
        else:
            prev = tb_iy[i][j]
            j -= 1
            state = prev

    return "".join(ref_to_char[p] for p in positions)


def _extract_from_imgt_domain_alignment(
    domain_alignment,
    mature_positions: Sequence[int],
) -> str:
    """Extract sanitized residues at canonical mature positions from a DomainAlignment."""
    mature_to_aa = domain_alignment.mature_to_aa
    chars = []
    for pos in mature_positions:
        aa = mature_to_aa.get(pos)
        if aa is None or aa == "-":
            chars.append("X")
        else:
            chars.append(_sanitize_residue(aa))
    return "".join(chars)


def _extract_groove1_residues(
    groove1: str,
    g1_positions: Sequence[int],
    *,
    gap_open: int = _DEFAULT_GAP_OPEN,
    gap_extend: int = _DEFAULT_GAP_EXTEND,
    cache: dict[str, str] | None = None,
    imgt_aligner=None,
) -> str:
    """Extract α1 pseudosequence residues via mhcseqs IMGT alignment or fallback."""
    g1 = (groove1 or "").strip().upper()
    if not g1:
        return "X" * len(g1_positions)
    if cache is not None and g1 in cache:
        return cache[g1]

    if (
        imgt_aligner is not None
        and gap_open == _DEFAULT_GAP_OPEN
        and gap_extend == _DEFAULT_GAP_EXTEND
    ):
        domain_aln = imgt_aligner(g1, domain="G-ALPHA1")
        mature_positions = tuple(p + 1 for p in g1_positions)
        result = _extract_from_imgt_domain_alignment(domain_aln, mature_positions)
    else:
        seq_len = len(g1)
        # Fast path 1: canonical 90-aa α1 verified by conserved structural anchors
        # (Tyr27 at index 26, Trp60 at index 59, Asn86-Gln87 glycosylation motif).
        if seq_len == 90 and g1[26] == "Y" and g1[59] == "W" and g1[85:87] == "NQ":
            result = "".join(_sanitize_residue(g1[p]) for p in g1_positions)
        # Fast path 2: IMGT partial exon-2 sequence missing only mature residue 1
        # (leading Gly), leaving 89 aa shifted by -1.
        elif (
            seq_len == 89
            and g1[25] == "Y"
            and g1[58] == "W"
            and g1[84:86] == "NQ"
            and g1[0] != "G"
        ):
            result = "".join(_sanitize_residue(g1[p - 1]) for p in g1_positions)
        else:
            result = _align_domain_to_reference(
                g1,
                HLA_A0201_GROOVE1,
                g1_positions,
                gap_open=gap_open,
                gap_extend=gap_extend,
            )

    if cache is not None:
        cache[g1] = result
    return result


def _extract_groove2_residues(
    groove2: str,
    g2_positions: Sequence[int],
    *,
    gap_open: int = _DEFAULT_GAP_OPEN,
    gap_extend: int = _DEFAULT_GAP_EXTEND,
    cache: dict[str, str] | None = None,
    imgt_aligner=None,
) -> str:
    """Extract α2 pseudosequence residues via mhcseqs IMGT alignment or fallback."""
    g2 = (groove2 or "").strip().upper()
    if not g2:
        return "X" * len(g2_positions)
    if cache is not None and g2 in cache:
        return cache[g2]

    if (
        imgt_aligner is not None
        and gap_open == _DEFAULT_GAP_OPEN
        and gap_extend == _DEFAULT_GAP_EXTEND
    ):
        domain_aln = imgt_aligner(g2, domain="G-ALPHA2")
        mature_positions = tuple(p + len(HLA_A0201_GROOVE1) + 1 for p in g2_positions)
        result = _extract_from_imgt_domain_alignment(domain_aln, mature_positions)
    else:
        # Always align in the fallback path: conserved Cys101/Cys164 spacing
        # alone does not guard against compensating internal indels between
        # the disulfide cysteines (which shift α2 binding-pocket residues).
        result = _align_domain_to_reference(
            g2,
            HLA_A0201_GROOVE2,
            g2_positions,
            gap_open=gap_open,
            gap_extend=gap_extend,
        )

    if cache is not None:
        cache[g2] = result
    return result


def extract_class1_pseudosequence_from_grooves(
    groove1: str,
    groove2: str,
    *,
    length: int = 39,
    gap_open: int = _DEFAULT_GAP_OPEN,
    gap_extend: int = _DEFAULT_GAP_EXTEND,
) -> str:
    """Extract a fixed-length Class I pseudosequence from α1 and α2 domains.

    When ``mhcseqs >= 2.9.0`` is installed (and default alignment parameters
    are used), delegates to ``mhcseqs.align_domain_to_imgt``; otherwise uses
    the built-in BLOSUM62 affine-gap semi-global aligner.

    Parameters
    ----------
    groove1 : str
        Class I α1 (G-ALPHA1) binding-groove domain sequence.
    groove2 : str
        Class I α2 (G-ALPHA2) binding-groove domain sequence.
    length : int, optional
        Target pseudosequence length: ``39`` (MHCflurry 2.x, default) or
        ``34`` (NetMHCpan).
    gap_open : int, optional
        Affine gap-opening score for non-canonical lengths. Default: -10.
    gap_extend : int, optional
        Affine gap-extension score for non-canonical lengths. Default: -1.

    Returns
    -------
    str
        Fixed-length pseudosequence of length ``length``. Missing or gapped
        positions are represented as ``"X"``.
    """
    g1_positions, g2_positions = _resolve_groove_indices(length)
    imgt_aligner = _get_mhcseqs_imgt_aligner()
    p1 = _extract_groove1_residues(
        groove1,
        g1_positions,
        gap_open=gap_open,
        gap_extend=gap_extend,
        imgt_aligner=imgt_aligner,
    )
    p2 = _extract_groove2_residues(
        groove2,
        g2_positions,
        gap_open=gap_open,
        gap_extend=gap_extend,
        imgt_aligner=imgt_aligner,
    )
    return p1 + p2


def _import_mhcseqs():
    """Import ``mhcseqs`` lazily with an actionable error if unavailable."""
    try:
        import mhcseqs
    except ImportError as exc:
        raise ImportError(
            "mhcseqs is required to decompose full MHC protein sequences or "
            "build pseudosequences from the mhcseqs database. Install it with: "
            "pip install 'mhcseqs>=2.7.0'"
        ) from exc
    return mhcseqs


def extract_class1_pseudosequence_with_alignment(
    sequence: str,
    *,
    length: int = 39,
    allele: str = "",
    gene: str = "",
    mutations: Sequence[str] = (),
    allow_fragments: bool = False,
    warn_on_indels: bool = True,
    warn_on_noop_mutations: bool = True,
    gap_open: int = _DEFAULT_GAP_OPEN,
    gap_extend: int = _DEFAULT_GAP_EXTEND,
):
    """Extract a Class I pseudosequence together with its ``mhcseqs`` IMGT alignment.

    Returns ``(pseudosequence, groove_alignment)`` where ``groove_alignment``
    is a ``mhcseqs.GrooveAlignment`` (when ``mhcseqs >= 2.9.0`` is installed
    and default alignment penalties are used) or ``None`` on older ``mhcseqs``
    releases.
    """
    _resolve_groove_indices(length)
    cleaned = (sequence or "").strip()
    if not cleaned:
        raise ValueError("Cannot extract Class I pseudosequence from an empty sequence.")

    if isinstance(mutations, str):
        normalized_mutations = (mutations.strip(),) if mutations.strip() else ()
    elif mutations:
        normalized_mutations = tuple(
            str(m).strip() for m in mutations if str(m).strip()
        )
    else:
        normalized_mutations = ()

    mhcseqs = _import_mhcseqs()
    record = mhcseqs.decompose_domains(
        cleaned,
        mhc_class="I",
        allele=allele,
        gene=gene,
        mutations=normalized_mutations,
    )
    if record.status in _NON_GROOVE_STATUSES:
        raise ValueError(
            "Sequence %s does not contain a Class I binding groove "
            "(mhcseqs status=%r, flags=%r)."
            % (allele or "<unnamed>", record.status, record.flags)
        )
    has_g1 = bool(record.groove1)
    has_g2 = bool(record.groove2)
    record_status = (record.status or "").strip().lower()
    if not allow_fragments and (
        not (has_g1 and has_g2) or record_status in _FRAGMENT_STATUSES
    ):
        raise ValueError(
            "Incomplete Class I groove for %s (groove1=%d aa, groove2=%d aa, "
            "status=%r). Pass allow_fragments=True to allow partial/short "
            "grooves and pad missing residues with 'X'."
            % (allele or "<unnamed>", len(record.groove1), len(record.groove2), record.status)
        )
    if not (has_g1 or has_g2):
        raise ValueError(
            "No Class I groove domains extracted for %s (status=%r)."
            % (allele or "<unnamed>", record.status)
        )

    align_groove = getattr(mhcseqs, "align_groove_to_imgt", None)
    groove_alignment = None
    if (
        align_groove is not None
        and gap_open == _DEFAULT_GAP_OPEN
        and gap_extend == _DEFAULT_GAP_EXTEND
    ):
        groove_alignment = align_groove(record)
        positions = PSEUDOSEQUENCE_MATURE_POSITIONS[int(length)]
        raw_pseudo = groove_alignment.extract_positions(
            positions, missing="X", deleted="X"
        )
        pseudosequence = "".join(_sanitize_residue(c) for c in raw_pseudo)
        if warn_on_indels and groove_alignment.has_insertions:
            logging.warning(
                "Class I groove for %s contains internal insertion(s) (%s) "
                "that are omitted by fixed %d-aa pseudosequences; predictions "
                "for alleles with pocket-reshaping insertions may be less accurate.",
                allele or "<unnamed>",
                groove_alignment.summary(),
                int(length),
            )
    else:
        pseudosequence = extract_class1_pseudosequence_from_grooves(
            record.groove1,
            record.groove2,
            length=length,
            gap_open=gap_open,
            gap_extend=gap_extend,
        )

    if normalized_mutations and warn_on_noop_mutations:
        wt_record = mhcseqs.decompose_domains(
            cleaned,
            mhc_class="I",
            allele=allele,
            gene=gene,
        )
        if wt_record.status not in _NON_GROOVE_STATUSES and (
            wt_record.groove1 or wt_record.groove2
        ):
            wt_pseudo = extract_class1_pseudosequence_from_grooves(
                wt_record.groove1,
                wt_record.groove2,
                length=length,
                gap_open=gap_open,
                gap_extend=gap_extend,
            )
            if wt_pseudo == pseudosequence:
                logging.warning(
                    "Mutation(s) %r on %s did not change any of the %d "
                    "extracted pseudosequence positions; predicted binding "
                    "will be identical to wild-type.",
                    normalized_mutations,
                    allele or "<unnamed>",
                    int(length),
                )

    return pseudosequence, groove_alignment


def extract_class1_pseudosequence(
    sequence: str,
    *,
    length: int = 39,
    allele: str = "",
    gene: str = "",
    mutations: Sequence[str] = (),
    allow_fragments: bool = False,
    warn_on_indels: bool = True,
    warn_on_noop_mutations: bool = True,
    gap_open: int = _DEFAULT_GAP_OPEN,
    gap_extend: int = _DEFAULT_GAP_EXTEND,
) -> str:
    """Extract a Class I pseudosequence from a raw MHC protein sequence.

    Uses ``mhcseqs.decompose_domains(..., mhc_class="I")`` to strip any signal
    peptide (including truncated N-terminal signal peptides), identify the α1
    (``groove1``) and α2 (``groove2``) binding-groove domains, optionally apply
    IEDB-style point mutations (e.g. ``mutations=("K66A",)``), and extract the
    pinned binding groove positions.

    Parameters
    ----------
    sequence : str
        Full-length, mature, or partial MHC Class I heavy-chain amino acid
        sequence.
    length : int, optional
        Target pseudosequence length: ``39`` (MHCflurry 2.x, default) or
        ``34`` (NetMHCpan).
    allele : str, optional
        Optional allele name passed to ``mhcseqs.decompose_domains`` for
        metadata and gene/species cues.
    gene : str, optional
        Optional gene symbol passed to ``mhcseqs.decompose_domains``.
    mutations : sequence of str, optional
        Optional mature-numbered point mutations (e.g. ``["W167S"]`` or
        ``"W167S"``) applied by ``mhcseqs`` before groove extraction.
    allow_fragments : bool, optional
        If False (default), require both α1 and α2 groove domains to be
        present. If True, allow partial single-domain fragments (filling the
        missing domain with ``"X"``).
    warn_on_indels : bool, optional
        When True (default) and ``mhcseqs >= 2.9.0`` detects internal
        insertions in the binding groove, log a warning noting that fixed-width
        pseudosequences omit inserted residues.
    warn_on_noop_mutations : bool, optional
        When True (default) and ``mutations`` does not change any of the
        extracted pseudosequence positions, log a warning.
    gap_open : int, optional
        Affine gap-opening score for non-canonical groove lengths.
    gap_extend : int, optional
        Affine gap-extension score for non-canonical groove lengths.

    Returns
    -------
    str
        Fixed-length Class I pseudosequence of length ``length``.

    Raises
    ------
    ValueError
        If ``sequence`` is empty, not a valid Class I groove (e.g. Ig-only or
        non-groove gene), or missing a required groove domain.
    """
    pseudosequence, _alignment = extract_class1_pseudosequence_with_alignment(
        sequence,
        length=length,
        allele=allele,
        gene=gene,
        mutations=mutations,
        allow_fragments=allow_fragments,
        warn_on_indels=warn_on_indels,
        warn_on_noop_mutations=warn_on_noop_mutations,
        gap_open=gap_open,
        gap_extend=gap_extend,
    )
    return pseudosequence


# Standard 6- or 10-character UniProtKB accessions (without MHC separators).
# Guarded before mhcgnomes parsing so accessions like Japanese medaka fish
# H2LR16..H2LR44 cannot be misparsed as mouse H2-L*r16..r44 (mhcgnomes#306).
_BARE_UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)


def _is_bare_uniprot_accession(raw_name: str) -> bool:
    """Return True if ``raw_name`` is a bare UniProtKB accession string."""
    text = (raw_name or "").strip()
    if not text or "-" in text or "*" in text or ":" in text:
        return False
    return bool(_BARE_UNIPROT_ACCESSION_RE.match(text))


def _normalize_pseudosequence_allele_key(raw_name: str, prefer_no_alias: bool = True) -> str | None:
    """Normalize an allele key with mhcgnomes, matching predictor persistence."""
    from .common import normalize_allele_name

    if not raw_name or _is_bare_uniprot_accession(raw_name):
        return None
    if prefer_no_alias:
        normalized = normalize_allele_name(
            raw_name, raise_on_error=False, use_allele_aliases=False
        )
        if normalized is not None:
            return normalized
    return normalize_allele_name(
        raw_name, raise_on_error=False, use_allele_aliases=True
    )


def build_pseudosequences_from_mhcseqs(
    sequences_csv: str | None = None,
    *,
    rows: Iterable[Mapping[str, object]] | None = None,
    length: int = 39,
    existing_allele_to_sequence: Mapping[str, str] | None = None,
    prefer_existing: bool = False,
    lock_existing: bool = False,
    max_x: int | None = None,
    include_fragments: bool = False,
    normalize_alleles: bool = True,
    gap_open: int = -10,
    gap_extend: int = -1,
) -> dict[str, str]:
    """Build a Class I ``allele -> pseudosequence`` mapping from ``mhcseqs``.

    Parameters
    ----------
    sequences_csv : str, optional
        Path to ``mhc-full-seqs.csv``. Ignored when ``rows`` is supplied. When
        both are None, uses ``mhcseqs.load_sequences_dict()``.
    rows : iterable of mapping, optional
        Pre-loaded ``mhcseqs`` row mappings (each containing at least
        ``mhc_class``, ``two_field_allele`` or ``allele``, ``groove1``, and
        ``groove2``).
    length : int, optional
        Pseudosequence length: ``39`` (MHCflurry 2.x, default) or ``34``
        (NetMHCpan).
    existing_allele_to_sequence : mapping of str -> str, optional
        Optional existing pseudosequence table to strictly generalize. Every
        valid entry in ``existing_allele_to_sequence`` is retained; new alleles
        from ``mhcseqs`` are added.
    prefer_existing : bool, optional
        Only relevant when ``existing_allele_to_sequence`` is given. When True,
        keep existing non-``X`` pseudosequences verbatim so predictions on
        previously supported alleles never drift (while still replacing
        ``X``-containing entries if ``mhcseqs`` provides strictly fewer ``X``
        positions). When False (default), ``mhcseqs`` values overwrite existing
        entries unless the existing entry has strictly fewer ``X`` positions.
    lock_existing : bool, optional
        Only relevant when ``existing_allele_to_sequence`` is given. When True,
        every existing entry is locked bit-for-bit verbatim (even entries
        containing ``"X"``, and exempt from ``max_x`` filtering) so previously
        trained alleles can never change; ``mhcseqs`` only adds alleles not
        already present in ``existing_allele_to_sequence``.
    max_x : int, optional
        Maximum allowed ``"X"`` characters in emitted pseudosequences. Entries
        exceeding ``max_x`` are omitted (except entries locked via
        ``lock_existing=True``).
    include_fragments : bool, optional
        If False (default), skip records missing either α1 or α2.
    normalize_alleles : bool, optional
        If True (default), canonicalize allele names with ``mhcgnomes`` via
        :func:`mhcflurry.common.normalize_allele_name`, skipping non-Class-I,
        null, questionable, pseudogene, or unparseable names so every emitted
        key is directly usable by ``Class1AffinityPredictor``. If False, retain
        raw ``two_field_allele`` identifiers (including UniProt accessions).
    gap_open : int, optional
        Affine gap-opening score for non-canonical groove lengths.
    gap_extend : int, optional
        Affine gap-extension score for non-canonical groove lengths.

    Returns
    -------
    dict of str -> str
        Sorted ``{allele: pseudosequence}`` mapping.
    """
    g1_positions, g2_positions = _resolve_groove_indices(length)

    if rows is None:
        mhcseqs = _import_mhcseqs()
        rows = mhcseqs.load_sequences_dict(sequences_csv)

    result: dict[str, str] = {}
    existing_keys: set[str] = set()

    def _clean_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and value != value:
            return ""
        text = str(value).strip()
        if text.lower() in ("nan", "<na>"):
            return ""
        return text

    if existing_allele_to_sequence:
        for raw_allele, seq in existing_allele_to_sequence.items():
            seq_str = _clean_text(seq).upper()
            if not seq_str:
                continue
            if len(seq_str) != length:
                raise ValueError(
                    "Existing pseudosequence for %r has length %d, expected %d."
                    % (raw_allele, len(seq_str), length)
                )
            key = (
                _normalize_pseudosequence_allele_key(str(raw_allele))
                if normalize_alleles
                else str(raw_allele).strip()
            )
            if not key:
                continue
            if key in result and seq_str.count("X") >= result[key].count("X"):
                continue
            result[key] = seq_str
            existing_keys.add(key)

    g1_cache: dict[str, str] = {}
    g2_cache: dict[str, str] = {}
    imgt_aligner = _get_mhcseqs_imgt_aligner()
    skipped_unparseable = 0

    for row in rows:
        mhc_class = _clean_text(row.get("mhc_class")).upper()
        if mhc_class != "I":
            continue
        status = (
            _clean_text(row.get("groove_status")) or _clean_text(row.get("status"))
        ).lower()
        if status in _NON_GROOVE_STATUSES:
            continue
        if not include_fragments and status in _FRAGMENT_STATUSES:
            continue

        groove1 = _clean_text(row.get("groove1"))
        groove2 = _clean_text(row.get("groove2"))
        if not include_fragments and not (groove1 and groove2):
            continue
        if not (groove1 or groove2):
            continue

        raw_name = (
            _clean_text(row.get("two_field_allele"))
            or _clean_text(row.get("allele"))
            or _clean_text(row.get("representative_allele"))
        )
        if not raw_name:
            continue

        if normalize_alleles:
            key = _normalize_pseudosequence_allele_key(raw_name)
            if key is None:
                skipped_unparseable += 1
                continue
        else:
            key = raw_name

        pseudo = (
            _extract_groove1_residues(
                groove1,
                g1_positions,
                gap_open=gap_open,
                gap_extend=gap_extend,
                cache=g1_cache,
                imgt_aligner=imgt_aligner,
            )
            + _extract_groove2_residues(
                groove2,
                g2_positions,
                gap_open=gap_open,
                gap_extend=gap_extend,
                cache=g2_cache,
                imgt_aligner=imgt_aligner,
            )
        )
        if max_x is not None and pseudo.count("X") > max_x:
            continue

        if key in result:
            if key in existing_keys and lock_existing:
                # Exact-lock mode: never alter any existing entry, even if it has X's.
                continue
            existing_seq = result[key]
            new_x = pseudo.count("X")
            old_x = existing_seq.count("X")
            if key in existing_keys and prefer_existing:
                # Strict compatibility mode: only replace if mhcseqs fills in X's.
                if new_x < old_x:
                    result[key] = pseudo
            else:
                if new_x <= old_x:
                    result[key] = pseudo
        else:
            result[key] = pseudo

    if max_x is not None:
        result = {
            k: v
            for k, v in result.items()
            if (lock_existing and k in existing_keys) or v.count("X") <= max_x
        }

    if skipped_unparseable:
        logging.debug(
            "Skipped %d mhcseqs Class I rows that could not be canonicalized "
            "to a Class I allele name by mhcgnomes (e.g. unannotated UniProt accessions).",
            skipped_unparseable,
        )

    return dict(sorted(result.items()))


def extend_allele_to_sequence_from_mhcseqs(
    allele_to_sequence: Mapping[str, str],
    *,
    sequences_csv: str | None = None,
    rows: Iterable[Mapping[str, object]] | None = None,
    prefer_existing: bool = True,
    lock_existing: bool = False,
    max_x: int | None = None,
    include_fragments: bool = False,
    normalize_alleles: bool = True,
) -> dict[str, str]:
    """Return a strictly generalized copy of ``allele_to_sequence`` using ``mhcseqs``.

    Infers the pseudosequence length (39 or 34) from ``allele_to_sequence``,
    preserves all existing non-``X`` entries verbatim by default
    (``prefer_existing=True``), upgrades ``X``-containing entries when
    ``mhcseqs`` supplies more complete grooves (unless ``lock_existing=True``,
    which freezes every existing entry verbatim), and adds all newly covered
    Class I alleles from ``mhcseqs``.
    """
    from .pseudosequences import pseudosequence_length

    length = pseudosequence_length(allele_to_sequence)
    if length not in PSEUDOSEQUENCE_MATURE_POSITIONS:
        raise ValueError(
            "Cannot infer supported pseudosequence length (34 or 39) from "
            "allele_to_sequence (inferred length=%r)." % (length,)
        )
    return build_pseudosequences_from_mhcseqs(
        sequences_csv=sequences_csv,
        rows=rows,
        length=length,
        existing_allele_to_sequence=allele_to_sequence,
        prefer_existing=prefer_existing,
        lock_existing=lock_existing,
        max_x=max_x,
        include_fragments=include_fragments,
        normalize_alleles=normalize_alleles,
    )
