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

"""
GPU motif-summary helpers used by the fast percentile-rank calibration path.

These are stateless tensor/numpy routines: given a chunk of predicted IC50s
for a set of (allele, peptide) pairs, they reproduce the per-allele
``DataFrame(...).drop_duplicates().groupby('length').nsmallest(...)`` motif
summary (top-binder amino-acid frequency matrices + length distributions) of
the legacy slow path, but vectorized on the calibration device.

They live in their own module — rather than on
:class:`~mhcflurry.class1_affinity_predictor.Class1AffinityPredictor` — because
they hold no predictor state and are unit-tested in isolation (see
``test/test_calibrate_percentile_ranks_fast.py``). The predictor only
orchestrates: it builds the per-worker state once via
:func:`prepare_motif_summary_state_gpu` and then calls
:func:`motif_summary_chunk_gpu` per allele chunk.
"""
import numpy
import pandas

from .amino_acid import AMINO_ACID_INDEX, AMINO_ACIDS


def prepare_motif_summary_state_gpu(encoded_peptides, device):
    """One-time setup for GPU motif_summary; lifts the per-allele
    ``drop_duplicates`` + length-bucket + AA-encoding work out of
    the calibrate chunk loop so each chunk is pure tensor math.

    Returns a dict of device-resident tensors:
        unique_idx_t : (n_unique,) long — first-occurrence indices
            into the full peptide list. Selecting columns with this
            from the chunk's ``ic50_device`` reproduces the legacy
            ``drop_duplicates('peptide')`` semantics (first row wins).
        length_groups : dict[L] -> (n_at_L,) long indices into the
            unique-peptide axis for peptides of length L.
        aa_codes_per_length : dict[L] -> (n_at_L, L) long tensor of
            amino-acid index codes (matches ``AMINO_ACID_INDEX``,
            so X = 20 if it ever appears).
        unique_lengths_t : (n_unique,) long — peptide length per
            unique peptide; powers the per-allele length distribution.
        n_unique : int.
    """
    import torch

    seqs = list(encoded_peptides.sequences)
    seen = set()
    unique_idx = []
    for i, s in enumerate(seqs):
        if s not in seen:
            seen.add(s)
            unique_idx.append(i)
    unique_idx = numpy.asarray(unique_idx, dtype=numpy.int64)
    unique_seqs = [seqs[i] for i in unique_idx]
    lengths = numpy.fromiter(
        (len(p) for p in unique_seqs),
        dtype=numpy.int64,
        count=len(unique_seqs),
    )

    unique_lengths_t = torch.from_numpy(lengths).to(device)
    unique_idx_t = torch.from_numpy(unique_idx).to(device)

    # ASCII -> AA-index lookup so the per-residue encoding is one
    # vectorized numpy gather instead of 8M Python ops over 800k
    # peptides. Unknown bytes map to X and the X bucket is dropped
    # from the final 20-column matrix, matching
    # positional_frequency_matrix's treatment of non-common letters.
    ascii_lut = numpy.full(
        256, AMINO_ACID_INDEX["X"], dtype=numpy.int64,
    )
    for letter, idx in AMINO_ACID_INDEX.items():
        if len(letter) == 1:
            ascii_lut[ord(letter)] = idx

    length_groups = {}
    aa_codes_per_length = {}
    for L_np in numpy.unique(lengths):
        L = int(L_np)
        sel = numpy.where(lengths == L)[0].astype(numpy.int64)
        n_at_L = int(sel.shape[0])
        joined = "".join(unique_seqs[i] for i in sel).encode("ascii")
        byte_arr = numpy.frombuffer(
            joined, dtype=numpy.uint8,
        ).reshape(n_at_L, L)
        codes = ascii_lut[byte_arr]  # (n_at_L, L) int64
        length_groups[L] = torch.from_numpy(sel).to(device)
        aa_codes_per_length[L] = torch.from_numpy(
            numpy.ascontiguousarray(codes),
        ).to(device)

    return {
        "unique_idx_t": unique_idx_t,
        "length_groups": length_groups,
        "aa_codes_per_length": aa_codes_per_length,
        "unique_lengths_t": unique_lengths_t,
        "n_unique": int(unique_idx.shape[0]),
    }


def topk_first_tie_indices(values, k):
    """Return k smallest column indices per row with pandas tie semantics.

    ``torch.topk`` is fast but does not promise which entries it
    returns when the kth value is tied. Pandas ``Series.nsmallest``
    defaults to ``keep='first'``; after ``drop_duplicates('peptide')``,
    "first" means lower position on the unique-peptide axis. Keep
    the fast ``topk`` path for normal continuous predictions and
    repair only rows whose cutoff falls inside an exact tie block.
    """
    import torch

    n = int(values.shape[1])
    if k >= n:
        return torch.arange(
            n, dtype=torch.long, device=values.device,
        ).expand(int(values.shape[0]), n)

    top = torch.topk(values, k, dim=1, largest=False, sorted=False)
    top_idx = top.indices

    boundary = top.values.max(dim=1).values.unsqueeze(1)
    lt_counts = (values < boundary).sum(dim=1)
    eq_counts = (values == boundary).sum(dim=1)
    tie_needed = k - lt_counts
    tied_cutoff_rows = eq_counts > tie_needed
    if not bool(tied_cutoff_rows.any().item()):
        return top_idx

    top_idx = top_idx.clone()
    row_ids = torch.nonzero(
        tied_cutoff_rows, as_tuple=False,
    ).flatten()
    row_values = values.index_select(0, row_ids)
    row_boundary = boundary.index_select(0, row_ids)
    row_lt = row_values < row_boundary
    row_eq = row_values == row_boundary
    row_tie_needed = tie_needed.index_select(0, row_ids).unsqueeze(1)
    eq_rank = row_eq.to(torch.int64).cumsum(dim=1)
    selected = row_lt | (row_eq & (eq_rank <= row_tie_needed))
    stable_idx = selected.nonzero(as_tuple=False)[:, 1].reshape(
        int(row_ids.numel()), k,
    )
    top_idx[row_ids] = stable_idx
    return top_idx


def motif_summary_chunk_gpu(
        ic50_device,
        state,
        summary_top_peptide_fractions,
        batch_alleles):
    """GPU motif_summary for one allele chunk.

    Replaces the per-allele
    ``DataFrame(...).drop_duplicates().groupby('length').nsmallest(...)``
    block in the fast path with vectorized tensor ops:

    * ``torch.topk(largest=False)`` selects top-k tightest binders
      per (allele, length), with a small tie repair to match pandas
      ``nsmallest(keep='first')`` at exact cutoff ties.
    * AA frequency matrices are computed by gathering precomputed
      per-length AA-code tensors and ``scatter_add`` into a one-hot
      counts buffer of shape ``(a_size, L, 21)``; pandas only
      assembles the final per-row schema.
    * Length distributions are ``topk`` over the full
      unique-peptide axis followed by a per-row ``scatter_add``
      (a row-wise ``bincount``).

    All math runs on the calibration device; per-batch ``.cpu()``
    transfers happen once per (cutoff_fraction, length) at chunk-end
    instead of inside the per-allele Python loop, and pandas only
    stamps the persistent per-row schema on the consolidated
    per-block tensors. Returns ``(freq_matrix_dfs, length_dist_dfs)``
    — lists of ``pandas.DataFrame`` matching the legacy slow path's
    per-row schema, ready to ``pandas.concat`` once all chunks are
    done.
    """
    import torch

    a_size = len(batch_alleles)
    device = ic50_device.device
    count_dtype = torch.float32 if device.type == "mps" else torch.float64
    n_unique = state["n_unique"]
    # AMINO_ACIDS = COMMON_AMINO_ACIDS_WITH_UNKNOWN keys, alphabetical
    # then X — first 20 entries are the BLOSUM62-ordered non-X rows
    # the legacy ``positional_frequency_matrix`` returns.
    aa_columns = AMINO_ACIDS[:20]

    ic50_unique = ic50_device.index_select(1, state["unique_idx_t"])

    # Stage 1 (Torch on device): build all freq + length-dist tensors.
    # Each entry is (cutoff_fraction, k, L, freq_t (a_size, L, 20)).
    freq_tensors = []
    # Each entry is (cutoff_fraction, k_total, length_fractions_t (a_size, max_len_p1)).
    length_fraction_tensors = []

    for cutoff_fraction in summary_top_peptide_fractions:
        for L, idx_in_unique in state["length_groups"].items():
            n_at_L = int(idx_in_unique.numel())
            k = min(max(int(n_at_L * cutoff_fraction), 1), n_at_L)
            ic50_L = ic50_unique.index_select(1, idx_in_unique)
            top_idx = topk_first_tie_indices(
                ic50_L, k,
            )  # (a_size, k)
            codes = state["aa_codes_per_length"][L]  # (n_at_L, L) long
            selected_codes = codes[top_idx]  # (a_size, k, L) long
            # Permute to (a_size, L, k) so scatter_add packs counts
            # along the last (AA) axis. We allocate 21 AA slots to
            # absorb X (index 20) and discard the X column at the
            # end — this matches the legacy semantics where
            # ``positional_frequency_matrix``'s row index excludes X
            # while the divisor stays at ``k`` (so positions with
            # any X residues sum to <1 across the 20 columns).
            scatter_dst = selected_codes.permute(0, 2, 1)
            counts = torch.zeros(
                a_size, L, 21, dtype=count_dtype, device=device,
            )
            counts.scatter_add_(
                2, scatter_dst,
                torch.ones_like(scatter_dst, dtype=count_dtype),
            )
            freq_t = counts[:, :, :20] / float(k)  # device tensor
            freq_tensors.append(
                (float(cutoff_fraction), int(k), int(L), freq_t),
            )

        k_total = min(max(int(n_unique * cutoff_fraction), 1), n_unique)
        top_full_idx = topk_first_tie_indices(
            ic50_unique, k_total,
        )  # (a_size, k_total)
        lengths_per_topk = state["unique_lengths_t"][top_full_idx]
        max_len_p1 = int(state["unique_lengths_t"].max().item()) + 1
        length_counts = torch.zeros(
            a_size, max_len_p1, dtype=count_dtype, device=device,
        )
        length_counts.scatter_add_(
            1, lengths_per_topk,
            torch.ones_like(lengths_per_topk, dtype=count_dtype),
        )
        length_fractions_t = length_counts / float(k_total)
        length_fraction_tensors.append(
            (float(cutoff_fraction), int(k_total), length_fractions_t),
        )

    # Stage 2 (single device→host transfer per logical block): build
    # one wide-format pandas DataFrame per (cutoff_fraction, L) in
    # one shot — the per-allele Python loop is replaced by a flat
    # reshape + numpy.repeat for the allele/position axes.
    batch_alleles_arr = numpy.asarray(batch_alleles)

    freq_matrices = []
    for cutoff_fraction, k, L, freq_t in freq_tensors:
        freq_arr = freq_t.cpu().numpy()  # (a_size, L, 20)
        row_count = a_size * L
        wide = freq_arr.reshape(row_count, 20)
        row_metadata = pandas.DataFrame(
            {
                "allele": numpy.repeat(batch_alleles_arr, L),
                "length": numpy.full(row_count, L, dtype=numpy.int64),
                "cutoff_fraction": numpy.full(row_count, cutoff_fraction),
                "cutoff_count": numpy.full(row_count, k, dtype=numpy.int64),
                "position": numpy.tile(numpy.arange(1, L + 1), a_size),
            }
        )
        df = pandas.concat(
            [row_metadata, pandas.DataFrame(wide, columns=aa_columns)],
            axis=1,
        )
        freq_matrices.append(df)

    length_dists = []
    for cutoff_fraction, k_total, length_fractions_t in length_fraction_tensors:
        length_fractions = length_fractions_t.cpu().numpy()
        a_idx, l_idx = numpy.where(length_fractions > 0)
        if a_idx.size == 0:
            continue
        ld = pandas.DataFrame({
            "allele": batch_alleles_arr[a_idx],
            "cutoff_fraction": cutoff_fraction,
            "cutoff_count": k_total,
            "length": l_idx.astype(numpy.int64),
            "fraction": length_fractions[a_idx, l_idx],
        })[[
            "allele", "cutoff_fraction", "cutoff_count",
            "length", "fraction",
        ]].sort_values(
            ["allele", "cutoff_fraction", "length"]
        ).reset_index(drop=True)
        length_dists.append(ld)

    return freq_matrices, length_dists
