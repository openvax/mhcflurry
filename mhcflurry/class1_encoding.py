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

"""Peptide encoding helpers for class I neural networks."""

import logging

from . import amino_acid
from .encodable_sequences import EncodableSequences

_DEVICE_PEPTIDE_ENCODING_TRUE_VALUES = {
    "1",
    "true",
    "yes",
    "on",
    "auto",
    "cpu",
    "device",
    "torch",
    "gpu",
    "mps",
}
_DEVICE_PEPTIDE_ENCODING_FALSE_VALUES = {
    "0",
    "false",
    "no",
    "off",
    "numpy",
    "legacy",
}

_warned_legacy_peptide_vector_encoding = []


def _warn_legacy_peptide_vector_encoding(value):
    """Warn once that the legacy dense-vector peptide path is gone."""
    if not _warned_legacy_peptide_vector_encoding:
        _warned_legacy_peptide_vector_encoding.append(True)
        logging.warning(
            "peptide_amino_acid_encoding_torch=%r is deprecated and ignored: "
            "peptides are always index-encoded ((N, L) int8) and embedded on "
            "device via a frozen table. The legacy dense-vector peptide "
            "encoding path has been removed.", value)


def _peptide_torch_encoding_name(hyperparameters):
    """Return the fixed amino-acid encoding for the peptide embedding table.

    Peptides are always index-encoded ((N, L) int8) and embedded via a frozen
    ``torch.nn.functional.embedding`` table (device-agnostic: CUDA/MPS/CPU). The
    table comes from ``amino_acid.get_vector_encoding_df`` and is registered as
    a non-persistent buffer, so it moves with ``.to(device)`` but never trains
    or bloats the NPZ weight list.

    The legacy ``peptide_amino_acid_encoding_torch=False`` dense-vector path has
    been removed; a falsy value is accepted but ignored (with a one-time
    deprecation warning) so older configs still load. A non-default value names
    the encoding directly; an unknown name still raises.
    """
    mode = hyperparameters.get("peptide_amino_acid_encoding_torch", True)
    if isinstance(mode, str):
        mode_normalized = mode.strip().lower()
        if mode_normalized in _DEVICE_PEPTIDE_ENCODING_FALSE_VALUES:
            _warn_legacy_peptide_vector_encoding(mode)
        else:
            try:
                amino_acid.get_vector_encoding_df(mode)
                return mode
            except KeyError:
                pass
            if mode_normalized not in _DEVICE_PEPTIDE_ENCODING_TRUE_VALUES:
                raise ValueError(
                    "Unsupported peptide_amino_acid_encoding_torch value %r. "
                    "Expected bool, a true/false string, or one of %s."
                    % (mode, sorted(amino_acid.ENCODING_DATA_FRAMES))
                )
    elif not mode:
        _warn_legacy_peptide_vector_encoding(mode)

    peptide_encoding = hyperparameters.get("peptide_encoding", {})
    encoding_name = peptide_encoding.get("vector_encoding_name", "BLOSUM62")
    try:
        amino_acid.get_vector_encoding_df(encoding_name)
    except KeyError:
        raise ValueError(
            "Peptide encoding requires a fixed vector encoding with a torch "
            "lookup table; got %r. Available: %s"
            % (encoding_name, sorted(amino_acid.ENCODING_DATA_FRAMES))
        ) from None
    return encoding_name


def _peptide_uses_torch_encoding(hyperparameters):
    """Deprecated: peptides are always index-encoded now (always True)."""
    return True


def _peptide_torch_encoding_table(encoding_name):
    """Return a float32 lookup table indexed by ``AMINO_ACID_INDEX``."""
    return amino_acid.vector_encoding_index_table(encoding_name)


def _peptide_torch_encoding_shape(index_shape, encoding_name):
    """Expand an ``(L,)`` categorical peptide shape to ``(L, V)``."""
    return (
        index_shape[0],
        amino_acid.vector_encoding_length(encoding_name),
    )


def _categorical_kwargs_for_peptide_encoding(peptide_encoding):
    """Extract kwargs shared by vector and categorical peptide encoders."""
    return {
        key: value
        for key, value in peptide_encoding.items()
        if key in (
            "alignment_method",
            "left_edge",
            "right_edge",
            "max_length",
            "trim",
            "allow_unsupported_amino_acids",
        )
    }


def peptide_sequences_to_network_input(
        peptides,
        peptide_encoding,
        peptide_amino_acid_encoding_torch=True):
    """Encode peptide strings to the representation consumed by a network.

    This is the central peptide-only string-to-array conversion. Peptides are
    always returned as compact ``(N, L)`` int8 amino-acid indices; the
    fixed-vector lookup then happens through the network's frozen torch
    embedding table. (``peptide_amino_acid_encoding_torch`` is accepted for
    config compatibility but the legacy dense-vector path is gone.)
    """
    if not peptide_amino_acid_encoding_torch:
        _warn_legacy_peptide_vector_encoding(peptide_amino_acid_encoding_torch)
    encoder = EncodableSequences.create(peptides)
    return (
        encoder.variable_length_to_fixed_length_categorical(
            **_categorical_kwargs_for_peptide_encoding(peptide_encoding)
        )
        .astype("int8", copy=False)
    )
