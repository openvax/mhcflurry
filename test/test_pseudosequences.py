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

"""Tests for pseudosequence filename registry and mhcseqs-backed extraction."""
import csv
import importlib.util

import pytest

from mhcflurry.class1_affinity_predictor import Class1AffinityPredictor
from mhcflurry.pseudosequences import (
    HLA_A0201_GROOVE1,
    HLA_A0201_GROOVE2,
    LEGACY_ALLELE_SEQUENCES_FILENAME,
    MHCFLURRY_39AA_MATURE_POSITIONS,
    NETMHCPAN_34AA_MATURE_POSITIONS,
    build_pseudosequences_from_mhcseqs,
    extend_allele_to_sequence_from_mhcseqs,
    extract_class1_pseudosequence,
    extract_class1_pseudosequence_from_grooves,
    extract_class1_pseudosequence_with_alignment,
    main,
    pseudosequence_filename_for_length,
    pseudosequence_filename_for_mapping,
    pseudosequence_path,
)

HAS_MHCSEQS = importlib.util.find_spec("mhcseqs") is not None

# Canonical HLA-A*02:01 reference pseudosequences (verified against MHCflurry
# 2.3.0 allele_sequences.csv and NetMHCpan 34aa table).
HLA_A0201_EXPECTED_39AA = "YFGERAMPYGEKVAHTHVDTLYGVRYHYYTWAVLAYTWY"
HLA_A0201_EXPECTED_34AA = "YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY"

# Full HLA-A*02:01 protein (24-aa signal peptide + alpha1 + alpha2 + alpha3 + TM/cyto).
HLA_A0201_FULL = (
    "MAVMAPRTLVLLLSGALALTQTWA"
    + HLA_A0201_GROOVE1
    + HLA_A0201_GROOVE2
    + "APKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWE"
    + "PSSQPTIPIVGIIAGLVLFGAVITGAVVAAVMWRRKSSDRKGGSYSQAASSDSAQGSDVSLTACKV"
)


def test_pseudosequence_filename_for_length():
    assert (
        pseudosequence_filename_for_length(34)
        == "pseudosequences.netmhcpan.34aa.csv"
    )
    assert (
        pseudosequence_filename_for_length(37)
        == "pseudosequences.mhcflurry.37aa.csv"
    )
    assert (
        pseudosequence_filename_for_length(39)
        == "pseudosequences.mhcflurry.39aa.csv"
    )
    assert pseudosequence_filename_for_length(None) is None


def test_pseudosequence_filename_for_ambiguous_mapping():
    assert pseudosequence_filename_for_mapping({}) is None
    assert pseudosequence_filename_for_mapping({"HLA-A*02:01": None}) is None
    assert pseudosequence_filename_for_mapping({
        "HLA-A*02:01": "A" * 34,
        "HLA-A*03:01": "A" * 39,
    }) is None


def test_pseudosequence_path_prefers_canonical(tmp_path):
    canonical = tmp_path / "pseudosequences.mhcflurry.39aa.csv"
    canonical.write_text("allele,pseudosequence\n")

    assert pseudosequence_path(str(tmp_path), 39) == str(canonical)


def test_pseudosequence_path_falls_back_to_legacy(tmp_path):
    assert pseudosequence_path(str(tmp_path), 39) == str(
        tmp_path / LEGACY_ALLELE_SEQUENCES_FILENAME)


def test_pseudosequence_cli_filename(capsys):
    main(["filename", "--length", "39"])

    assert capsys.readouterr().out == "pseudosequences.mhcflurry.39aa.csv\n"


def test_pinned_positions_geometry():
    assert len(MHCFLURRY_39AA_MATURE_POSITIONS) == 39
    assert len(NETMHCPAN_34AA_MATURE_POSITIONS) == 34
    assert set(NETMHCPAN_34AA_MATURE_POSITIONS) < set(MHCFLURRY_39AA_MATURE_POSITIONS)
    assert tuple(sorted(MHCFLURRY_39AA_MATURE_POSITIONS)) == MHCFLURRY_39AA_MATURE_POSITIONS
    assert tuple(sorted(NETMHCPAN_34AA_MATURE_POSITIONS)) == NETMHCPAN_34AA_MATURE_POSITIONS


def test_extract_from_grooves_canonical_hla_a0201():
    assert (
        extract_class1_pseudosequence_from_grooves(
            HLA_A0201_GROOVE1, HLA_A0201_GROOVE2, length=39
        )
        == HLA_A0201_EXPECTED_39AA
    )
    assert (
        extract_class1_pseudosequence_from_grooves(
            HLA_A0201_GROOVE1, HLA_A0201_GROOVE2, length=34
        )
        == HLA_A0201_EXPECTED_34AA
    )


def test_extract_from_grooves_partial_exon2_89_92():
    # IMGT partial exon-2/exon-3 sequences commonly omit mature pos 1 (G)
    # and the final residue of alpha-2, giving (89, 92) groove lengths.
    partial_g1 = HLA_A0201_GROOVE1[1:]
    partial_g2 = HLA_A0201_GROOVE2[:-1]
    assert len(partial_g1) == 89
    assert len(partial_g2) == 92
    assert (
        extract_class1_pseudosequence_from_grooves(partial_g1, partial_g2, length=39)
        == HLA_A0201_EXPECTED_39AA
    )
    assert (
        extract_class1_pseudosequence_from_grooves(partial_g1, partial_g2, length=34)
        == HLA_A0201_EXPECTED_34AA
    )


def test_extract_from_grooves_handles_indels_across_species():
    # BoLA-2*057:01: (90, 92) cattle allele with 1-aa deletion in alpha2
    bola_g1 = "GSHSLRYFYTAVSRPGLGEPSFIIVGYVDDTQFVRFDSNTPNPRMEPRARWVEQEGPEYWDRETRISKETAQTFRVDLNTLRGYYNQSEA"
    bola_g2 = "GSHNIQEMYGCDVGPDGLLRGYDQFAYDGRDYLALNEDLRSWTAADTAAQITKRKWEAEGYAESLRNYLEGRCVEGLRRYLENGKDTLLRAD"
    assert (
        extract_class1_pseudosequence_from_grooves(bola_g1, bola_g2, length=39)
        == "YYGESIMPYREISETTFVDTLYGIEYDFYTWEYLNYRGY"
    )

    # Mafa-B*016:01: (90, 94) macaque allele with 1-aa insertion in alpha2 loop
    mafa_g1 = "GSHSLRYFTTALSRPGRGEPRFIAVGYVDDTQFVRFDSDAESPRMEPRAPWMGQEGPEYWEEETRRAKAHAQTDLGGLGILRGYYNQSEA"
    mafa_g2 = "GSHTYQRMSGCDVGPDDGRLLRGYYQRAYDGKDYIALNEDLRSWTAADVAAQKTQRKWEAALAAEQLRAYLEGECVEWLRRYLENGKETLQRAD"
    assert (
        extract_class1_pseudosequence_from_grooves(mafa_g1, mafa_g2, length=39)
        == "YTGERAMPYEERAAHTDGGILYGYRSYRYTWAALAYEWY"
    )

    # Onmy-UBA*01:01: (89, 94) rainbow trout divergent teleost with alpha1 del + alpha2 ins
    onmy_g1 = "ATHSLKYFYTASSEVPNFPEFVVVAMVDGAQMVHYDSNSQRAVPKQDWMNRAAETLPQYWESQTGIFKGAQQTFKANIDIVKQRFNQSG"
    onmy_g2 = "GVHIVQKMYGCEWDDEAGVTEGFYQYGYDGEDFIALDLKTTKWIAPTPQAVITKLKWDSNTANTEYWNNYLTQTCIEWLKKYVDYGKSTLMRTV"
    onmy_39 = extract_class1_pseudosequence_from_grooves(onmy_g1, onmy_g2, length=39)
    assert len(onmy_39) == 39
    # All 17 alpha2 positions match MHCflurry 2.3.0's clustalo alignment verbatim.
    assert onmy_39[22:] == "GVKYYYYTWNAWNYTWY"

    # Truncated fragment (missing first 15 residues of alpha1) pads missing slots with X
    frag_g1 = HLA_A0201_GROOVE1[15:]
    frag_39 = extract_class1_pseudosequence_from_grooves(frag_g1, HLA_A0201_GROOVE2, length=39)
    assert len(frag_39) == 39
    assert frag_39[:2] == "XX"
    assert frag_39[2:] == HLA_A0201_EXPECTED_39AA[2:]


def test_extract_from_grooves_invalid_length_and_sanitization():
    with pytest.raises(ValueError, match="Unsupported pseudosequence length"):
        extract_class1_pseudosequence_from_grooves(
            HLA_A0201_GROOVE1, HLA_A0201_GROOVE2, length=37
        )
    # Non-standard amino acid characters normalize to X.
    mutated_g1 = HLA_A0201_GROOVE1[:6] + "*" + HLA_A0201_GROOVE1[7:]
    pseudo = extract_class1_pseudosequence_from_grooves(mutated_g1, HLA_A0201_GROOVE2)
    assert pseudo[0] == "X"
    assert pseudo[1:] == HLA_A0201_EXPECTED_39AA[1:]


def test_build_and_extend_strict_generalization():
    # Mock mhcseqs rows: canonical HLA-A*02:01, a novel species allele BoLA-2*057:01,
    # a Class II row (ignored), a non-groove row (ignored), and a UniProt accession.
    bola_g1 = "GSHSLRYFYTAVSRPGLGEPSFIIVGYVDDTQFVRFDSNTPNPRMEPRARWVEQEGPEYWDRETRISKETAQTFRVDLNTLRGYYNQSEA"
    bola_g2 = "GSHNIQEMYGCDVGPDGLLRGYDQFAYDGRDYLALNEDLRSWTAADTAAQITKRKWEAEGYAESLRNYLEGRCVEGLRRYLENGKDTLLRAD"
    rows = [
        {
            "two_field_allele": "HLA-A*02:01",
            "mhc_class": "I",
            "groove1": HLA_A0201_GROOVE1,
            "groove2": HLA_A0201_GROOVE2,
            "groove_status": "ok",
        },
        {
            "two_field_allele": "BoLA-2*057:01",
            "mhc_class": "I",
            "groove1": bola_g1,
            "groove2": bola_g2,
            "groove_status": "ok",
        },
        {
            "two_field_allele": "HLA-DRB1*01:01",
            "mhc_class": "II",
            "groove1": HLA_A0201_GROOVE1,
            "groove2": HLA_A0201_GROOVE2,
            "groove_status": "ok",
        },
        {
            "two_field_allele": "HLA-A*99:99",
            "mhc_class": "I",
            "groove1": HLA_A0201_GROOVE1,
            "groove2": HLA_A0201_GROOVE2,
            "groove_status": "missing_groove",
        },
        {
            "two_field_allele": "A0A1234567",
            "mhc_class": "I",
            "groove1": HLA_A0201_GROOVE1,
            "groove2": HLA_A0201_GROOVE2,
            "groove_status": "ok",
        },
    ]

    built = build_pseudosequences_from_mhcseqs(rows=rows, length=39)
    assert set(built) == {"HLA-A*02:01", "BoLA-2*057:01"}
    assert built["HLA-A*02:01"] == HLA_A0201_EXPECTED_39AA
    assert built["BoLA-2*057:01"] == "YYGESIMPYREISETTFVDTLYGIEYDFYTWEYLNYRGY"

    # Without allele normalization, UniProt accessions are also retained.
    built_raw = build_pseudosequences_from_mhcseqs(
        rows=rows, length=39, normalize_alleles=False
    )
    assert "A0A1234567" in built_raw

    # Strict generalization: preserve existing non-X entry verbatim when
    # prefer_existing=True, fill X-containing entries when mhcseqs is more
    # complete, retain existing alleles absent from mhcseqs, and add new alleles.
    custom_non_x = "A" * 39
    existing = {
        "HLA-A*02:01": "X" + HLA_A0201_EXPECTED_39AA[1:],  # has 1 X -> upgraded
        "HLA-B*07:02": custom_non_x,  # absent from mhcseqs -> preserved
    }
    extended = extend_allele_to_sequence_from_mhcseqs(
        existing, rows=rows, prefer_existing=True
    )
    assert set(extended) == {"HLA-A*02:01", "HLA-B*07:02", "BoLA-2*057:01"}
    assert extended["HLA-A*02:01"] == HLA_A0201_EXPECTED_39AA
    assert extended["HLA-B*07:02"] == custom_non_x
    assert extended["BoLA-2*057:01"] == "YYGESIMPYREISETTFVDTLYGIEYDFYTWEYLNYRGY"

    # Non-X existing entry is preserved verbatim under prefer_existing=True
    # even when mhcseqs produces a different sequence.
    existing_pinned = {"HLA-A*02:01": custom_non_x}
    extended_pinned = extend_allele_to_sequence_from_mhcseqs(
        existing_pinned, rows=rows, prefer_existing=True
    )
    assert extended_pinned["HLA-A*02:01"] == custom_non_x
    assert "BoLA-2*057:01" in extended_pinned


def test_predictor_extend_alleles_from_mhcseqs():
    bola_g1 = "GSHSLRYFYTAVSRPGLGEPSFIIVGYVDDTQFVRFDSNTPNPRMEPRARWVEQEGPEYWDRETRISKETAQTFRVDLNTLRGYYNQSEA"
    bola_g2 = "GSHNIQEMYGCDVGPDGLLRGYDQFAYDGRDYLALNEDLRSWTAADTAAQITKRKWEAEGYAESLRNYLEGRCVEGLRRYLENGKDTLLRAD"
    predictor = Class1AffinityPredictor(
        allele_to_sequence={"HLA-A*02:01": HLA_A0201_EXPECTED_39AA}
    )
    added = predictor.extend_alleles_from_mhcseqs(
        rows=[
            {
                "two_field_allele": "BoLA-2*057:01",
                "mhc_class": "I",
                "groove1": bola_g1,
                "groove2": bola_g2,
                "groove_status": "ok",
            }
        ]
    )
    assert added == 1
    assert "BoLA-2*057:01" in predictor.supported_alleles
    assert predictor.allele_to_sequence["HLA-A*02:01"] == HLA_A0201_EXPECTED_39AA


def test_pseudosequence_cli_build(tmp_path):
    mhcseqs_csv = tmp_path / "mhc-full-seqs.csv"
    with open(mhcseqs_csv, "w", newline="", encoding="utf-8") as fd:
        writer = csv.DictWriter(
            fd,
            fieldnames=["two_field_allele", "mhc_class", "groove1", "groove2", "groove_status"],
        )
        writer.writeheader()
        writer.writerow({
            "two_field_allele": "HLA-A*02:01",
            "mhc_class": "I",
            "groove1": HLA_A0201_GROOVE1,
            "groove2": HLA_A0201_GROOVE2,
            "groove_status": "ok",
        })

    out_csv = tmp_path / "out.csv"
    if HAS_MHCSEQS:
        main(["build", "--mhcseqs-csv", str(mhcseqs_csv), "--out", str(out_csv)])
        rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
        assert rows == [{"allele": "HLA-A*02:01", "pseudosequence": HLA_A0201_EXPECTED_39AA}]


@pytest.mark.skipif(not HAS_MHCSEQS, reason="mhcseqs is not installed")
def test_extract_class1_pseudosequence_with_mhcseqs(capsys, caplog):
    import logging

    # Full-length protein with 24-aa signal peptide
    assert (
        extract_class1_pseudosequence(HLA_A0201_FULL, allele="HLA-A*02:01", length=39)
        == HLA_A0201_EXPECTED_39AA
    )
    assert (
        extract_class1_pseudosequence(HLA_A0201_FULL, allele="HLA-A*02:01", length=34)
        == HLA_A0201_EXPECTED_34AA
    )

    # Mature-numbered point mutation (K66A -> position 66 is index 11 in 39aa),
    # accepting either a tuple or a single mutation string.
    with caplog.at_level(logging.WARNING):
        mutated = extract_class1_pseudosequence(
            HLA_A0201_FULL, allele="HLA-A*02:01", mutations=("K66A",), length=39
        )
    assert not caplog.records
    mutated_from_str = extract_class1_pseudosequence(
        HLA_A0201_FULL, allele="HLA-A*02:01", mutations="K66A", length=39
    )
    assert mutated == mutated_from_str
    assert HLA_A0201_EXPECTED_39AA[11] == "K"
    assert mutated[11] == "A"
    assert mutated[:11] == HLA_A0201_EXPECTED_39AA[:11]
    assert mutated[12:] == HLA_A0201_EXPECTED_39AA[12:]

    # A mutation outside the 39 pseudosequence positions (mature S2A) emits a
    # clear warning that predicted binding will match wild-type.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        noop_mut = extract_class1_pseudosequence(
            HLA_A0201_FULL, allele="HLA-A*02:01", mutations=("S2A",), length=39
        )
    assert noop_mut == HLA_A0201_EXPECTED_39AA
    assert any("did not change any" in rec.message for rec in caplog.records)

    # Bat Ptal-N*01:01 (3-aa MDL insertion at mature pos 51): verify structured
    # alignment and insertion warning when mhcseqs >= 2.9.0 is installed.
    ptal_seq = (
        "GFHSLRYFYTAWSRPGSGEPRFVAVGYVDDTQFVRFDSDNASPRAEPRAPWMDLVEQQDPQYWDRNTRNARDAAQTYRVGLDNVRGYYNQSEA"
        "GSHTIQRMYGCDVGPHGRLLRGYDQLAYDGADYIALNEDLRSWTAADLAAQNTRRKWEEAGYAERDRAYLEGECVEWLLKHLENGRETLLRAD"
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ptal_39, ptal_aln = extract_class1_pseudosequence_with_alignment(
            ptal_seq, allele="Ptal-N*01:01", length=39
        )
    assert ptal_39 == "YYGERAAPYRNNADATYVGNVYGIRYDLYTWAYDAYEWH"
    if ptal_aln is not None:
        assert ptal_aln.has_insertions
        assert any("MDL" in rec.message for rec in caplog.records)

    # Pure alpha3 Ig+TM fragment must be rejected
    ig_only = HLA_A0201_FULL[24 + 90 + 93 :]
    with pytest.raises(ValueError, match="binding groove|Incomplete"):
        extract_class1_pseudosequence(ig_only, allele="HLA-A*02:01")

    # Predictor registration helper
    predictor = Class1AffinityPredictor(
        allele_to_sequence={"HLA-A*01:01": "Y" * 39}
    )
    added_seq = predictor.add_allele_from_sequence("HLA-A0201", HLA_A0201_FULL)
    assert added_seq == HLA_A0201_EXPECTED_39AA
    assert predictor.allele_to_sequence["HLA-A*02:01"] == HLA_A0201_EXPECTED_39AA

    # CLI extract subcommand
    main(["extract", "--sequence", HLA_A0201_FULL, "--allele", "HLA-A*02:01"])
    assert capsys.readouterr().out.strip() == HLA_A0201_EXPECTED_39AA


def test_train_pan_allele_predictor_with_mhcseqs_pseudosequences_and_extend(tmp_path):
    """End-to-end check: train a tiny pan-allele predictor on pseudosequences
    built via build_pseudosequences_from_mhcseqs, extend it with a novel allele
    and a retired-alias allele, predict, and round-trip save/load."""
    bola_g1 = "GSHSLRYFYTAVSRPGLGEPSFIIVGYVDDTQFVRFDSNTPNPRMEPRARWVEQEGPEYWDRETRISKETAQTFRVDLNTLRGYYNQSEA"
    bola_g2 = "GSHNIQEMYGCDVGPDGLLRGYDQFAYDGRDYLALNEDLRSWTAADTAAQITKRKWEAEGYAESLRNYLEGRCVEGLRRYLENGKDTLLRAD"
    initial_mapping = build_pseudosequences_from_mhcseqs(
        rows=[
            {
                "two_field_allele": "HLA-A*02:01",
                "mhc_class": "I",
                "groove1": HLA_A0201_GROOVE1,
                "groove2": HLA_A0201_GROOVE2,
                "groove_status": "ok",
            },
        ],
        length=39,
    )
    predictor = Class1AffinityPredictor(allele_to_sequence=initial_mapping)
    predictor.fit_class1_pan_allele_models(
        n_models=1,
        architecture_hyperparameters={
            "max_epochs": 1,
            "layer_sizes": [8],
            "allele_dense_layer_sizes": [],
            "peptide_allele_merge_method": "concatenate",
            "validation_split": 0.0,
            "early_stopping": False,
            "random_negative_rate": 0.0,
            "random_negative_constant": 0,
            "data_dependent_initialization_method": None,
            "minibatch_size": 4,
            "locally_connected_layers": [],
        },
        alleles=["HLA-A*02:01"] * 4,
        peptides=["SIINFEKL", "NLVPMVATV", "GILGFVFTL", "LLFGYPVYV"],
        affinities=[50.0, 20.0, 15.0, 100.0],
        inequalities=["="] * 4,
        verbose=0,
        seed=42,
    )
    baseline_pred = predictor.predict(["NLVPMVATV"], allele="HLA-A*02:01")[0]

    # Extend the trained predictor with a new non-human allele and an allele
    # whose mhcgnomes alias differs from its no-alias canonical form.
    added = predictor.extend_alleles_from_mhcseqs(
        rows=[
            {
                "two_field_allele": "BoLA-2*057:01",
                "mhc_class": "I",
                "groove1": bola_g1,
                "groove2": bola_g2,
                "groove_status": "ok",
            },
            {
                "two_field_allele": "HLA-C*01:01",
                "mhc_class": "I",
                "groove1": HLA_A0201_GROOVE1,
                "groove2": HLA_A0201_GROOVE2,
                "groove_status": "ok",
            },
        ],
        prefer_existing=True,
    )
    assert added == 2
    assert "BoLA-2*057:01" in predictor.supported_alleles
    # Existing allele prediction is bit-for-bit unchanged.
    after_pred = predictor.predict(["NLVPMVATV"], allele="HLA-A*02:01")[0]
    assert after_pred == pytest.approx(baseline_pred, rel=1e-6)
    # Newly added allele predicts finite nM affinity cleanly.
    bola_pred = predictor.predict(["NLVPMVATV"], allele="BoLA-2*057:01")[0]
    assert 0.0 < bola_pred <= 50000.0
    # Retired alias maps cleanly back to the newly registered key.
    assert predictor.canonicalize_allele_name("HLA-C*01:02") == "HLA-C*01:01"

    # Save and reload round-trip preserves all alleles and predictions.
    models_dir = str(tmp_path / "trained_pan")
    predictor.save(models_dir)
    reloaded = Class1AffinityPredictor.load(models_dir, optimization_level=0)
    assert "BoLA-2*057:01" in reloaded.supported_alleles
    assert reloaded.predict(["NLVPMVATV"], allele="BoLA-2*057:01")[0] == pytest.approx(
        bola_pred, rel=1e-6
    )


def test_imgt_delegation_and_diagnostic_warnings_mocked(monkeypatch, caplog):
    """Verify mhcseqs >= 2.9.0 IMGT delegation and insertion/no-op warnings
    even in environments where the optional mhcseqs package is not installed."""
    import logging
    import types

    from mhcflurry import mhcseqs_pseudosequences as mp

    calls = []

    class FakeDomainAlignment:
        def __init__(self, seq, domain):
            offset = 0 if domain == "G-ALPHA1" else 90
            self.mature_to_aa = {offset + i + 1: aa for i, aa in enumerate(seq)}

    class FakeGrooveAlignment:
        def __init__(self, g1, g2, has_insertions=False):
            self._full = g1 + g2
            self.has_insertions = has_insertions

        def extract_positions(self, positions, missing="X", deleted="X"):
            return "".join(
                self._full[p - 1] if 1 <= p <= len(self._full) else missing
                for p in positions
            )

        def summary(self):
            return "alpha1: +3aa ins@51(MDL)" if self.has_insertions else "canonical"

    def fake_align_domain_to_imgt(seq, domain="G-ALPHA1"):
        calls.append(domain)
        return FakeDomainAlignment(seq, domain)

    def fake_decompose_domains(seq, mhc_class="I", allele="", gene="", mutations=()):
        g1 = HLA_A0201_GROOVE1
        g2 = HLA_A0201_GROOVE2
        return types.SimpleNamespace(
            status="ok",
            flags="",
            groove1=g1,
            groove2=g2,
            has_insertions=(allele == "Ptal-N*01:01"),
        )

    def fake_align_groove_to_imgt(record):
        return FakeGrooveAlignment(
            record.groove1,
            record.groove2,
            has_insertions=getattr(record, "has_insertions", False),
        )

    fake_mhcseqs = types.SimpleNamespace(
        align_domain_to_imgt=fake_align_domain_to_imgt,
        align_groove_to_imgt=fake_align_groove_to_imgt,
        decompose_domains=fake_decompose_domains,
    )
    monkeypatch.setattr(mp, "_get_mhcseqs_imgt_aligner", lambda: fake_align_domain_to_imgt)
    monkeypatch.setattr(mp, "_import_mhcseqs", lambda: fake_mhcseqs)

    # Groove extraction delegates to align_domain_to_imgt for both domains.
    pseudo = extract_class1_pseudosequence_from_grooves(
        HLA_A0201_GROOVE1, HLA_A0201_GROOVE2, length=39
    )
    assert pseudo == HLA_A0201_EXPECTED_39AA
    assert calls == ["G-ALPHA1", "G-ALPHA2"]

    # Insertion warning and no-op mutation warning both fire cleanly.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        extracted, aln = extract_class1_pseudosequence_with_alignment(
            HLA_A0201_FULL,
            allele="Ptal-N*01:01",
            mutations="S2A",
            length=39,
        )
    assert extracted == HLA_A0201_EXPECTED_39AA
    assert aln is not None and aln.has_insertions
    messages = [rec.message for rec in caplog.records]
    assert any("MDL" in m for m in messages)
    assert any("did not change any" in m for m in messages)

    # Disabling warning flags suppresses both warnings.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        extract_class1_pseudosequence_with_alignment(
            HLA_A0201_FULL,
            allele="Ptal-N*01:01",
            mutations="S2A",
            length=39,
            warn_on_indels=False,
            warn_on_noop_mutations=False,
        )
    assert not caplog.records


def test_short_fragment_status_guard(monkeypatch):
    """mhcseqs >= 2.12.2 emits status='short' for short both-halves exon
    fragments; reject them unless allow_fragments / include_fragments is True."""
    import types
    from mhcflurry import mhcseqs_pseudosequences as mp

    short_g1 = HLA_A0201_GROOVE1[15:75]  # 60 aa (<70 aa)
    short_g2 = HLA_A0201_GROOVE2[15:75]  # 60 aa (<70 aa)

    rows = [
        {
            "two_field_allele": "HLA-A*02:01",
            "mhc_class": "I",
            "groove1": HLA_A0201_GROOVE1,
            "groove2": HLA_A0201_GROOVE2,
            "groove_status": "ok",
        },
        {
            "two_field_allele": "HLA-B*07:02",
            "mhc_class": "I",
            "groove1": short_g1,
            "groove2": short_g2,
            "groove_status": "short",
        },
        {
            "two_field_allele": "HLA-C*07:01",
            "mhc_class": "I",
            "groove1": short_g1,
            "groove2": short_g2,
            "groove_status": "too_short",
        },
    ]

    strict = build_pseudosequences_from_mhcseqs(rows=rows, include_fragments=False)
    assert set(strict) == {"HLA-A*02:01"}

    with_frags = build_pseudosequences_from_mhcseqs(rows=rows, include_fragments=True)
    assert set(with_frags) == {"HLA-A*02:01", "HLA-B*07:02"}
    assert "HLA-C*07:01" not in with_frags  # too_short is always non-groove

    fake_mhcseqs = types.SimpleNamespace(
        decompose_domains=lambda seq, **kwargs: types.SimpleNamespace(
            status="short", flags=("short_groove",), groove1=short_g1, groove2=short_g2
        ),
        align_groove_to_imgt=None,
    )
    monkeypatch.setattr(mp, "_import_mhcseqs", lambda: fake_mhcseqs)
    with pytest.raises(ValueError, match="Incomplete Class I groove"):
        extract_class1_pseudosequence(short_g1 + short_g2, allele="HLA-B*07:02")
    pseudo = extract_class1_pseudosequence(
        short_g1 + short_g2, allele="HLA-B*07:02", allow_fragments=True
    )
    assert len(pseudo) == 39
