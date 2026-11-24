"""Reproducible cross-species zero-shot validation benchmark & publication figure
generator for MHCflurry 2.5.0 (mhcseqs 2.9.0 integration).

Usage:
    python run_cross_species_figures.py --collect --plot --out-dir ./figures_250
    python run_cross_species_figures.py --plot --out-dir ./figures_250
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import average_precision_score, roc_auc_score

import mhcgnomes
import mhcseqs
from mhcflurry import Class1AffinityPredictor
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.cli.figure_style import (
    POSITIVE_DELTA_COLOR,
    SIDE_A_COLOR,
    SIDE_B_COLOR,
    apply_paper_style,
    despine,
)
from mhcflurry.downloads import get_path
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.ranking_metrics import ppv_at_n

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_BG_PROBS = np.array([
    0.070, 0.023, 0.047, 0.071, 0.037, 0.066, 0.026, 0.043, 0.057, 0.100,
    0.021, 0.036, 0.063, 0.048, 0.056, 0.083, 0.054, 0.060, 0.012, 0.027,
])
AA_BG_PROBS /= AA_BG_PROBS.sum()

POS39 = (
    7, 9, 16, 19, 21, 24, 45, 47, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77,
    80, 81, 84,
    91, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163,
    167, 171,
)

# Colorblind-safe clade palette (harmonized with figure_style.py)
CLADE_COLORS = {
    "Marsupialia": "#0072B2",     # deep blue (~160 Myr)
    "Carnivora": "#009E73",       # bluish green (cat)
    "Aves": "#E69F00",            # amber/orange (bird, ~310 Myr)
    "Artiodactyla": "#56B4E9",    # sky blue (pig/cattle)
    "Perissodactyla": "#CC79A7",  # reddish purple (horse)
    "Chiroptera": "#D55E00",      # vermillion (bat failure mode)
    "Amphibia": "#8C564B",        # brown (frog, ~350 Myr)
    "Held-out ref": "#666666",    # neutral gray
}


def extract_39_with_mhcseqs(seq: str, *, allele: str = "", mutations: tuple[str, ...] = ()):
    rec = mhcseqs.decompose_domains(
        seq, mhc_class="I", allele=allele, mutations=mutations
    )
    aln = mhcseqs.align_groove_to_imgt(rec)
    p39 = aln.extract_positions(POS39, missing="X", deleted="X")
    p39 = "".join(c if c in VALID_AA else "X" for c in p39.upper())
    return p39, aln


def fetch_iedb_cached(cache_dir: Path, restriction: str, limit: int = 5000) -> list[dict]:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in restriction)
    cache_file = cache_dir / f"iedb_{safe}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    for attempt in range(3):
        try:
            resp = requests.get(
                "https://query-api.iedb.org/mhc_search",
                params={
                    "mhc_class": "eq.I",
                    "mhc_restriction": f"eq.{restriction}",
                    "select": "linear_sequence,mhc_restriction,qualitative_measure,quantitative_measure",
                    "limit": str(limit),
                },
                headers={"Accept": "application/json"},
                timeout=45,
            )
            resp.raise_for_status()
            rows = resp.json()
            cache_file.write_text(json.dumps(rows), encoding="utf-8")
            time.sleep(0.35)
            return rows
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
    return []


def fetch_pdb_heavy_chain(cache_dir: Path, pdb_id: str) -> str:
    cache_file = cache_dir / f"pdb_{pdb_id}.fasta"
    if cache_file.exists():
        txt = cache_file.read_text(encoding="utf-8")
    else:
        resp = requests.get(f"https://www.rcsb.org/fasta/entry/{pdb_id}", timeout=30)
        resp.raise_for_status()
        txt = resp.text
        cache_file.write_text(txt, encoding="utf-8")
    recs, hdr, buf = [], None, []
    for line in txt.splitlines():
        if line.startswith(">"):
            if hdr:
                recs.append((hdr, "".join(buf)))
            hdr, buf = line[1:], []
        else:
            buf.append(line.strip())
    if hdr:
        recs.append((hdr, "".join(buf)))
    for _h, s in recs:
        if 170 <= len(s) <= 420:
            return s
    return recs[0][1]


def fetch_ncbi_protein(cache_dir: Path, name: str, term: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    cache_file = cache_dir / f"ncbi_{safe}.fasta"
    if cache_file.exists():
        txt = cache_file.read_text(encoding="utf-8")
    else:
        r1 = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "protein", "term": term, "retmode": "json", "retmax": 5},
            timeout=30,
        )
        r1.raise_for_status()
        ids = r1.json()["esearchresult"]["idlist"]
        time.sleep(0.5)
        r2 = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "protein", "id": ",".join(ids), "rettype": "fasta", "retmode": "text"},
            timeout=30,
        )
        r2.raise_for_status()
        txt = r2.text
        cache_file.write_text(txt, encoding="utf-8")
        time.sleep(0.5)
    recs, hdr, buf = [], None, []
    for line in txt.splitlines():
        if line.startswith(">"):
            if hdr:
                recs.append((hdr, "".join(buf)))
            hdr, buf = line[1:], []
        else:
            buf.append(line.strip())
    if hdr:
        recs.append((hdr, "".join(buf)))
    good = [r for r in recs if 170 <= len(r[1]) <= 420]
    return max(good or recs, key=lambda r: len(r[1]))[1]


def make_length_matched_decoys(positives: list[str], ratio: int = 10, seed: int = 42) -> list[str]:
    rng = np.random.default_rng(seed)
    pos_set = set(positives)
    decoys = []
    for p in positives:
        L = len(p)
        for _ in range(ratio):
            while True:
                cand = "".join(rng.choice(AA_LIST, size=L, p=AA_BG_PROBS))
                if cand not in pos_set:
                    decoys.append(cand)
                    break
    return decoys


def score_peptides(predictor: Class1AffinityPredictor, peptides: list[str], pseudo39: str) -> np.ndarray:
    enc_p = EncodableSequences.create(peptides)
    enc_a = AlleleEncoding(alleles=["T"] * len(peptides), allele_to_sequence={"T": pseudo39})
    preds = np.array([
        m.predict(enc_p, allele_encoding=enc_a)
        for m in predictor.class1_pan_allele_models
    ])
    return np.exp(np.mean(np.log(np.clip(preds, 1e-6, None)), axis=0))


def stratified_bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    metric_fn,
    *,
    n_boot: int = 1000,
    seed: int = 2026,
) -> tuple[float, float, float]:
    point = float(metric_fn(y, scores))
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        vals[b] = float(metric_fn(y[idx], scores[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def compute_9mer_ic_matrix(peptides: list[str]) -> list[dict[str, float]]:
    """Return 9-position information-content (bits) matrix for logomaker."""
    nine = [p for p in peptides if len(p) == 9 and set(p) <= VALID_AA]
    if not nine:
        return []
    n = float(len(nine))
    pseudocount = 0.05
    rows = []
    for pos in range(9):
        counts = collections.Counter(p[pos] for p in nine)
        probs = {aa: (counts.get(aa, 0) + pseudocount) / (n + 20 * pseudocount) for aa in AA_LIST}
        entropy = -sum(p * np.log2(p) for p in probs.values())
        ic = max(0.0, np.log2(20.0) - entropy)
        rows.append({aa: float(probs[aa] * ic) for aa in AA_LIST})
    return rows


def classify_clade(species_name: str) -> str:
    s = (species_name or "").lower()
    if any(k in s for k in ["homo", "macaca", "pan ", "gorilla", "pongo", "papio", "callithrix", "chlorocebus", "cercocebus", "aotus", "saguinus", "saimiri", "ateles", "cebus", "hylobates", "nomascus", "symphalangus", "trachypithecus", "mandrillus", "colobus", "piliocolobus", "theropithecus", "microcebus", "lemur", "otolemur", "tarsius", "carlito"]):
        return "Primates"
    if any(k in s for k in ["bos ", "ovis ", "capra", "sus ", "equus", "cervus", "camelus", "vicugna", "bubalus", "bison", "giraffa", "balaenoptera", "tursiops", "orcinus", "delphinapterus", "physeter", "phocoena", "odocoileus", "rangifer", "alces"]):
        return "Ungulates & Cetacea"
    if any(k in s for k in ["canis", "felis", "vulpes", "ursus", "mustela", "ailuropoda", "panthera", "acinonyx", "lynx", "zalophus", "phoca", "odobenus", "enhvdra", "enhydra", "neogale", "martes", "lutra", "crocuta", "suricata", "meles", "procyon"]):
        return "Carnivora"
    if any(k in s for k in ["mus ", "rattus", "cavia", "mesocricetus", "cricetulus", "oryctolagus", "peromyscus", "microtus", "meriones", "marmota", "ictidomys", "chinchilla", "heterocephalus", "octodon", "dipodomys", "jaculus", "castor", "sciurus", "tamias", "ochotona"]):
        return "Rodents & Lagomorphs"
    if any(k in s for k in ["pteropus", "myotis", "rhinolophus", "desmodus", "eptesicus", "pipistrellus", "phyllostomus", "artibeus", "rousettus", "hipposideros", "molossus", "tadarida", "miniopterus", "megaderma"]):
        return "Bats (Chiroptera)"
    if any(k in s for k in ["trichosurus", "sarcophilus", "monodelphis", "phascolarctos", "vombatus", "macropus", "notamacropus", "didelphis", "dasyurus", "antichinus", "antechinus", "ornithorhynchus", "tachyglossus"]):
        return "Marsupials & Monotremes"
    if any(k in s for k in ["gallus", "anas ", "meleagris", "coturnix", "taeniopygia", "passer", "parus", "cyanistes", "corvus", "falco", "aquila", "haliaeetus", "columba", "struthio", "dromaius", "apteryx", "anser", "cygnus", "cairina", "numida", "phasianus", "tyto", "strix", "melopsittacus", "amazona", "serinus", "ficedula", "hirundo", "sturnus", "turdus", "sylvia", "acrocephalus", "calidris", "larus", "spheniscus", "aptenodytes", "pygoscelis", "pelecanus", "ardea", "ciconia", "grus", "otis", "cuculus", "apus", "picoides", "dryobates", "merops"]):
        return "Birds (Aves)"
    if any(k in s for k in ["xenopus", "salmo", "oncorhynchus", "danio", "oryzias", "takifugu", "gadus", "cyprinus", "ictalurus", "oreochromis", "gasterosteus", "dicentrarchus", "sparus", "scophthalmus", "paralichthys", "carassius", "ctenopharyngodon", "esox", "clarias", "silurus", "anguilla", "acipenser", "lepisosteus", "latimeria", "callorhinchus", "scyliorhinus", "rhincodon", "carcharodon", "ambystoma", "rana ", "bufo", "nanorana", "anolis", "alligator", "crocodylus", "chrysemys", "chelonia", "trachemys", "pelodiscus", "pogona", "gekko", "python", "thamnophis", "notechis", "pseudonaja", "podarcis", "lacerta", "zootoca", "sphenodon"]):
        return "Reptiles, Amphibians & Fish"
    return "Other Vertebrates"


def collect_results(v290_csv: Path, out_dir: Path) -> dict:
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MHCflurry 2.3.0 predictor and curated training data...")
    predictor = Class1AffinityPredictor.load()
    orig_supported = set(predictor.allele_to_sequence)
    train_df = pd.read_csv(get_path("data_curated", "curated_training_data.csv.bz2"))
    train_pairs = set(zip(train_df["allele"].astype(str), train_df["peptide"].astype(str)))
    train_counts = train_df["allele"].value_counts().to_dict()

    # -------------------------------------------------------------------------
    # 1. Taxonomic & Groove-Indel Coverage Audit (2.3.0 vs mhcseqs 2.9.0)
    # -------------------------------------------------------------------------
    print("Auditing taxonomic coverage and IMGT groove indels across mhcseqs 2.9.0...")
    orig_by_clade = collections.Counter()
    orig_species = set()
    for a in orig_supported:
        p = mhcgnomes.parse(a, raise_on_error=False)
        if p is not None and getattr(p, "species", None) is not None:
            sp = p.species.name
            orig_species.add(sp)
            orig_by_clade[classify_clade(sp)] += 1

    ms_by_clade = collections.Counter()
    ms_species = set()
    ms_species_by_clade = collections.defaultdict(set)
    orig_species_by_clade = collections.defaultdict(set)
    for sp in orig_species:
        orig_species_by_clade[classify_clade(sp)].add(sp)

    by_two_field = {}
    by_source_id = {}
    saha_uniprot_rows = []
    indel_counts = collections.Counter()
    ins_pos_counts = collections.Counter()
    del_pos_counts = collections.Counter()
    g1_cache = {}
    g2_cache = {}
    total_ci_complete = 0

    with v290_csv.open(newline="", encoding="utf-8") as fd:
        for r in csv.DictReader(fd):
            if (r.get("mhc_class") or "").upper() != "I":
                continue
            g1 = (r.get("groove1") or "").strip()
            g2 = (r.get("groove2") or "").strip()
            if not (g1 and g2):
                continue
            total_ci_complete += 1
            sp = (r.get("species") or "").strip()
            sp_clean = sp.split("(")[0].strip() if sp else ""
            if sp_clean:
                ms_species.add(sp_clean)
                clade = classify_clade(sp_clean)
                ms_by_clade[clade] += 1
                ms_species_by_clade[clade].add(sp_clean)
            tfa = (r.get("two_field_allele") or "").strip()
            sid = (r.get("source_id") or "").strip()
            if tfa:
                by_two_field[tfa] = r
            if sid:
                by_source_id[sid] = r
            if "sarcophilus" in sp.lower() and r.get("source") == "uniprot":
                saha_uniprot_rows.append(r)

            if g1 not in g1_cache:
                g1_cache[g1] = mhcseqs.align_domain_to_imgt(g1, domain="G-ALPHA1")
            if g2 not in g2_cache:
                g2_cache[g2] = mhcseqs.align_domain_to_imgt(g2, domain="G-ALPHA2")
            a1, a2 = g1_cache[g1], g2_cache[g2]
            has_ins = bool(a1.insertions or a2.insertions)
            has_del = bool(a1.deletions or a2.deletions)
            if not has_ins and not has_del and not a1.is_truncated and not a2.is_truncated:
                indel_counts["canonical"] += 1
            else:
                if has_ins:
                    indel_counts["with_insertions"] += 1
                if has_del:
                    indel_counts["with_deletions"] += 1
                if not has_ins and not has_del:
                    indel_counts["truncated_only"] += 1
            for ins in a1.insertions:
                ins_pos_counts[int(ins.after_mature_pos)] += 1
            for ins in a2.insertions:
                ins_pos_counts[int(ins.after_mature_pos)] += 1
            for d in a1.deletions:
                del_pos_counts[int(d.mature_pos)] += 1
            for d in a2.deletions:
                del_pos_counts[int(d.mature_pos)] += 1

    coverage = {
        "mhcflurry_230_alleles": len(orig_supported),
        "mhcflurry_230_species": len(orig_species),
        "mhcseqs_290_alleles": total_ci_complete,
        "mhcseqs_290_species": len(ms_species),
        "clades": [],
        "indel_counts": dict(indel_counts),
        "ins_pos_counts": {str(k): v for k, v in sorted(ins_pos_counts.items())},
        "del_pos_counts": {str(k): v for k, v in sorted(del_pos_counts.items())},
    }
    clade_order = [
        "Primates",
        "Ungulates & Cetacea",
        "Carnivora",
        "Rodents & Lagomorphs",
        "Birds (Aves)",
        "Bats (Chiroptera)",
        "Marsupials & Monotremes",
        "Reptiles, Amphibians & Fish",
    ]
    for cl in clade_order:
        coverage["clades"].append({
            "clade": cl,
            "alleles_230": int(orig_by_clade.get(cl, 0)),
            "alleles_290": int(ms_by_clade.get(cl, 0)),
            "species_230": len(orig_species_by_clade.get(cl, set())),
            "species_290": len(ms_species_by_clade.get(cl, set())),
        })

    # -------------------------------------------------------------------------
    # 2. Resolve exact Class I sequences via mhcseqs 2.9.0 (+ pinned PDB/NCBI)
    # -------------------------------------------------------------------------
    print("Resolving curated benchmark sequences...")
    resolved: dict[str, dict] = {}

    def add_from_seq(name: str, seq: str, source_desc: str, *, mutations: tuple[str, ...] = ()):
        p39, aln = extract_39_with_mhcseqs(seq, allele=name, mutations=mutations)
        resolved[name] = {
            "sequence": seq,
            "pseudo39": p39,
            "source": source_desc,
            "alignment_summary": aln.summary(),
            "has_insertions": bool(aln.has_insertions),
            "has_deletions": bool(aln.has_deletions),
        }
        print(f"  {name:24s} | {source_desc:28s} | {aln.summary():32s} | {p39}")

    v290_lookup = {
        "Trvu-UB*01:01": "TrichosurusVulpecula-UB*01:01",
        "Ptal-N*01:01": "PteropusAlecto-N*01:01",
        "FLA-E*003:01": "FelisCatus-E*003:01",
        "FLA-E*018:01": "FelisCatus-E*018:01",
        "FLA-K*007:01": "FelisCatus-K*007:01",
        "Anpl-UAA*01": "AnasPlatyrhynchos-UAA*01",
        "Anpl-UAA*76": "AnasPlatyrhynchos-UAA*76",
        "SLA-1*04:01": "SusSp-1*04:01",
        "Gaga-BF2*002:01": "GallusGallus-BF2*002:01",
        "BoLA-6*013:01": "BosSp-6*013:01",
    }
    for name, key in v290_lookup.items():
        r = by_two_field.get(key)
        assert r is not None, f"Missing {key} in v2.9.0 CSV"
        add_from_seq(name, r["sequence"], f"mhcseqs 2.9.0 ({r['source']}:{r['source_id']})")

    # Point mutants via mhcseqs mutations=
    add_from_seq(
        "SLA-1*04:01 R156A",
        resolved["SLA-1*04:01"]["sequence"],
        "mhcseqs mutations=('R156A',)",
        mutations=("R156A",),
    )
    add_from_seq(
        "FLA-E*018:01 W167S",
        resolved["FLA-E*018:01"]["sequence"],
        "mhcseqs mutations=('W167S',)",
        mutations=("W167S",),
    )

    # Additional verified crystal structures for clades/alleles not yet named in CSV
    add_from_seq("Xela-UAAg", fetch_pdb_heavy_chain(cache_dir, "6A2B"), "PDB 6A2B (X. laevis)")
    add_from_seq("Eqca-N*006:02", fetch_pdb_heavy_chain(cache_dir, "4ZUU"), "PDB 4ZUU (E. caballus)")

    hla_a0201_pseudo = predictor.allele_to_sequence["HLA-A*02:01"]

    # -------------------------------------------------------------------------
    # 3. Evaluate Zero-Shot & Held-Out Benchmarks
    # -------------------------------------------------------------------------
    decoy_benchmarks = []
    exp_benchmarks = []
    motif_records = {}
    mutant_records = []

    def eval_decoy_benchmark(
        label: str,
        allele_key: str,
        clade: str,
        pos_peps: list[str],
        *,
        pseudo39: str | None = None,
        multi_pseudos: list[str] | None = None,
        aln_summary: str = "canonical",
        has_insertions: bool = False,
        save_motif_key: str | None = None,
        seed: int = 42,
    ):
        pos_peps = sorted({p for p in pos_peps if 8 <= len(p) <= 14 and set(p) <= VALID_AA})
        dec = make_length_matched_decoys(pos_peps, ratio=10, seed=seed)
        y = np.array([1] * len(pos_peps) + [0] * len(dec))
        all_peps = pos_peps + dec
        if multi_pseudos is not None:
            mat = np.vstack([score_peptides(predictor, all_peps, p) for p in multi_pseudos])
            ic50 = mat.min(axis=0)
        else:
            ic50 = score_peptides(predictor, all_peps, pseudo39)
        sc = -np.log10(np.clip(ic50, 1e-3, 1e6))
        auroc, auroc_lo, auroc_hi = stratified_bootstrap_ci(y, sc, roc_auc_score)
        auprc, auprc_lo, auprc_hi = stratified_bootstrap_ci(y, sc, average_precision_score)
        ppv, ppv_lo, ppv_hi = stratified_bootstrap_ci(y, sc, ppv_at_n)

        ic50_ctrl = score_peptides(predictor, all_peps, hla_a0201_pseudo)
        sc_ctrl = -np.log10(np.clip(ic50_ctrl, 1e-3, 1e6))
        ctrl_auroc, ctrl_lo, ctrl_hi = stratified_bootstrap_ci(y, sc_ctrl, roc_auc_score)

        rec = {
            "label": label,
            "allele": allele_key,
            "clade": clade,
            "n_pos": len(pos_peps),
            "n_dec": len(dec),
            "auroc": auroc, "auroc_lo": auroc_lo, "auroc_hi": auroc_hi,
            "auprc": auprc, "auprc_lo": auprc_lo, "auprc_hi": auprc_hi,
            "ppv": ppv, "ppv_lo": ppv_lo, "ppv_hi": ppv_hi,
            "ctrl_auroc": ctrl_auroc, "ctrl_lo": ctrl_lo, "ctrl_hi": ctrl_hi,
            "med_ic50_pos": float(np.median(ic50[:len(pos_peps)])),
            "med_ic50_dec": float(np.median(ic50[len(pos_peps):])),
            "alignment_summary": aln_summary,
            "has_insertions": has_insertions,
        }
        decoy_benchmarks.append(rec)
        print(
            f"[DEC] {label:32s} | n={len(pos_peps):4d} | AUROC={auroc:.3f} [{auroc_lo:.3f},{auroc_hi:.3f}] "
            f"(HLA-A2={ctrl_auroc:.3f}) | PPV={ppv:.3f}"
        )

        if save_motif_key:
            # Generate 30,000 random 9-mers and take top 1% (300) predicted binders for crisp predicted logo
            rng = np.random.default_rng(999)
            rand_9 = ["".join(rng.choice(AA_LIST, size=9, p=AA_BG_PROBS)) for _ in range(30000)]
            if multi_pseudos is not None:
                ic_r = np.vstack([score_peptides(predictor, rand_9, p) for p in multi_pseudos]).min(axis=0)
            else:
                ic_r = score_peptides(predictor, rand_9, pseudo39)
            top_idx = np.argsort(ic_r)[:300]
            top_pred_9 = [rand_9[i] for i in top_idx]
            emp_9 = [p for p in pos_peps if len(p) == 9]
            motif_records[save_motif_key] = {
                "title": label,
                "clade": clade,
                "n_emp_9mer": len(emp_9),
                "auroc": auroc,
                "alignment_summary": aln_summary,
                "has_insertions": has_insertions,
                "empirical_ic": compute_9mer_ic_matrix(emp_9),
                "predicted_ic": compute_9mer_ic_matrix(top_pred_9),
            }

    # A. Brushtail possum (Trvu-UB*01:01 — 5,607 monoallelic MS ligands, 0 training rows in 2.3.0!)
    trvu_peps = list(train_df[train_df["allele"] == "Trvu-UB*01:01"]["peptide"].astype(str).str.strip().str.upper())
    eval_decoy_benchmark(
        "Possum Trvu-UB*01:01 (MS)",
        "Trvu-UB*01:01",
        "Marsupialia",
        trvu_peps,
        pseudo39=resolved["Trvu-UB*01:01"]["pseudo39"],
        aln_summary=resolved["Trvu-UB*01:01"]["alignment_summary"],
        has_insertions=resolved["Trvu-UB*01:01"]["has_insertions"],
        save_motif_key="Trvu-UB*01:01",
    )

    # B. Tasmanian devil (Saha — multi-allelic MS ligands, 0 training rows in 2.3.0!)
    saha_rows = fetch_iedb_cached(cache_dir, "Saha class I", limit=5000)
    saha_peps = sorted({
        (r.get("linear_sequence") or "").strip().upper()
        for r in saha_rows
        if (r.get("qualitative_measure") or "").startswith("Positive")
        and 8 <= len((r.get("linear_sequence") or "").strip()) <= 14
        and set((r.get("linear_sequence") or "").strip().upper()) <= VALID_AA
    })[:1500]
    saha_pseudos = []
    seen_p = set()
    for r in saha_uniprot_rows:
        p39, _ = extract_39_with_mhcseqs(r["sequence"])
        if "X" not in p39 and p39 not in seen_p:
            seen_p.add(p39)
            saha_pseudos.append(p39)
    eval_decoy_benchmark(
        f"Tas. devil Saha ({len(saha_pseudos)} alleles, MS)",
        "Saha class I",
        "Marsupialia",
        saha_peps,
        multi_pseudos=saha_pseudos,
        aln_summary="multi-allelic UniProt grooves",
        has_insertions=False,
        save_motif_key="Saha",
        seed=123,
    )

    # C. IEDB allele-resolved benchmarks (both decoy-ranking and experimental pos vs neg)
    iedb_specs = [
        ("Cat FLA-E*003:01 (MS)", "FLA-E*00301", "FLA-E*003:01", "Carnivora", False, "FLA-E*003:01"),
        ("Cat FLA-K*007:01", "FLA-K*00701", "FLA-K*007:01", "Carnivora", False, None),
        ("Cat FLA-E*018:01", "FLA-E*01801", "FLA-E*018:01", "Carnivora", False, None),
        ("Duck Anpl-UAA*01", "Anpl-UAA*01", "Anpl-UAA*01", "Aves", False, None),
        ("Duck Anpl-UAA*76", "Anpl-UAA*76", "Anpl-UAA*76", "Aves", False, None),
        ("Horse Eqca-N*006:02", "Eqca-N*00602", "Eqca-N*006:02", "Perissodactyla", False, None),
        ("Frog Xela-UAAg", "Xela-UAAg", "Xela-UAAg", "Amphibia", False, None),
        ("Bat Ptal-N*01:01", "Ptal-N*01:01", "Ptal-N*01:01", "Chiroptera", False, "Ptal-N*01:01"),
        ("Chicken Gaga-BF2*002:01 (held-out)", "Gaga-BF2*002:01", "Gaga-BF2*002:01", "Held-out ref", True, None),
        ("Cattle BoLA-6*013:01 (held-out)", "BoLA-6*013:01", "BoLA-6*013:01", "Held-out ref", True, None),
    ]

    for label, rest, key_name, clade, is_heldout, motif_key in iedb_specs:
        rows = fetch_iedb_cached(cache_dir, rest, limit=3000)
        peps_map = {}
        for r in rows:
            p = (r.get("linear_sequence") or "").strip().upper()
            if not (8 <= len(p) <= 14 and set(p) <= VALID_AA):
                continue
            if is_heldout and (key_name, p) in train_pairs:
                continue
            q = r.get("qualitative_measure") or ""
            y_val = 1 if q.startswith("Positive") else (0 if q == "Negative" else None)
            if y_val is not None:
                peps_map[p] = y_val
        pos = sorted(p for p, yv in peps_map.items() if yv == 1)
        neg = sorted(p for p, yv in peps_map.items() if yv == 0)
        info = resolved[key_name]

        if not is_heldout and len(pos) >= 7:
            eval_decoy_benchmark(
                label,
                key_name,
                clade,
                pos,
                pseudo39=info["pseudo39"],
                aln_summary=info["alignment_summary"],
                has_insertions=info["has_insertions"],
                save_motif_key=motif_key,
            )

        if len(pos) >= 5 and len(neg) >= 5:
            peps_all = pos + neg
            y = np.array([1] * len(pos) + [0] * len(neg))
            ic50 = score_peptides(predictor, peps_all, info["pseudo39"])
            sc = -np.log10(np.clip(ic50, 1e-3, 1e6))
            auroc, lo, hi = stratified_bootstrap_ci(y, sc, roc_auc_score)
            auprc, ap_lo, ap_hi = stratified_bootstrap_ci(y, sc, average_precision_score)
            ic50_ctrl = score_peptides(predictor, peps_all, hla_a0201_pseudo)
            ctrl_auroc, _, _ = stratified_bootstrap_ci(
                y, -np.log10(np.clip(ic50_ctrl, 1e-3, 1e6)), roc_auc_score
            )
            exp_benchmarks.append({
                "label": label,
                "allele": key_name,
                "clade": clade,
                "zero_shot": not is_heldout,
                "n_train_230": int(train_counts.get(key_name, 0)) if is_heldout else 0,
                "n_pos": len(pos),
                "n_neg": len(neg),
                "auroc": auroc, "auroc_lo": lo, "auroc_hi": hi,
                "auprc": auprc, "auprc_lo": ap_lo, "auprc_hi": ap_hi,
                "ctrl_auroc": ctrl_auroc,
                "med_ic50_pos": float(np.median(ic50[:len(pos)])),
                "med_ic50_neg": float(np.median(ic50[len(pos):])),
                "alignment_summary": info["alignment_summary"],
                "has_insertions": info["has_insertions"],
            })
            print(
                f"[EXP] {label:32s} | pos={len(pos):3d} neg={len(neg):3d} | "
                f"AUROC={auroc:.3f} [{lo:.3f},{hi:.3f}] (HLA-A2={ctrl_auroc:.3f}) | "
                f"medIC50 pos={np.median(ic50[:len(pos)]):.0f} neg={np.median(ic50[len(pos):]):.0f}"
            )

    # D. Binding-pocket point mutants (SLA-1*04:01 R156A, FLA-E*018:01 W167S)
    mutant_specs = [
        ("Pig SLA-1*04:01 R156A", "SLA-1*04:01 R156A mutant", "SLA-1*04:01 R156A", "SLA-1*04:01", "Artiodactyla"),
        ("Cat FLA-E*018:01 W167S", "FLA-E*01801 W167S mutant", "FLA-E*018:01 W167S", "FLA-E*018:01", "Carnivora"),
    ]
    for label, rest, mut_key, wt_key, clade in mutant_specs:
        rows = fetch_iedb_cached(cache_dir, rest, limit=3000)
        peps_map = {}
        for r in rows:
            p = (r.get("linear_sequence") or "").strip().upper()
            if not (8 <= len(p) <= 14 and set(p) <= VALID_AA):
                continue
            q = r.get("qualitative_measure") or ""
            yv = 1 if q.startswith("Positive") else (0 if q == "Negative" else None)
            if yv is not None:
                peps_map[p] = yv
        pos = sorted(p for p, yv in peps_map.items() if yv == 1)
        neg = sorted(p for p, yv in peps_map.items() if yv == 0)
        if len(neg) >= 5:
            mode = "Experimental pos vs neg"
            eval_peps = pos + neg
            y = np.array([1] * len(pos) + [0] * len(neg))
        else:
            mode = "Mutant MS ligands vs 10:1 decoys"
            dec = make_length_matched_decoys(pos, ratio=10, seed=42)
            eval_peps = pos + dec
            y = np.array([1] * len(pos) + [0] * len(dec))
        ic_mut = score_peptides(predictor, eval_peps, resolved[mut_key]["pseudo39"])
        ic_wt = score_peptides(predictor, eval_peps, resolved[wt_key]["pseudo39"])
        sc_mut = -np.log10(np.clip(ic_mut, 1e-3, 1e6))
        sc_wt = -np.log10(np.clip(ic_wt, 1e-3, 1e6))
        m_auc, m_lo, m_hi = stratified_bootstrap_ci(y, sc_mut, roc_auc_score)
        w_auc, w_lo, w_hi = stratified_bootstrap_ci(y, sc_wt, roc_auc_score)
        # Paired bootstrap delta
        rng = np.random.default_rng(2026)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        deltas = []
        for _ in range(1000):
            idx = np.concatenate([
                rng.choice(pos_idx, len(pos_idx), replace=True),
                rng.choice(neg_idx, len(neg_idx), replace=True),
            ])
            deltas.append(roc_auc_score(y[idx], sc_mut[idx]) - roc_auc_score(y[idx], sc_wt[idx]))
        d_lo, d_hi = np.percentile(deltas, [2.5, 97.5])
        mutant_records.append({
            "label": label,
            "mut_key": mut_key,
            "wt_key": wt_key,
            "clade": clade,
            "mode": mode,
            "n_pos": len(pos),
            "n_neg": len(eval_peps) - len(pos),
            "mut_auroc": m_auc, "mut_lo": m_lo, "mut_hi": m_hi,
            "wt_auroc": w_auc, "wt_lo": w_lo, "wt_hi": w_hi,
            "delta_auroc": m_auc - w_auc, "delta_lo": float(d_lo), "delta_hi": float(d_hi),
            "pos_ic50_wt": [float(x) for x in ic_wt[:len(pos)]],
            "pos_ic50_mut": [float(x) for x in ic_mut[:len(pos)]],
            "neg_ic50_wt": [float(x) for x in ic_wt[len(pos):len(pos) + min(200, len(eval_peps) - len(pos))]],
            "neg_ic50_mut": [float(x) for x in ic_mut[len(pos):len(pos) + min(200, len(eval_peps) - len(pos))]],
            "med_pos_wt": float(np.median(ic_wt[:len(pos)])),
            "med_pos_mut": float(np.median(ic_mut[:len(pos)])),
        })
        print(
            f"[MUT] {label:26s} | WT={w_auc:.3f} -> MUT={m_auc:.3f} "
            f"(delta={m_auc - w_auc:+.3f} [{d_lo:+.3f},{d_hi:+.3f}]) | "
            f"medIC50 pos: WT={np.median(ic_wt[:len(pos)]):.0f} -> MUT={np.median(ic_mut[:len(pos)]):.0f} nM"
        )

    results = {
        "mhcseqs_version": mhcseqs.__version__,
        "mhcgnomes_version": mhcgnomes.__version__,
        "coverage": coverage,
        "resolved": {k: {kk: vv for kk, vv in v.items() if kk != "sequence"} for k, v in resolved.items()},
        "decoy_benchmarks": decoy_benchmarks,
        "exp_benchmarks": exp_benchmarks,
        "motifs": motif_records,
        "mutants": mutant_records,
    }
    out_json = out_dir / "cross_species_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote results JSON to {out_json}")
    return results


def save_figure(fig, out_dir: Path, stem: str):
    for ext in ("svg", "png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_species_coverage(results: dict, out_dir: Path):
    apply_paper_style()
    cov = results["coverage"]
    clades = list(cov["clades"])
    accounted_230 = sum(c["species_230"] for c in clades)
    accounted_290 = sum(c["species_290"] for c in clades)
    rem_230 = max(0, cov["mhcflurry_230_species"] - accounted_230)
    rem_290 = max(0, cov["mhcseqs_290_species"] - accounted_290)
    if rem_290 > 0 or rem_230 > 0:
        clades.append({
            "clade": "Other vertebrates",
            "species_230": rem_230,
            "species_290": rem_290,
        })

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.0, 4.6), gridspec_kw={"width_ratios": [1.22, 1.0]}
    )

    # Panel A: Species count per vertebrate clade (2.3.0 vs mhcseqs 2.9.0)
    labels = [c["clade"] for c in clades][::-1]
    sp_230 = [c["species_230"] for c in clades][::-1]
    sp_290 = [c["species_290"] for c in clades][::-1]
    y = np.arange(len(labels))
    h = 0.36

    ax1.barh(
        y - h / 2, sp_230, height=h, color=SIDE_B_COLOR,
        label=f"MHCflurry 2.3.0 ({cov['mhcflurry_230_species']} species, {cov['mhcflurry_230_alleles']:,} alleles)",
    )
    ax1.barh(
        y + h / 2, sp_290, height=h, color=SIDE_A_COLOR,
        label=f"MHCflurry 2.5.0 + mhcseqs 2.9.0 ({cov['mhcseqs_290_species']} species, {cov['mhcseqs_290_alleles']:,} grooves)",
    )
    for yi, v0, v1 in zip(y, sp_230, sp_290):
        ax1.text(v1 + 1.4, yi + h / 2, f"{v1}", va="center", ha="left", fontsize=7.5, color="#222222", fontweight="bold")
        if v0 > 0:
            ax1.text(v0 + 1.4, yi - h / 2, f"{v0}", va="center", ha="left", fontsize=7.5, color="#666666")
        else:
            ax1.text(1.4, yi - h / 2, "0", va="center", ha="left", fontsize=7.5, color="#999999", style="italic")

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Distinct vertebrate species with complete Class I grooves")
    ax1.set_title("A. Taxonomic expansion unlocked by mhcseqs 2.9.0", loc="left", fontweight="bold", pad=38)
    ax1.set_xlim(0, max(sp_290) * 1.16)
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=1, fontsize=7.5, frameon=False)
    despine(ax1)
    ax1.xaxis.grid(True, color="0.90", linewidth=0.6)
    ax1.set_axisbelow(True)

    # Panel B: IMGT groove indel hotspots across 27,243 Class I grooves
    ins_map = {int(k): v for k, v in cov["ins_pos_counts"].items()}
    del_map = {int(k): v for k, v in cov["del_pos_counts"].items()}
    positions = np.arange(1, 184)
    ins_vals = np.array([ins_map.get(p, 0) for p in positions])
    del_vals = np.array([del_map.get(p, 0) for p in positions])

    ax2.bar(positions, ins_vals, width=1.6, color="#D55E00", alpha=0.88, label="Insertions")
    ax2.bar(positions, -del_vals, width=1.6, color="#0072B2", alpha=0.88, label="Deletions")
    ax2.axhline(0, color="0.25", linewidth=0.8)
    ax2.axvline(90.5, color="0.55", linestyle="--", linewidth=0.8)
    ax2.text(45, max(ins_vals) * 1.08, r"$\alpha_1$ (G-ALPHA1)", ha="center", va="bottom", fontsize=8, color="0.35")
    ax2.text(137, max(ins_vals) * 1.08, r"$\alpha_2$ (G-ALPHA2)", ha="center", va="bottom", fontsize=8, color="0.35")

    ax2.annotate(
        "α1 hinge\n(bat MDL / possum RVG)",
        xy=(52, ins_map.get(52, 0) + ins_map.get(53, 0) * 0.35),
        xytext=(5, max(ins_vals) * 0.72), fontsize=7,
        arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.7},
    )
    ax2.annotate(
        "α2 kink", xy=(149, max(ins_vals[144:156])),
        xytext=(118, max(ins_vals) * 0.75), fontsize=7,
        arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.7},
    )
    ax2.annotate(
        "CD-turn\n(-2 aa)", xy=(41, -max(del_vals[38:44])),
        xytext=(4, -max(del_vals) * 0.85), fontsize=7,
        arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.7},
    )

    ax2.set_xlabel("Canonical mature Class I groove position (HLA-A*02:01 frame)")
    ax2.set_ylabel("Alleles with indel (+ins / −del)")
    ax2.set_title("B. Groove indel hotspots across 27,243 Class I alleles", loc="left", fontweight="bold", pad=38)
    ax2.set_xlim(0, 184)
    ax2.set_ylim(-max(del_vals) * 1.25, max(ins_vals) * 1.25)
    ax2.legend(loc="lower right", fontsize=7.5)
    despine(ax2)
    ax2.yaxis.grid(True, color="0.90", linewidth=0.6)
    ax2.set_axisbelow(True)

    fig.tight_layout(w_pad=2.2)
    save_figure(fig, out_dir, "species_coverage")


def plot_zero_shot_performance(results: dict, out_dir: Path):
    apply_paper_style()
    dec_rows = sorted(results["decoy_benchmarks"], key=lambda r: r["auroc"])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.9), gridspec_kw={"width_ratios": [1.22, 1.0]}
    )

    # Panel A: Zero-shot ligand ranking vs 10:1 length-matched decoys
    y1 = np.arange(len(dec_rows))
    ax1.axvline(0.5, color="0.65", linestyle=":", linewidth=0.9)
    for idx, r in enumerate(dec_rows):
        color = CLADE_COLORS.get(r["clade"], SIDE_A_COLOR)
        ax1.plot(
            r["ctrl_auroc"], idx, marker="D", markersize=4.8,
            color="0.68", markeredgecolor="0.35", markeredgewidth=0.6,
            zorder=2, label="HLA-A*02:01 negative control" if idx == 0 else None,
        )
        ax1.errorbar(
            r["auroc"], idx,
            xerr=[[r["auroc"] - r["auroc_lo"]], [r["auroc_hi"] - r["auroc"]]],
            fmt="o", markersize=6.2, color=color, ecolor=color,
            elinewidth=1.4, capsize=2.8, zorder=4,
            label="mhcseqs 39-aa groove (0-shot)" if idx == 0 else None,
        )
        ax1.plot([r["ctrl_auroc"], r["auroc"]], [idx, idx], color="0.80", linewidth=1.0, zorder=1)
        ax1.text(
            max(r["auroc_hi"], r["ctrl_auroc"]) + 0.015, idx,
            f"{r['auroc']:.3f}  (n={r['n_pos']:,})",
            va="center", ha="left", fontsize=7.3, color="0.2",
        )
    ax1.set_yticks(y1)
    ax1.set_yticklabels([
        r["label"] + (" *" if r["has_insertions"] else "")
        for r in dec_rows
    ], fontsize=8)
    ax1.set_xlim(0.42, 1.26)
    ax1.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax1.set_xlabel("AUROC (95% stratified bootstrap CI)")
    ax1.set_title("A. Zero-shot ligand ranking (10:1 decoys vs HLA-A*02:01 ctrl)", loc="left", fontweight="bold", pad=22)
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2, fontsize=7.4, frameon=False)
    despine(ax1)
    ax1.xaxis.grid(True, color="0.90", linewidth=0.6)
    ax1.set_axisbelow(True)

    # Panel B: Predicted median IC50 separation (Positives vs Decoys, log scale)
    ax2.axvline(500.0, color="0.65", linestyle=":", linewidth=0.9)
    for idx, r in enumerate(dec_rows):
        color = CLADE_COLORS.get(r["clade"], SIDE_A_COLOR)
        pos_med = r["med_ic50_pos"]
        dec_med = r["med_ic50_dec"]
        ax2.plot([pos_med, dec_med], [idx, idx], color="0.78", linewidth=1.4, zorder=1)
        ax2.plot(
            dec_med, idx, marker="s", markersize=5.2,
            color="0.72", markeredgecolor="0.35", markeredgewidth=0.6, zorder=3,
            label="Background decoys (median)" if idx == 0 else None,
        )
        ax2.plot(
            pos_med, idx, marker="o", markersize=6.4, color=color,
            markeredgecolor="white", markeredgewidth=0.5, zorder=4,
            label="True ligands (median)" if idx == 0 else None,
        )
        fold = dec_med / max(pos_med, 1.0)
        fold_lbl = f"{fold:.0f}×" if fold >= 10 else f"{fold:.1f}×"
        ax2.text(
            dec_med * 1.22, idx, fold_lbl,
            va="center", ha="left", fontsize=7.3,
            color="#B03A2E" if fold < 2.0 else "0.25",
            fontweight="bold" if fold < 2.0 else "normal",
        )
    ax2.set_xscale("log")
    ax2.set_yticks(y1)
    ax2.set_yticklabels([])
    ax2.set_xlim(14, 120000)
    ax2.set_xticks([25, 100, 500, 2000, 10000, 40000])
    ax2.set_xticklabels(["25", "100", "500", "2k", "10k", "40k"])
    ax2.set_xlabel("Predicted median IC50 (nM, log scale; lower = stronger binding)")
    ax2.set_title("B. Predicted affinity separation (fold decoy / ligand IC50)", loc="left", fontweight="bold", pad=22)
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2, fontsize=7.4, frameon=False)
    despine(ax2)
    ax2.xaxis.grid(True, color="0.90", linewidth=0.6)
    ax2.set_axisbelow(True)

    fig.text(
        0.01, -0.02,
        "* Allele carries internal binding-groove insertion(s) omitted by fixed 39-aa pseudosequences (mhcseqs >= 2.9.0 logs a diagnostic warning). "
        "Every allele shown has 0 training rows in MHCflurry 2.3.0. Bat Ptal-N*01:01 (+3aa MDL A-pocket ins) collapses to flat ~26–30 µM IC50 (1.1×).",
        fontsize=7.3, color="0.35",
    )
    fig.tight_layout(w_pad=1.6)
    save_figure(fig, out_dir, "zero_shot_cross_species")


def plot_zero_shot_motifs(results: dict, out_dir: Path):
    import logomaker

    apply_paper_style()
    order = ["Trvu-UB*01:01", "FLA-E*003:01", "Saha", "Ptal-N*01:01"]
    keys = [k for k in order if k in results["motifs"]]
    n_cols = len(keys)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.1 * n_cols, 4.1), sharey=True)
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, k in enumerate(keys):
        m = results["motifs"][k]
        emp_df = pd.DataFrame(m["empirical_ic"])
        emp_df.index = np.arange(1, 10)
        pred_df = pd.DataFrame(m["predicted_ic"])
        pred_df.index = np.arange(1, 10)

        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        logomaker.Logo(emp_df, ax=ax_top, color_scheme="chemistry", vpad=0.02, width=0.88)
        logomaker.Logo(pred_df, ax=ax_bot, color_scheme="chemistry", vpad=0.02, width=0.88)

        if k == "FLA-E*003:01":
            title = f"Cat FLA-E*003:01\n(Feline, 0-shot, AUROC {m['auroc']:.2f})"
        elif k == "Saha":
            title = f"Tas. devil Saha\n(Marsupial, 0-shot, AUROC {m['auroc']:.2f})"
        elif k == "Trvu-UB*01:01":
            title = f"Possum Trvu-UB*01:01\n(Marsupial, 0-shot, AUROC {m['auroc']:.2f})"
        elif k == "Ptal-N*01:01":
            title = "Bat Ptal-N*01:01 (+3aa MDL)\n(Failure mode, EXP AUROC 0.60)"
        else:
            title = m["title"]
        ax_top.set_title(title, fontsize=8.3, fontweight="bold", pad=6)
        ax_top.text(
            0.97, 0.94, f"n={m['n_emp_9mer']:,} 9-mers",
            transform=ax_top.transAxes, ha="right", va="top", fontsize=7.0, color="0.35",
        )
        for ax in (ax_top, ax_bot):
            ax.set_xticks(range(1, 10))
            ax.set_xticklabels([f"P{i}" for i in range(1, 10)], fontsize=7.5)
            ax.set_ylim(0, 2.65)
            despine(ax)
        ax_bot.set_xlabel("Peptide position", fontsize=8)
        if col == 0:
            ax_top.set_ylabel("Empirical ligands\nInformation (bits)", fontsize=8, fontweight="bold")
            ax_bot.set_ylabel("Zero-shot predicted\nInformation (bits)", fontsize=8, fontweight="bold")

    fig.tight_layout(h_pad=1.0, w_pad=0.9)
    save_figure(fig, out_dir, "zero_shot_motifs")


def plot_pocket_mutants(results: dict, out_dir: Path):
    apply_paper_style()
    muts = results["mutants"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), gridspec_kw={"width_ratios": [1.05, 1.0, 1.0]})

    # Panel A: Paired AUROC comparison (WT vs Mutant groove)
    ax0 = axes[0]
    x = np.arange(len(muts))
    w = 0.34
    wt_vals = [m["wt_auroc"] for m in muts]
    mut_vals = [m["mut_auroc"] for m in muts]
    wt_err = [[m["wt_auroc"] - m["wt_lo"] for m in muts], [m["wt_hi"] - m["wt_auroc"] for m in muts]]
    mut_err = [[m["mut_auroc"] - m["mut_lo"] for m in muts], [m["mut_hi"] - m["mut_auroc"] for m in muts]]

    ax0.bar(x - w / 2, wt_vals, width=w, color=SIDE_B_COLOR, yerr=wt_err, capsize=3, error_kw={"lw": 1.1}, label="Wild-type groove")
    ax0.bar(x + w / 2, mut_vals, width=w, color=POSITIVE_DELTA_COLOR, yerr=mut_err, capsize=3, error_kw={"lw": 1.1}, label="Mutant groove (mhcseqs)")
    for xi, m in zip(x, muts):
        ax0.text(xi - w / 2, m["wt_hi"] + 0.012, f"{m['wt_auroc']:.3f}", ha="center", va="bottom", fontsize=7.2, color="0.35")
        ax0.text(xi + w / 2, m["mut_hi"] + 0.012, f"{m['mut_auroc']:.3f}", ha="center", va="bottom", fontsize=7.2, color="#1D5E46", fontweight="bold")
        ax0.text(
            xi, 0.72, f"Δ = {m['delta_auroc']:+.3f}", ha="center", va="center", fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.94},
        )

    ax0.set_xticks(x)
    ax0.set_xticklabels([m["label"].replace(" ", "\n", 1) for m in muts], fontsize=8)
    ax0.set_ylim(0.5, 1.07)
    ax0.set_ylabel("AUROC on mutant ligands (95% CI)")
    ax0.set_title("A. Point-mutant groove discrimination", loc="left", fontweight="bold", pad=22)
    ax0.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2, fontsize=7.1, frameon=False)
    despine(ax0)
    ax0.yaxis.grid(True, color="0.90", linewidth=0.6)
    ax0.set_axisbelow(True)

    # Panels B & C: Paired per-peptide IC50 shift (log10 nM)
    for panel_idx, m in enumerate(muts[:2]):
        ax = axes[panel_idx + 1]
        pos_wt = np.log10(np.clip(m["pos_ic50_wt"], 1.0, 50000.0))
        pos_mut = np.log10(np.clip(m["pos_ic50_mut"], 1.0, 50000.0))
        neg_wt = np.log10(np.clip(m["neg_ic50_wt"], 1.0, 50000.0))
        neg_mut = np.log10(np.clip(m["neg_ic50_mut"], 1.0, 50000.0))

        ax.scatter(neg_wt, neg_mut, s=12, alpha=0.35, color="0.65", edgecolors="none", label="Negatives / decoys")
        ax.scatter(pos_wt, pos_mut, s=18, alpha=0.78, color=CLADE_COLORS.get(m["clade"], SIDE_A_COLOR),
                   edgecolors="white", linewidths=0.3, label="Mutant ligands")
        ax.plot([0.5, 4.8], [0.5, 4.8], linestyle="--", color="0.4", linewidth=0.9)
        ax.set_xlim(0.5, 4.8)
        ax.set_ylim(0.5, 4.8)
        ax.set_xlabel(r"Wild-type $\log_{10}(\mathrm{IC}_{50}\ \mathrm{nM})$")
        ax.set_ylabel(r"Mutant $\log_{10}(\mathrm{IC}_{50}\ \mathrm{nM})$")
        letter = "B" if panel_idx == 0 else "C"
        ax.set_title(f"{letter}. {m['label']}", loc="left", fontweight="bold", pad=22)
        ax.text(
            0.05, 0.94,
            f"Pos med: {m['med_pos_wt']:.0f} → {m['med_pos_mut']:.0f} nM",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.9},
        )
        if panel_idx == 0:
            ax.legend(loc="lower right", fontsize=7.2)
        despine(ax)
        ax.grid(True, color="0.92", linewidth=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout(w_pad=1.8)
    save_figure(fig, out_dir, "pocket_mutants")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--mhcseqs-csv",
        default=None,
        help="Path to mhc-full-seqs.csv (defaults to mhcseqs.find_csv())",
    )
    parser.add_argument(
        "--out-dir",
        default="./cross_species_250",
        help="Output directory for results JSON, cache, and rendered figures",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_json = out_dir / "cross_species_results.json"

    if args.collect or not results_json.exists():
        csv_path = Path(args.mhcseqs_csv) if args.mhcseqs_csv else Path(mhcseqs.find_csv())
        results = collect_results(csv_path, out_dir)
    else:
        results = json.loads(results_json.read_text(encoding="utf-8"))

    if args.plot:
        plot_species_coverage(results, out_dir)
        plot_zero_shot_performance(results, out_dir)
        plot_zero_shot_motifs(results, out_dir)
        plot_pocket_mutants(results, out_dir)
        print(f"Rendered 4 publication-ready figures to {out_dir}")


if __name__ == "__main__":
    main()
