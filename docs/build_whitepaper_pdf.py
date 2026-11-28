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

"""Build the publication-ready MHCflurry 2.5.x cross-species & pan-class whitepaper PDF.

Converts the sixteen standalone single-panel publication vector SVGs in
``docs/_static/whitepaper/figures/`` losslessly to vector PDF via
``rsvg-convert``, compiles the XeLaTeX manuscript, and writes
``docs/_static/whitepaper/mhcflurry_25x_cross_species_whitepaper.pdf``.

Requirements: ``rsvg-convert`` and ``xelatex`` (TeX Live).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


DOCS_DIR = Path(__file__).resolve().parent
FIG_DIR = DOCS_DIR / "_static" / "whitepaper" / "figures"
DEFAULT_OUTPUT = (
    DOCS_DIR / "_static" / "whitepaper" / "mhcflurry_25x_cross_species_whitepaper.pdf"
)

FIGURES = {
    "fig01.pdf": FIG_DIR / "fig01_taxonomic_expansion.svg",
    "fig02.pdf": FIG_DIR / "fig02_hitlist_validation_landscape.svg",
    "fig03.pdf": FIG_DIR / "fig03_ms_presentation_auroc.svg",
    "fig04.pdf": FIG_DIR / "fig04_ms_presentation_ppv.svg",
    "fig05.pdf": FIG_DIR / "fig05_experimental_binding_assays.svg",
    "fig06.pdf": FIG_DIR / "fig06_zeroshot_ligand_ranking.svg",
    "fig07.pdf": FIG_DIR / "fig07_groove_specificity_vs_control.svg",
    "fig08.pdf": FIG_DIR / "fig08_scrambled_peptide_control.svg",
    "fig09.pdf": FIG_DIR / "fig09_zero_shot_motifs.svg",
    "fig10.pdf": FIG_DIR / "fig10_training_parity.svg",
    "fig11.pdf": FIG_DIR / "fig11_cattle_dog_overlap_audit.svg",
    "fig12.pdf": FIG_DIR / "fig12_cattle_bola_disjoint_auroc.svg",
    "fig13.pdf": FIG_DIR / "fig13_cattle_bola_fewshot_ablation.svg",
    "fig14.pdf": FIG_DIR / "fig14_class_ii_auroc.svg",
    "fig15.pdf": FIG_DIR / "fig15_class_ii_spearman.svg",
    "fig16.pdf": FIG_DIR / "fig16_class_ii_specificity_matrix.svg",
}


TEX_SOURCE = r"""\documentclass[10.5pt,letterpaper]{article}

\usepackage[margin=0.82in, top=0.88in, bottom=0.88in, headheight=19pt]{geometry}
\usepackage{fontspec}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{tcolorbox}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{caption}
\usepackage{float}
\usepackage{enumitem}
\usepackage{amsmath,amssymb}
\usepackage{xurl}
\usepackage[hidelinks,
  pdftitle={Pan-Vertebrate MHC Class I Generalization and Exploratory Pan-Class Transfer in MHCflurry 2.5},
  pdfauthor={MHCflurry Contributors (OpenVax / PIRL-UNC)},
  pdfsubject={Cross-species MHC Class I Affinity and Presentation Validation and Zero-Shot Class I to Class II Boundary Study}
]{hyperref}

\DeclareUrlCommand\code{\urlstyle{tt}}

% --- Typography & Palette ---
\setmainfont{Charter}[
  BoldFont={Charter Bold},
  ItalicFont={Charter Italic},
  BoldItalicFont={Charter Bold Italic}
]
\setsansfont{Avenir Next}[
  BoldFont={Avenir Next Demi Bold},
  ItalicFont={Avenir Next Italic},
  BoldItalicFont={Avenir Next Demi Bold Italic}
]
\setmonofont{Menlo}[Scale=0.84]

\definecolor{Navy}{HTML}{123254}
\definecolor{Teal}{HTML}{0E6E75}
\definecolor{Slate}{HTML}{4A5568}
\definecolor{LightBg}{HTML}{F4F7FA}
\definecolor{RuleGray}{HTML}{CBD5E1}
\definecolor{RowTint}{HTML}{F8FAFC}
\definecolor{WarnBg}{HTML}{FFF8EB}
\definecolor{WarnBorder}{HTML}{C26D0A}

\hypersetup{
  colorlinks=true,
  linkcolor=Navy,
  urlcolor=Teal,
  citecolor=Navy
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{5.5pt}
\renewcommand{\arraystretch}{1.16}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\setlist[itemize]{leftmargin=1.4em, itemsep=2.5pt, topsep=2pt}
\setlist[enumerate]{leftmargin=1.6em, itemsep=2.5pt, topsep=2pt}

% --- Section Styling ---
\titleformat{\section}
  {\sffamily\bfseries\large\color{Navy}}
  {\thesection.}{0.55em}{}
  [\vspace{-0.45em}{\color{RuleGray}\rule{\textwidth}{0.6pt}}]
\titlespacing*{\section}{0pt}{14pt}{5pt}

\titleformat{\subsection}
  {\sffamily\bfseries\normalsize\color{Teal}}
  {\thesubsection}{0.55em}{}
\titlespacing*{\subsection}{0pt}{9pt}{3pt}

% --- Captions ---
\captionsetup{
  font={small,sf},
  labelfont={bf,color=Navy},
  textfont={color=Slate},
  justification=justified,
  singlelinecheck=false,
  skip=6pt
}

% --- Headers & Footers ---
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\sffamily\footnotesize\color{Slate}MHCflurry 2.5 Technical Whitepaper}
\fancyhead[R]{\sffamily\footnotesize\color{Slate}Pan-Vertebrate Class I Generalization \& Pan-Class Boundary Study}
\fancyfoot[L]{\sffamily\footnotesize\color{Slate}\href{https://www.openvax.org/mhcflurry/whitepaper_25x_cross_species.html}{www.openvax.org/mhcflurry}}
\fancyfoot[R]{\sffamily\footnotesize\color{Slate}Page \thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\headrule}{\hbox to\headwidth{\color{RuleGray}\leaders\hrule height \headrulewidth\hfill}}
\renewcommand{\footrulewidth}{0pt}

\fancypagestyle{firstpage}{
  \fancyhf{}
  \fancyfoot[L]{\sffamily\footnotesize\color{Slate}OpenVax / PIRL-UNC $\cdot$ MHCflurry 2.5.x (frozen 2.3.0 PyTorch weights) $\cdot$ \href{https://github.com/openvax/mhcflurry}{github.com/openvax/mhcflurry}}
  \fancyfoot[R]{\sffamily\footnotesize\color{Slate}Page \thepage}
  \renewcommand{\headrulewidth}{0pt}
}

% --- Callout Boxes ---
\newtcolorbox{summarybox}[1]{
  colback=LightBg,
  colframe=Navy,
  boxrule=0.8pt,
  arc=3pt,
  left=10pt, right=10pt, top=7pt, bottom=7pt,
  title={\sffamily\bfseries\small #1},
  coltitle=white,
  colbacktitle=Navy,
  fontupper=\small
}

\newtcolorbox{caveatbox}[1]{
  colback=WarnBg,
  colframe=WarnBorder,
  boxrule=0.7pt,
  arc=3pt,
  left=10pt, right=10pt, top=6pt, bottom=6pt,
  title={\sffamily\bfseries\small #1},
  coltitle=white,
  colbacktitle=WarnBorder,
  fontupper=\small
}

\begin{document}
\thispagestyle{firstpage}

% ============================================================================
% TITLE BLOCK
% ============================================================================
{\sffamily\footnotesize\bfseries\color{Teal}TECHNICAL WHITEPAPER $\cdot$ OPENVAX / MHCFLURRY 2.5 SERIES}
\vspace{3pt}

{\sffamily\bfseries\LARGE\color{Navy}Pan-Vertebrate MHC Class I Generalization and Exploratory Pan-Class Transfer in MHCflurry 2.5\par}
\vspace{4pt}

{\sffamily\normalsize\color{Slate}Zero-Shot Binding Affinity and Antigen Presentation Across 223 Vertebrate Species, Audited Livestock and Companion-Animal Benchmarks, and an Experimental Class I $\rightarrow$ Class II Structural Boundary Test\par}
\vspace{6pt}

{\sffamily\small\textbf{MHCflurry Contributors} (OpenVax \& PIRL-UNC) $\cdot$ \textbf{Date:} November 2026\\
\textbf{Software:} \texttt{mhcflurry 2.5.x} (frozen \texttt{2.3.0} weights), \texttt{mhcseqs 2.9–2.14}, \texttt{mhcgnomes 3.79–3.96}\\
\textbf{Web Companion:} \href{https://www.openvax.org/mhcflurry/whitepaper_25x_cross_species.html}{\texttt{www.openvax.org/mhcflurry/whitepaper\_25x\_cross\_species.html}}\par}
\vspace{6pt}
{\color{Navy}\rule{\textwidth}{1.2pt}}
\vspace{6pt}

% ============================================================================
% ABSTRACT + EXECUTIVE SUMMARY
% ============================================================================
\begin{summarybox}{Abstract \& Key Findings}
MHCflurry~2.5 integrates the optional \href{https://github.com/pirl-unc/mhcseqs}{\texttt{mhcseqs}} MHC sequence database and structural aligner alongside \href{https://github.com/pirl-unc/mhcgnomes}{\texttt{mhcgnomes}} allele-name parsing, expanding complete Class~I binding-groove coverage nearly five-fold—from 47~vertebrate species (10~trained species / 308~alleles in \texttt{2.3.0}) to \textbf{223--231~vertebrate species} (26,156--26,549~complete grooves across 14~clades)—\textbf{without retraining or altering existing \texttt{2.3.0} model weights}. Using rigorous negative controls (\textbf{50-allele mismatched real-allele null distributions}, \textbf{30-groove column-permuted pseudosequences}, and \textbf{within-peptide residue-scrambled positive controls}), we evaluate how frozen \texttt{2.3.0} weights generalize across newly added clades, engineered pocket mutants, audited cattle/dog MS cohorts, and an exploratory Class~I~$\rightarrow$~Class~II transfer test:
\begin{itemize}
  \item \textbf{Broad zero-shot Class~I transfer verified against null distributions:} Across 15~\texttt{hitlist} unseen-species and binding-pocket mutant cohorts (0~training rows), frozen \texttt{2.3.0} weights achieve \textbf{0.77--0.98 AUROC} across Carnivora (Cat \texttt{0.977}, Giant panda \texttt{0.883}), Perissodactyla (Horse \texttt{0.967}), Aves (Mallard duck \texttt{0.893–0.938}), Marsupialia (Brushtail possum \texttt{0.869}, Tasmanian devil \texttt{0.806 +F}), and engineered pocket mutants (Pig \texttt{R156A} \texttt{0.951}, $p=0.020$, beating 50/50 mismatched real alleles; Cat \texttt{W167S} \texttt{0.976}). Holding amino-acid composition constant via \textbf{within-peptide scrambled positives} confirms true positional anchor decoding ($+0.20$ to $+0.29$ AUROC over mismatched alleles on cat, duck \texttt{Anpl-UAA*76}, possum, cynomolgus, and pig \texttt{R156A}), while cleanly exposing non-specific or indel-limited cases (rabbit \texttt{RLA-A*01}, frog \texttt{Xela-UAAg}, bat \texttt{Ptal-N*01:01}).
  \item \textbf{Cross-clade transfer of human-trained antigen processing:} On Tasmanian devil (\textit{Sarcophilus harrisii}, $\sim$160~Myr diverged from placentals), MHCflurry's human-trained processing model alone achieves \textbf{0.750~AUROC} with real marsupial flanking sequences and lifts combined Presentation from \textbf{0.749 $\rightarrow$ 0.806~AUROC} ($+0.057$, $+30\%$ relative PPV@N; Affinity beats 50/50 size-matched 11-allele mismatched draws, \texttt{0.749} vs.\ \texttt{0.695 [0.674, 0.718]}).
  \item \textbf{Mandatory three-table overlap audit on Cattle \texttt{BoLA-I} \& Dog \texttt{DLA-88}:} Auditing against all three bundled \texttt{2.3.0} training tables catches \textbf{100\% in-sample overlap} on \texttt{hitlist} Dog \texttt{DLA-88} and small IEDB \texttt{BoLA-I} cohorts, while isolating \textbf{21,332 strictly unseen flanked Cattle \texttt{BoLA-I} MS ligands} (\texttt{PXD008151}) that reach \textbf{0.902--0.960 Presentation (+F) AUROC} (\texttt{0.833–0.922} on real vs.\ scrambled ligands vs.\ \texttt{0.604–0.731} null) and confirm strong ultra-few-shot transfer (4--12~training rows).
  \item \textbf{Clean negative result on zero-shot Class~I~$\rightarrow$~Class~II transfer:} Mapping 36/39 Class~II \texttt{DR}/\texttt{DQ}/\texttt{DP} groove positions onto frozen Class~I weights yields \textbf{0.551~AUROC} (9-mer core)—indistinguishable from a \textbf{30-allele mismatched Class~I null (0.549)}, below \textbf{column-permuted grooves (0.571)}, and far below \texttt{NetMHCIIpan-4.3 BA} (\textbf{0.923}). On \textbf{within-peptide scrambled Class~II binders}, frozen Class~I collapses to \textbf{0.502} (coin flip) while \texttt{NetMHCIIpan-4.3 BA} holds \textbf{0.725} ($+0.223$), proving that pan-class prediction requires \textbf{joint pan-class training}.
  \item \textbf{Bit-identical training parity:} Locking existing trained alleles via \texttt{--lock-existing} reproduces legacy \texttt{clustalo} pan-allele training with \textbf{Max $|\Delta| = 0.0$} and \textbf{Pearson $r = 1.000000$} across all 308~trained alleles.
\end{itemize}
\end{summarybox}

% ============================================================================
% SECTION 1: GLOSSARY & ARCHITECTURE
% ============================================================================
\newpage
\section{Companion Packages, Key Terms, and MHCflurry 2.5 Architecture}

\begin{table}[H]
\centering
\caption{\textbf{Companion packages, biological terms, and negative-control designs used in this whitepaper.}}
\label{tab:glossary}
\scriptsize\sffamily
\begin{tabularx}{\textwidth}{@{}L{3.7cm} X@{}}
\toprule
\textbf{Term / Package} & \textbf{Plain-English Meaning} \\
\midrule
\rowcolor{RowTint}
\href{https://github.com/pirl-unc/mhcseqs}{\textbf{\texttt{mhcseqs}}} & Optional companion library providing curated MHC protein sequences across $>220$ vertebrate species and a structural aligner mapping any MHC protein onto standard IMGT numbering. \\
\href{https://github.com/pirl-unc/mhcgnomes}{\textbf{\texttt{mhcgnomes}}} & Companion MHC allele-name parser used by MHCflurry to normalize naming conventions across species (e.g., \texttt{HLA-A0201} $\rightarrow$ \texttt{HLA-A*02:01}, plus UniProt-backed names for non-model vertebrates). \\
\rowcolor{RowTint}
\href{https://github.com/pirl-unc/hitlist}{\textbf{\texttt{hitlist}}} & Open benchmark corpus of curated MHC peptide-binding and mass-spectrometry elution datasets paired with matched species reference proteomes. \\
\textbf{39-aa pseudosequence} & The 39 amino-acid positions of an MHC class~I protein that line its peptide-binding pocket (groove) and directly contact the bound peptide. \\
\rowcolor{RowTint}
\textbf{Flanking sequences (\texttt{+F}/\texttt{noF})} & The source-protein amino acids immediately upstream (\texttt{n\_flank}) and downstream (\texttt{c\_flank}) of a peptide, used by MHCflurry's antigen-processing network to estimate cleavage and transport likelihood. \\
\textbf{AUROC \& PPV@N} & Ranking metrics against $10{:}1$ length-matched background decoys. AUROC ($0.5=$ random, $1.0=$ perfect) measures overall separation; PPV@N is precision among the top $N$ predictions ($N=$ true ligands). \\
\rowcolor{RowTint}
\textbf{Mismatched real-allele null ($N=50$)} & Scoring the identical test peptides with 50 diverse, unrelated real Class~I alleles (excluding the target species; size-matched $k$-allele random draws for multi-allelic haplotypes). Yields a true null distribution (median, 5--95\% band, empirical $p$-value) without the motif bias of a single control allele. \\
\textbf{Column-permuted grooves ($N=30$)} & Synthetic 39-aa pseudosequences created by independently shuffling each of the 39 columns across real Class~I alleles, keeping per-position residue frequencies in-distribution while breaking pocket identity. \\
\rowcolor{RowTint}
\textbf{Within-peptide scrambled positives} & Scoring real positive ligands against $10\times$ residue-shuffled copies of those exact same ligands (identical length and amino-acid composition). Eliminates hydrophobicity bias so AUROC above $0.50$ measures true positional anchor recognition. \\
\textbf{Closed vs.\ open groove} & MHC class~I holds peptides in a \emph{closed} pocket pinning both ends (favoring 8--11mers); MHC class~II has an \emph{open} groove where an internal 9-mer core binds while both ends of a longer peptide (15+ aa) overhang freely. \\
\rowcolor{RowTint}
\textbf{Groove insertion} & Extra amino acids inserted into the peptide-binding pocket of some non-model species (e.g., a 3-aa insertion in bat \texttt{Ptal-N*01:01}) that cannot be represented in a fixed 39-position vector. \\
\bottomrule
\end{tabularx}
\end{table}

In MHCflurry~2.0--2.3, building the 39-residue pseudosequence table required downloading bulk IPD-IMGT/HLA, IPD-MHC, and UniProt FASTA files and running external \texttt{clustalo} multiple-sequence alignments. That workflow covered 47~vertebrate species (20,205~alleles, with training measurements concentrated in only 10~species and 308~alleles) and could not register novel heavy chains, non-model UniProt sequences, or engineered binding-pocket mutants at inference time.

MHCflurry~2.5 resolves these limitations without changing any trained \texttt{2.3.0} weights:
\begin{enumerate}
  \item \textbf{Direct structural extraction via \href{https://github.com/pirl-unc/mhcseqs}{\texttt{mhcseqs}}:} Installing \texttt{mhcflurry[mhcseqs]} delegates signal-peptide trimming, binding-domain extraction, and mature-numbered 39-aa pseudosequence extraction to \texttt{mhcseqs}'s structural aligner. Users can register any raw vertebrate Class~I heavy chain or mature-numbered point mutation at runtime (\code{add_allele_from_sequence(..., mutations=("W167S",))}) or extend an entire predictor (\code{extend_alleles_from_mhcseqs()}).
  \item \textbf{Automatic insertion warnings:} When \texttt{mhcseqs $\ge$ 2.9} encounters extra amino acids inserted inside a peptide-binding pocket (which a fixed 39-position vector must skip), MHCflurry exposes a structured \texttt{GrooveAlignment} object and logs an explicit warning rather than silently returning an out-of-register pseudosequence.
  \item \textbf{Sequence-backed naming for non-model species:} Through \href{https://github.com/pirl-unc/mhcgnomes}{\texttt{mhcgnomes}} (\code{AccessionedAllele} support in \texttt{mhcflurry $\ge$ 2.5.9}), sequence-backed vertebrate identifiers (e.g., shark \code{TriakisScyllium-UAA*uniprot:O46817}, Tasmanian devil \code{Saha-classI*uniprot:G3V1L5}) flow cleanly through \code{normalize_allele_name} and \code{Class1AffinityPredictor} under \code{normalize_alleles=True} (26,156~alleles across 223~canonical vertebrate species, including 2,884~accessioned entries and 0~unparsed rows), while blocking bare UniProt accessions from misparsing as mammalian alleles.
  \item \textbf{Zero-regression parity gate:} Across 19,886 overlapping non-\texttt{X} alleles in \texttt{2.3.0}, \texttt{mhcseqs} achieves \textbf{99.45\% exact 39/39 match} (19,777/19,886) and \textbf{99.98\% per-residue identity} (775,385/775,554). The updated \code{downloads-generation/allele_sequences/GENERATE.sh} uses \texttt{--lock-existing} plus \code{verify_trained_allele_parity.py} to enforce 0~character differences across all 308~trained alleles.
\end{enumerate}

Complete Class~I coverage expands from 47~species in \texttt{2.3.0} to 227~species in \texttt{mhcseqs 2.12} (231~union, 26,549~complete grooves, \textbf{+184 net new vertebrate species} across 14~clades; \textbf{Figure~\ref{fig:taxonomic}}). Experimental validation data across these clades in \texttt{hitlist} span proteome-flanked cellular MS elutions, direct ligands, experimental positive/negative binding assays, and pocket mutants (\textbf{Figure~\ref{fig:hitlist_map}}).

\begin{figure}[H]
  \centering
  \includegraphics[width=0.92\textwidth]{fig01.pdf}
  \caption{\textbf{Taxonomic expansion of complete MHC class~I binding grooves across 14 vertebrate clades.} Distinct vertebrate species with complete 39-aa pseudosequences increase from 47 in MHCflurry~2.3.0 (10~trained) to 227 in \texttt{mhcseqs 2.12} (231~union; $+184$~net new vertebrate species), led by bony fish ($+51$), birds ($+41$), other mammals ($+28$), reptiles ($+17$), and amphibians ($+17$).}
  \label{fig:taxonomic}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.92\textwidth]{fig02.pdf}
  \caption{\textbf{\texttt{hitlist} Class~I validation data availability by species and experimental modality.} Log-scaled peptide counts across proteome-flanked MS elutions, direct/unflanked ligands, experimental $+/-$ binding assays, and engineered pocket mutants for 10~unseen vertebrate species, 2~unseen allele/mutant cohorts in trained species, and 2~trained MS reference controls (below dashed line).}
  \label{fig:hitlist_map}
\end{figure}

% ============================================================================
% SECTION 2: ZERO-SHOT CLASS I AFFINITY & PRESENTATION
% ============================================================================
\section{Zero-Shot Class~I Affinity and Antigen Presentation Across Unseen Clades}

\subsection{Zero-Shot Presentation (+ Real Species Proteome Flanks) on MS Elutions}

MHCflurry's presentation model combines pan-allele binding affinity with a species-agnostic (human-trained) convolutional antigen-processing predictor that inspects each peptide and its upstream/downstream source-protein flanking sequences. Using cellular MS immunopeptidomics datasets in \texttt{hitlist} with verified species reference proteomes (\textit{Sarcophilus harrisii}, \textit{Canis lupus}, \textit{Macaca mulatta}), we paired mapped eluted ligands against $10{:}1$ length-matched decoys sampled from the same host proteome with symmetric real flanking sequences, benchmarking frozen \texttt{2.3.0} weights against size-matched mismatched real-allele nulls (\textbf{Figures~\ref{fig:ms_auroc}--\ref{fig:ms_ppv}} and \textbf{Table~\ref{tab:ms_pres}}).

\begin{figure}[H]
  \centering
  \includegraphics[width=0.92\textwidth]{fig03.pdf}
  \caption{\textbf{Zero-shot Presentation (+ real species proteome flanks) versus Affinity and mismatched-allele null AUROC on \texttt{hitlist} MS elutions.} AUROC (95\% bootstrap CI) against $10{:}1$ same-proteome length-matched decoys for zero-shot Tasmanian devil (\textit{Sarcophilus harrisii}, 0~training rows, 11~UniProt grooves, 15,301~mapped ligands) alongside trained Dog (\texttt{DLA-88}) and Rhesus macaque (\texttt{Mamu-B}) MS reference controls. Gray bars show the size-matched mismatched real-allele Affinity null (median and 5--95\% range). Across $\sim$160~Myr to marsupials, human-trained processing alone reaches 0.750~AUROC and lifts Presentation from 0.749 to 0.806 ($+0.057$).}
  \label{fig:ms_auroc}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.92\textwidth]{fig04.pdf}
  \caption{\textbf{Top-N enrichment gain (\texttt{PPV@N}) from human-trained antigen processing on \texttt{hitlist} MS elutions.} Precision among the top $N$ predictions (dashed line: 0.091 random baseline at $10{:}1$ decoy ratio). Adding human-trained processing and real \textit{S.~harrisii} flanks lifts Tasmanian devil \texttt{PPV@N} from 0.273 to 0.356 ($+30\%$ relative enrichment).}
  \label{fig:ms_ppv}
\end{figure}

\begin{table}[H]
\centering
\caption{\textbf{Zero-shot Presentation (+ real proteome flanks) versus Affinity and size-matched mismatched-allele null on \texttt{hitlist} MS elutions.} All evaluations use $10{:}1$ length-matched decoys with real flanking sequences drawn from the matching host proteome and frozen \texttt{2.3.0} weights.}
\label{tab:ms_pres}
\scriptsize\sffamily
\begin{tabularx}{\textwidth}{@{}X l r r r r r@{}}
\toprule
\textbf{Dataset (\texttt{hitlist} MS)} & \textbf{Status} & \textbf{Aff.} & \textbf{Null Med [5--95\%]} & \textbf{Proc. (+F)} & \textbf{Pres. (+F)} & \textbf{PPV@N (Aff$\rightarrow$+F)} \\
\midrule
\rowcolor{RowTint}
\textbf{Tas. devil \texttt{Saha}} (11 gr., 15k) & \textbf{0-shot Marsup.} & 0.749 & 0.695 [0.674, 0.718] & \textbf{0.750} & \textbf{0.806} ($+0.057$) & \textbf{0.273$\rightarrow$0.356} ($+30\%$) \\
Dog \texttt{DLA-88*501:01} (1,301) & Trained ref$^\dagger$ & 0.933 & 0.794 [0.703, 0.860] & 0.837 & \textbf{0.947} ($+0.014$) & 0.711$\rightarrow$0.746 \\
\rowcolor{RowTint}
Dog \texttt{DLA-88*034:01} (355) & Trained ref & 0.959 & 0.831 [0.683, 0.916] & 0.875 & \textbf{0.970} ($+0.011$) & 0.752$\rightarrow$0.780 \\
Rhesus \texttt{Mamu-B*008:01} (874) & Trained ref & 0.968 & 0.721 [0.476, 0.942] & 0.866 & \textbf{0.979} ($+0.011$) & 0.852$\rightarrow$0.874 \\
\bottomrule
\end{tabularx}
\vspace{2pt}
{\raggedright\scriptsize\sffamily\color{Slate}$^\dagger$As established by our mandatory three-table audit in Section~4, IEDB \texttt{DLA-88*501:01} peptides overlap \texttt{2.3.0}'s bundled affinity training table; these rows serve strictly as trained positive-control ceilings, not held-out canine benchmarks.\par}
\end{table}

\subsection{Experimental Positive/Negative Binding Assays and 15-Cohort Zero-Shot Ranking}

Non-MS datasets in \texttt{hitlist}—including in-vitro refolding assays, purified-MHC fluorescence screens, synthetic random-library elution panels, and viral/mutant binding assays—lack cellular proteasome/TAP processing and have minimal host-proteome mapping ($19/448 = 4.2\%$ mapped in \texttt{binding.parquet}; $0\%$ for the 5,607 synthetic random-library \texttt{Trvu-UB*01:01} ligands). For these cohorts, \textbf{Affinity} is the primary metric, evaluated both on experimental positive-versus-negative binding assays with no synthetic decoys (\textbf{Figure~\ref{fig:exp_bind}}) and on $10{:}1$ length-matched background ligand ranking (\textbf{Figure~\ref{fig:lig_rank}} and \textbf{Table~\ref{tab:all15}}), benchmarked against a \textbf{50-allele mismatched real-allele null distribution} (stratified across \texttt{HLA-A/B/C} and non-human Class~I alleles, excluding the target species).

\begin{figure}[H]
  \centering
  \includegraphics[width=0.92\textwidth]{fig05.pdf}
  \caption{\textbf{Zero-shot performance on experimental positive-versus-negative binding assays (no synthetic decoys) versus 50-allele mismatched null.} AUROC (95\% bootstrap CI) on measured binders versus measured non-binders from \texttt{hitlist/binding.parquet}, colored by clade, alongside the 50-allele mismatched real-allele null (black diamond = median; horizontal bar = 5--95\% range).}
  \label{fig:exp_bind}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.96\textwidth]{fig06.pdf}
  \caption{\textbf{Zero-shot Affinity and no-flank Presentation ligand ranking across 15 \texttt{hitlist} unseen-species and pocket-mutant cohorts.} Solid bars show zero-shot Affinity AUROC (95\% CI); hatched bars show no-flank Presentation AUROC; black diamonds and whiskers mark the 50-allele mismatched real-allele null (median and 5--95\% range; size-matched 11-allele draws for Tasmanian devil \texttt{Saha}). Asterisks ($*$) indicate alleles carrying internal groove insertions or deletions flagged by \texttt{mhcseqs}.}
  \label{fig:lig_rank}
\end{figure}

\begin{table}[H]
\centering
\caption{\textbf{Complete zero-shot evaluation with proper negative controls across 15 \texttt{hitlist} unseen-species and binding-pocket mutant cohorts.} Every target allele has 0~training rows in \texttt{2.3.0}. Mismatched null reports median [5--95\%] across 50~diverse real Class~I alleles (excluding target species; size-matched 11-allele draws for \texttt{Saha}), with empirical one-sided $p$-value. Real vs.\ Scrambled compares real ligands against $10\times$ within-peptide residue-shuffled copies (target vs.\ mismatched null median).}
\label{tab:all15}
\scriptsize\sffamily
\begin{tabularx}{\textwidth}{@{}X r r r r r@{}}
\toprule
\textbf{Unseen Species / Allele (Clade)} & \textbf{Aff / Pres} & \textbf{50-Allele Null [5--95\%]} & \textbf{Beats ($p$)} & \textbf{Col-Perm} & \textbf{Scrambled (Null)} \\
\midrule
\rowcolor{RowTint}
\textbf{Cat \texttt{FLA-E*018:01}} ($n=24$) & \textbf{0.977} / 0.966 & 0.663 [0.283, 0.978] & 46/50 (0.098) & 0.662 & \textbf{0.957} (0.703) \\
\textbf{Cat \texttt{FLA-E*018:01 W167S}} ($n=12$) & \textbf{0.976} / \textbf{0.981} & 0.663 [0.325, 0.987] & 45/50 (0.118) & 0.677 & \textbf{0.957} (0.664) \\
\rowcolor{RowTint}
\textbf{Horse \texttt{Eqca-N*006:02}} ($n=6$) & \textbf{0.967} / 0.956 & 0.833 [0.570, 0.988] & 41/50 (0.196) & 0.876 & \textbf{0.919} (0.821) \\
\textbf{Pig \texttt{SLA-1*04:01 R156A}} ($n=141$) & \textbf{0.951} / 0.946 & 0.804 [0.534, 0.930] & \textbf{50/50 (0.020)} & 0.807 & \textbf{0.890} (0.688) \\
\rowcolor{RowTint}
\textbf{Duck \texttt{Anpl-UAA*01*}} ($n=31$) & \textbf{0.938} / 0.920 & 0.862 [0.705, 0.926] & \textbf{50/50 (0.020)} & 0.900 & \textbf{0.876} (0.804) \\
\textbf{Pig \texttt{SLA-1*13:01 F99Y}} ($n=60$) & \textbf{0.898} / 0.892 & 0.780 [0.477, 0.886] & \textbf{49/50 (0.039)} & 0.740 & \textbf{0.852} (0.706) \\
\rowcolor{RowTint}
\textbf{Duck \texttt{Anpl-UAA*76*}} ($n=30$) & \textbf{0.893} / 0.856 & 0.636 [0.365, 0.901] & 46/50 (0.098) & 0.637 & \textbf{0.848} (0.620) \\
\textbf{Giant panda \texttt{Aime-128}} ($n=21$) & \textbf{0.883} / \textbf{0.906} & 0.819 [0.746, 0.911] & 37/50 (0.275) & 0.869 & \textbf{0.866} (0.769) \\
\rowcolor{RowTint}
\textbf{Possum \texttt{Trvu-UB*01:01*}} ($n=5{,}607$) & \textbf{0.869} / 0.849 & 0.715 [0.530, 0.838] & \textbf{49/50 (0.039)} & 0.725 & \textbf{0.754} (0.542) \\
\textbf{Cynomolgus \texttt{Mafa-A1*063}} ($n=54$) & \textbf{0.868} / 0.830 & 0.599 [0.403, 0.842] & 48/50 (0.059) & 0.645 & \textbf{0.763} (0.536) \\
\rowcolor{RowTint}
\textbf{Clawed frog \texttt{Xela-UAAg*}} ($n=26$) & 0.826 / 0.828 & 0.808 [0.742, 0.873] & 36/50 (0.294) & 0.832 & 0.812 (0.769) \\
\textbf{Rabbit \texttt{RLA-A*01}} ($n=11$) & 0.787 / 0.792 & 0.876 [0.708, 0.954] & 8/50 (0.843) & 0.905 & 0.734 (0.819) \\
\rowcolor{RowTint}
\textbf{Tas. devil \texttt{Saha*}} (11 gr., $n=23{,}973$) & \textbf{0.772} / \textbf{0.806}$^\ddagger$ & 0.725 [0.703, 0.745] & \textbf{50/50 (0.020)} & 0.732 & 0.676 (0.668) \\
\textbf{Norway rat \texttt{RT1-A*a}} ($n=5$) & 0.720 / \textbf{0.940} & 0.582 [0.337, 0.888] & 36/50 (0.294) & 0.522 & 0.752 (0.548) \\
\rowcolor{RowTint}
\textbf{Bat \texttt{Ptal-N*01:01*}} (3-aa ins, $n=37$) & 0.593 / 0.717 & 0.568 [0.483, 0.696] & 31/50 (0.392) & 0.597 & 0.540 (0.529) \\
\bottomrule
\end{tabularx}
\vspace{2pt}
{\raggedright\scriptsize\sffamily\color{Slate}$^\ddagger$0.803 noF; 0.806 with real \textit{S.~harrisii} flanks. Separate IEDB feline MS cohorts \texttt{FLA-E*003:01} ($n=1{,}612$) and \texttt{FLA-K*007:01} ($n=491$) achieve 0.830 and 0.880 AUROC zero-shot (see Figure~\ref{fig:motifs}).\par}
\end{table}

% ============================================================================
% SECTION 3: GROOVE SPECIFICITY, SCRAMBLED CONTROLS, MOTIFS & PARITY
% ============================================================================
\newpage
\section{Groove Specificity, Positional Anchor Controls, Motifs, and Training Parity}

\subsection{Mismatched-Allele Nulls and Within-Peptide Scrambled Controls}

A single fixed human control allele (such as \texttt{HLA-A*02:01}) is noisy because it represents one specific hydrophobic P2/P9 groove: alleles with overlapping motifs look artificially weak against it, while alleles with anti-correlated motifs look artificially strong. Benchmarking each unseen cohort against \textbf{two complementary null controls} resolves this cleanly (\textbf{Figures~\ref{fig:ctrl}--\ref{fig:scrambled}}):
\begin{enumerate}
  \item \textbf{Mismatched real-allele \& column-permuted groove nulls (\textbf{Figure~\ref{fig:ctrl}}):} Fourteen of fifteen cohorts exceed the 50-allele mismatched median, and five large/clean cohorts achieve empirical $p \le 0.05$ (49--50/50 unrelated real alleles beaten): Pig \texttt{SLA-1*04:01 R156A} ($p=0.020$), Duck \texttt{Anpl-UAA*01} ($p=0.020$), Tasmanian devil \texttt{Saha} ($p=0.020$ against size-matched 11-allele draws), Brushtail possum \texttt{Trvu-UB*01:01} ($p=0.039$), and Pig \texttt{SLA-1*13:01 F99Y} ($p=0.039$), with Cynomolgus \texttt{Mafa-A1*063:01} at 48/50 ($p=0.059$, and $p=0.020$ on experimental $+/-$ binding). Column-permuted synthetic grooves ($N=30$, open squares) closely track the real-allele null.
  \item \textbf{Within-peptide residue-scrambled positive controls (\textbf{Figure~\ref{fig:scrambled}}):} Comparing real ligands against $10\times$ residue-shuffled copies of those exact ligands holds length and amino-acid composition strictly constant. Strong positive lifts over mismatched alleles confirm true positional anchor decoding on Cat \texttt{FLA-E*018:01 W167S} ($0.957$ vs.\ $0.664$, $+0.293$), Cat \texttt{FLA-E*018:01} ($0.957$ vs.\ $0.703$, $+0.254$), Duck \texttt{Anpl-UAA*76} ($0.848$ vs.\ $0.620$, $+0.228$), Cynomolgus \texttt{Mafa-A1*063:01} ($0.763$ vs.\ $0.536$, $+0.227$), Brushtail possum \texttt{Trvu-UB*01:01} ($0.754$ vs.\ $0.542$, $+0.212$), and Pig \texttt{SLA-1*04:01 R156A} ($0.890$ vs.\ $0.688$, $+0.202$).
  \item \textbf{Honest caveats exposed:} Rabbit \texttt{RLA-A*01} sits below the 50-allele null ($0.787$ vs.\ $0.876$, 8/50; scrambled $0.734$ vs.\ $0.819$), African clawed frog \texttt{Xela-UAAg} ($\sim$350~Myr, 2~indels) sits inside the null ($0.826$ vs.\ $0.808$, $p=0.294$), and Bat \texttt{Ptal-N*01:01} (3-aa \texttt{MDL} pocket insertion) collapses to $0.540$ vs.\ $0.529$ null on scrambled peptides.
\end{enumerate}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.96\textwidth]{fig07.pdf}
  \caption{\textbf{Groove-specific zero-shot transfer versus 50-allele mismatched real-allele and 30 column-permuted pseudosequence nulls.} Colored circles show target-allele zero-shot AUROC; gray bars and black diamonds mark the 50-allele mismatched real-allele null (5--95\% band and median); open squares mark the median of 30 column-permuted synthetic grooves. Asterisks ($*$) denote empirical $p \le 0.05$.}
  \label{fig:ctrl}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.96\textwidth]{fig08.pdf}
  \caption{\textbf{Positional anchor specificity: real ligands versus $10\times$ within-peptide residue-scrambled copies.} Colored bars show target-groove AUROC (95\% bootstrap CI) when separating real ligands from their own composition-matched shuffled copies; black diamonds show the mismatched real-allele median on the exact same task, sorted by lift $\Delta$.}
  \label{fig:scrambled}
\end{figure}

\subsection{Empirical vs.\ Zero-Shot Sequence Logos and Insertion Failure Mode}

Comparing empirical 9-mer MS ligands against the top 1\% of 30,000 random 9-mers scored zero-shot (\textbf{Figure~\ref{fig:motifs}}) explains both why transfer succeeds and where fixed 39-aa pseudosequences hit their physical ceiling:
\begin{itemize}
  \item \textbf{Feline \& Marsupial anchor recovery:} With zero feline training data, \texttt{FLA-E*003:01} recovers both its empirical P2-Pro and P9-Arg/Lys anchor motif ($n=1{,}610$ 9-mers). Across $\sim$160~Myr to marsupials, Brushtail possum \texttt{Trvu-UB*01:01} ($n=5{,}597$) and Tasmanian devil \texttt{Saha} recover aromatic/hydrophobic P9 anchors.
  \item \textbf{Single-residue pocket-mutant sensitivity:} Extracting mature-numbered point mutations via \texttt{mhcseqs} improves AUROC over the corresponding wild-type groove on mutant-specific ligands for both Pig \texttt{SLA-1*04:01 R156A} ($\Delta\text{AUROC}=+0.010$, 95\% paired CI $[+0.007,+0.013]$) and Cat \texttt{FLA-E*018:01 W167S} ($\Delta\text{AUROC}=+0.006$, 95\% paired CI $[+0.000,+0.015]$).
  \item \textbf{Honest failure mode — Bat \texttt{Ptal-N*01:01} (3-aa \texttt{MDL} pocket insertion):} Black flying fox \texttt{Ptal-N*01:01} carries three extra amino acids (\texttt{MDL}) inserted at position~51 inside the N-terminal binding pocket (driving empirical P1-Asp and P9-Pro specificity) alongside \texttt{Y171H}. Because a fixed 39-aa vector must omit those three inserted residues, predicted affinities flatten to $\sim$26--30~$\mu$M ($1.1\times$ separation; $0.540$ on real vs.\ scrambled ligands). \texttt{mhcseqs $\ge$ 2.9} automatically detects and warns on every such insertion.
\end{itemize}

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{fig09.pdf}
  \caption{\textbf{Empirical versus zero-shot predicted 9-mer binding motifs.} Observed 9-mer ligands (top row) compared against the top 1\% zero-shot predicted 9-mers out of 30,000 random background peptides (bottom row) for Brushtail possum \texttt{Trvu-UB*01:01}, Cat \texttt{FLA-E*003:01}, Tasmanian devil \texttt{Saha}, and the Bat \texttt{Ptal-N*01:01} 3-aa insertion failure mode.}
  \label{fig:motifs}
\end{figure}

\subsection{Controlled A/B Pan-Allele Training Parity (\texttt{--lock-existing})}

In a controlled A/B pan-allele training run ($[1024,1024]$ architecture, 30,760~train / 7,690~held-out test across all 307~trained alleles, seed~42; \textbf{Figure~\ref{fig:parity}} and \textbf{Table~\ref{tab:parity}}), \texttt{--lock-existing} reproduced legacy \texttt{clustalo} training bit-for-bit while adding all $+184$ net new vertebrate species.

\begin{figure}[H]
  \centering
  \includegraphics[width=0.84\textwidth]{fig10.pdf}
  \caption{\textbf{Controlled A/B pan-allele training parity across all 307 trained \texttt{2.3.0} alleles.} Locking existing trained alleles verbatim (Arm~B: \texttt{mhcseqs + --lock-existing}) reproduces legacy \texttt{clustalo} training (Arm~A) bit-for-bit (Max $|\Delta|=0.0$, $\log_{10} r=1.000000$, $\text{AUROC}=0.80169$) while expanding the table by $+184$ vertebrate species.}
  \label{fig:parity}
\end{figure}

\begin{table}[H]
\centering
\caption{\textbf{Controlled A/B pan-allele training parity validating the \texttt{GENERATE.sh --lock-existing} migration.}}
\label{tab:parity}
\footnotesize\sffamily
\begin{tabularx}{\textwidth}{@{}l X r r r r@{}}
\toprule
\textbf{Arm} & \textbf{Pseudosequence Table} & \textbf{Held-out AUROC} & \textbf{Chicken \texttt{Gaga}} & \textbf{Max $|\Delta|$ vs A} & \textbf{$\log_{10}\, r$ vs A} \\
\midrule
\rowcolor{RowTint}
\textbf{Arm A} & Legacy \texttt{2.3.0} \texttt{clustalo} & 0.80169 & 0.57025 & 0.0 & 1.000000 \\
\textbf{Arm B} & \texttt{mhcseqs} + \texttt{--lock-existing} & \textbf{0.80169} & \textbf{0.57025} & \textbf{0.0} & \textbf{1.000000} \\
\rowcolor{RowTint}
\textbf{Arm C} & \texttt{mhcseqs} (\texttt{prefer\_existing}, unlocking 7 \texttt{Gaga-BF}) & 0.83151 & 0.52342 & non-zero & 0.920 \\
\bottomrule
\end{tabularx}
\end{table}

% ============================================================================
% SECTION 4: AUDITED CATTLE BOLA-I & DOG DLA-88
% ============================================================================
\section{Audited Livestock and Companion-Animal Validation: Cattle \texttt{BoLA-I} and Dog \texttt{DLA-88}}

Evaluating released pan-allele weights on refreshed public corpora carries a subtle contamination hazard: checking only \code{curated_training_data.affinity.csv.bz2} misses mass-spectrometry ligands embedded inside the actual bundled training tables of \code{models_class1_presentation/models/}—specifically \code{affinity_predictor_train_data.csv.bz2} (811,134~rows), \code{processing_predictor_with_flanks_train_data.csv.bz2} (399,392~rows), and \code{processing_predictor_no_flank_train_data.csv.bz2} (399,392~rows).

Auditing the refreshed November~2026 \texttt{hitlist} Cattle and Dog cohorts against all three tables (\textbf{Figure~\ref{fig:overlap_audit}} and \textbf{Table~\ref{tab:cattle}}) immediately separated in-sample cohorts from genuinely unseen data:
\begin{itemize}
  \item \textbf{Excluded (100\% in-sample):} All 1,959~\texttt{DLA-88*501:01} peptides (PMID~27893789), all 50~\texttt{DLA-88*508:01} peptides (PMID~29475511), and all 48~IEDB Cattle \texttt{BoLA-I} peptides from PMID~36423003 already appear verbatim in \code{affinity_predictor_train_data.csv.bz2}. This confirms issue~\texttt{\#490}: current public IEDB canine MS data cannot support held-out \texttt{DLA-88} accuracy claims.
  \item \textbf{Strictly unseen Cattle \texttt{BoLA-I} (\texttt{PXD008151}, 21,332 flanked ligands):} Stripping every exact peptide present in any of the three \texttt{2.3.0} training tables leaves 5,811 monoallelic \texttt{BoLA-6*013:01} ligands and 15,521 multi-allelic haplotype ligands mapped to the \textit{Bos taurus} UniProt proteome (51,509~proteins). Evaluated against $10\times$ unseen, length-matched bovine decoys with symmetric real 5-aa flanks (\textbf{Figure~\ref{fig:cattle_auroc}}), all three cohorts achieve \textbf{0.902--0.960 Presentation (+F) AUROC}—well above size-matched mismatched real-allele nulls (\texttt{0.691--0.752})—and \textbf{0.833--0.922} on real vs.\ residue-scrambled bovine ligands (vs.\ \texttt{0.604--0.731} null). Real bovine flanks lift AUROC on every cohort ($+0.003$ to $+0.006$) and lift multi-allelic PPV@N by $+4.1\%$ and $+4.2\%$.
  \item \textbf{Ultra-few-shot transfer (4--12 training rows):} In haplotype ablations (\textbf{Figure~\ref{fig:cattle_fewshot}}), adding \texttt{BoLA-3*002:01} (only 12~training rows in \texttt{2.3.0}) lifts the 2-allele haplotype by $+0.041$ Affinity AUROC ($+0.029$ Presentation +F) and claims 42.8\% of unseen ligands as top binder; \texttt{BoLA-4*024:01} (only 6~training rows) reaches \textbf{0.904 Presentation (+F) AUROC} alone, and \texttt{BoLA-2*025:01} (4~training rows) reaches \textbf{0.850}.
\end{itemize}

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{fig11.pdf}
  \caption{\textbf{Mandatory three-table \texttt{2.3.0} training-overlap audit for Cattle \texttt{BoLA-I} and Dog \texttt{DLA-88}.} Every \texttt{hitlist} Dog \texttt{DLA-88} MS peptide and small IEDB Cattle \texttt{BoLA-I} cohort (PMID~36423003) is 100\% in-sample and excluded from held-out claims, whereas 79.5\%--87.3\% (21,606~unique; 21,332~flanked) of \texttt{PXD008151} Cattle \texttt{BoLA-I} ligands have zero overlap with any bundled \texttt{2.3.0} training table.}
  \label{fig:overlap_audit}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.90\textwidth]{fig12.pdf}
  \caption{\textbf{Strictly unseen \texttt{PXD008151} Cattle \texttt{BoLA-I} ligand ranking (+ real \textit{Bos taurus} flanks).} Evaluated on 2,500 strictly unseen flanked ligands per cohort against 25,000 ($10\times$) unseen bovine proteome decoys with real 5-aa N/C flanks, alongside size-matched mismatched real-allele null distributions (median and 5--95\% whiskers).}
  \label{fig:cattle_auroc}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.88\textwidth]{fig13.pdf}
  \caption{\textbf{Ultra-few-shot Cattle \texttt{BoLA-I} allele ablation (4--12 training rows).} Presentation (+F) AUROC on 2,500 strictly unseen bovine ligands per haplotype comparing ultra-few-shot alleles alone, remaining haplotype alleles alone, and the full haplotype.}
  \label{fig:cattle_fewshot}
\end{figure}

\begin{table}[H]
\centering
\caption{\textbf{Three-table overlap audit and strictly unseen Cattle \texttt{BoLA-I} (\texttt{PXD008151}) benchmark.} Unseen cohorts evaluate 2,500 strictly unseen flanked positives against 25,000 ($10\times$) unseen \textit{Bos taurus} proteome decoys with real 5-aa N/C flanks, plus size-matched mismatched nulls and within-peptide scrambled controls.}
\label{tab:cattle}
\scriptsize\sffamily
\begin{tabularx}{\textwidth}{@{}X r r r r r r r@{}}
\toprule
\textbf{Cohort (Source)} & \textbf{Train Rows} & \textbf{Unseen (\%)} & \textbf{Aff.} & \textbf{Pres. (+F)} & \textbf{Null [5--95\%]} & \textbf{Scrambled (Null)} & \textbf{PPV@N (noF$\rightarrow$+F)} \\
\midrule
\rowcolor{RowTint}
\textbf{Cattle \texttt{BoLA-6*013:01}} & 126 & 5,873 (87.3\%) & 0.950 & \textbf{0.960} & 0.691 [0.505, 0.901] & \textbf{0.922} (0.721) & 0.726$\rightarrow$0.725 \\
\textbf{Cattle \texttt{2*012; 3*002}} & 64 + 12 & 8,304 (80.8\%) & 0.871 & \textbf{0.902} & 0.752 [0.653, 0.811] & \textbf{0.833} (0.731) & 0.538$\rightarrow$\textbf{0.560} \\
\rowcolor{RowTint}
\textbf{Cattle \texttt{1*023; 2*025; 4*024}} & 24+4+6 & 7,429 (79.5\%) & 0.892 & \textbf{0.918} & 0.691 [0.596, 0.784] & \textbf{0.857} (0.604) & 0.570$\rightarrow$\textbf{0.594} \\
\midrule
Cattle \texttt{BoLA} (PMID 36423003) & 21/14/13 & 0 (0.0\%) & \multicolumn{5}{c}{\textit{Excluded — 100\% in-sample in \texttt{2.3.0} bundled affinity table}} \\
\rowcolor{RowTint}
Dog \texttt{DLA-88*501:01} (27893789) & 1,963 & 0 (0.0\%) & \multicolumn{5}{c}{\textit{Excluded — 100\% in-sample (1,959/1,959 overlap; confirms \#490)}} \\
Dog \texttt{DLA-88*508:01} (29475511) & 96 & 0 (0.0\%) & \multicolumn{5}{c}{\textit{Excluded — 100\% in-sample (50/50 overlap; confirms \#490)}} \\
\bottomrule
\end{tabularx}
\end{table}

% ============================================================================
% SECTION 5: EXPERIMENTAL CLASS I -> CLASS II TRANSFER
% ============================================================================
\section{Exploratory Boundary Study: Zero-Shot Class~I \texorpdfstring{$\rightarrow$}{->} Class~II Transfer}

Because MHC Class~I (one heavy chain forming both halves of the binding pocket) and Class~II (separate $\alpha$ and $\beta$ chains each contributing one half) share the same overall 3D fold, \code{mhcseqs.align_domain_to_imgt(...)} maps all four domains onto one shared structural coordinate system. Mapping MHCflurry's 39 Class~I contact positions directly populates \textbf{36 of 39 positions} across Class~II heterodimers, with only 3 Class~I-only positions (mature 16, 19, 59) padded with \texttt{X}.

We tested whether these structurally aligned 39-aa Class~II pseudosequences allow \textbf{frozen Class~I MHCflurry \texttt{2.3.0} weights} to predict quantitative Class~II 15-mer binding affinity zero-shot across 15~HLA-DR, HLA-DQ, and HLA-DP alleles from \texttt{hitlist} (500 unique 15-mers per allele; 7,500 total IC50~nM pairs; binder threshold $\text{IC50}\le 500$~nM), benchmarked head-to-head against \texttt{NetMHCIIpan-4.3} (BA and EL), a \textbf{30-allele mismatched real Class~I null distribution}, \textbf{20 column-permuted grooves}, and \textbf{within-peptide residue-scrambled Class~II binders} (\textbf{Figures~\ref{fig:c2_auroc}--\ref{fig:c2_spec}} and \textbf{Table~\ref{tab:class2}}).

\begin{figure}[H]
  \centering
  \includegraphics[width=0.92\textwidth]{fig14.pdf}
  \caption{\textbf{Exploratory zero-shot Class~I $\rightarrow$ Class~II AUROC versus 30-allele mismatched Class~I null across 15~HLA-DR, DQ, and DP alleles.} Binary binder classification AUROC ($\text{IC50}\le 500$~nM; 500~15-mers per allele). Frozen Class~I MHCflurry with 9-mer core scanning averages 0.551—indistinguishable from the 30-allele mismatched Class~I null (0.549, gray bars with 5--95\% whiskers) and far below native \texttt{NetMHCIIpan-4.3 BA} (0.923). Direct 15-mer scoring drops to 0.235--0.255 on HLA-DP.}
  \label{fig:c2_auroc}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.92\textwidth]{fig15.pdf}
  \caption{\textbf{Quantitative Class~II 15-mer IC50 rank correlation (Spearman $\rho$) versus 30-allele mismatched Class~I null across 15 alleles.} Continuous rank correlation against experimental $-\log_{10}(\text{IC50})$. Native \texttt{NetMHCIIpan-4.3 BA} achieves mean $\rho=+0.801$, whereas frozen Class~I 9-mer core scanning reaches $+0.115$ (matching the 30-allele mismatched Class~I null at $+0.100$, below 20 column-permuted grooves at $+0.146$) and direct 15-mer scoring is systematically anti-correlated ($\rho=-0.224$, down to $-0.521$ on HLA-DP).}
  \label{fig:c2_spearman}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.78\textwidth]{fig16.pdf}
  \caption{\textbf{$15\times 15$ Class~I $\rightarrow$ Class~II cross-allele specificity matrix.} Row-centered AUROC ($\text{AUROC}-\text{row mean}$) when scoring each Class~II allele test set (rows) with every pan-class Class~II pseudosequence (columns) under 9-mer core scoring. Strong vertical column stripes rather than a diagonal ridge (mean diagonal 0.551 vs.\ off-diagonal 0.556, $\Delta=-0.005$) confirm zero Class~II allele specificity.}
  \label{fig:c2_spec}
\end{figure}

\begin{table}[H]
\centering
\caption{\textbf{Per-allele Class~I $\rightarrow$ Class~II transfer results and proper negative controls across 15~HLA-DR/DQ/DP alleles (500~15-mers per allele; 7,500~pairs total).} Scrambled Binders compares measured Class~II binders ($\text{IC50}\le 500$~nM) against $3\times$ within-peptide residue-shuffled copies of those exact 15-mers.}
\label{tab:class2}
\scriptsize\sffamily
\begin{tabularx}{\textwidth}{@{}X c r r r r r@{}}
\toprule
\textbf{Allele} & \textbf{Locus} & \textbf{NetMHCIIpan BA} & \textbf{MHCf 9-mer} & \textbf{30-Allele C1 Null [5--95\%]} & \textbf{MHCf 15-mer} & \textbf{Scrambled: MHCf / NMII} \\
\midrule
\rowcolor{RowTint}
\texttt{HLA-DRB1*01:01} & DR & 0.908 & 0.592 & 0.569 [0.496, 0.656] & 0.410 & 0.491 / \textbf{0.638} \\
\texttt{HLA-DRB1*03:01} & DR & 0.918 & 0.568 & 0.527 [0.472, 0.580] & 0.438 & 0.533 / \textbf{0.789} \\
\rowcolor{RowTint}
\texttt{HLA-DRB1*04:01} & DR & 0.859 & 0.498 & 0.524 [0.446, 0.591] & 0.406 & 0.478 / \textbf{0.774} \\
\texttt{HLA-DRB1*07:01} & DR & 0.924 & 0.617 & 0.536 [0.460, 0.671] & 0.319 & 0.526 / \textbf{0.705} \\
\rowcolor{RowTint}
\texttt{HLA-DRB1*08:02} & DR & 0.894 & 0.525 & 0.564 [0.469, 0.628] & 0.404 & 0.513 / \textbf{0.750} \\
\texttt{HLA-DRB1*09:01} & DR & 0.896 & 0.622 & 0.564 [0.508, 0.641] & 0.413 & 0.526 / \textbf{0.735} \\
\rowcolor{RowTint}
\texttt{HLA-DRB1*11:01} & DR & 0.929 & 0.484 & 0.531 [0.393, 0.647] & 0.333 & 0.453 / \textbf{0.741} \\
\texttt{HLA-DRB1*13:02} & DR & 0.920 & 0.567 & 0.537 [0.463, 0.621] & 0.409 & 0.494 / \textbf{0.706} \\
\rowcolor{RowTint}
\texttt{HLA-DRB1*15:01} & DR & 0.893 & 0.518 & 0.544 [0.463, 0.620] & 0.363 & 0.499 / \textbf{0.717} \\
\texttt{DQA1*05:01/DQB1*02:01} & DQ & 0.930 & 0.543 & 0.512 [0.355, 0.625] & 0.491 & 0.473 / \textbf{0.748} \\
\rowcolor{RowTint}
\texttt{DQA1*05:01/DQB1*03:01} & DQ & 0.945 & 0.534 & 0.530 [0.431, 0.605] & 0.650 & 0.523 / \textbf{0.655} \\
\texttt{DQA1*01:02/DQB1*06:02} & DQ & 0.940 & 0.565 & 0.556 [0.453, 0.617] & 0.532 & 0.498 / \textbf{0.734} \\
\rowcolor{RowTint}
\texttt{DPA1*01:03/DPB1*04:01} & DP & 0.961 & 0.514 & 0.571 [0.539, 0.660] & 0.250 & 0.422 / \textbf{0.731} \\
\texttt{DPA1*02:01/DPB1*01:01} & DP & 0.969 & 0.566 & 0.591 [0.509, 0.674] & 0.255 & 0.572 / \textbf{0.767} \\
\rowcolor{RowTint}
\texttt{DPA1*03:01/DPB1*04:02} & DP & 0.964 & 0.553 & 0.573 [0.521, 0.696] & 0.235 & 0.529 / \textbf{0.691} \\
\midrule
\textbf{Mean (15 alleles)} & --- & \textbf{0.923} ($\rho\,$+0.801) & \textbf{0.551} ($\rho\,$+0.115) & \textbf{0.549} ($\rho\,$+0.100)$^\ast$ & \textbf{0.394} ($\rho\,$-0.224) & \textbf{0.502} / \textbf{0.725} \\
\bottomrule
\end{tabularx}
\vspace{2pt}
{\raggedright\scriptsize\sffamily\color{Slate}$^\ast$20 column-permuted grooves average 0.571 AUROC ($\rho=+0.146$); mismatched Class~I null on scrambled binders averages 0.514.\par}
\end{table}

\begin{caveatbox}{Mechanistic Diagnosis of the Class~I $\rightarrow$ Class~II Negative Result}
\begin{enumerate}
  \item \textbf{Within-peptide scrambled binders prove 9-mer core scanning is pure hydrophobicity bias (0.502 vs.\ 0.725):} Scanning 9-mer cores lifts AUROC from $0.394$ to $0.551$ only because Class~II 15-mer binders contain hydrophobic internal cores favored by \emph{any} Class~I groove (30-allele mismatched Class~I null $=0.549$; 20 column-permuted grooves $=0.571$). Holding amino-acid composition constant by comparing real Class~II binders against their own residue-shuffled copies collapses frozen Class~I straight to \textbf{0.502} (coin flip), while native \texttt{NetMHCIIpan-4.3 BA} maintains \textbf{0.725} ($+0.223$).
  \item \textbf{Closed-groove vs.\ open-groove geometry causes severe anti-correlation on HLA-DP ($0.235$--$0.255$, $\rho=-0.40$ to $-0.52$):} Class~I binds 15-mers in a \emph{closed groove} with pinned N- and C-termini and a central bulge, whereas Class~II binds an \emph{open groove} where the 9-mer core floats internally with polar/charged flanking overhangs extending past both ends.
  \item \textbf{Zero diagonal specificity proves joint pan-class training is mandatory:} In \textbf{Figure~\ref{fig:c2_spec}}, mean on-diagonal AUROC ($0.551$) is slightly \emph{below} mean off-diagonal AUROC ($0.556$, $\Delta=-0.005$; vertical column stripes dominate). While \texttt{mhcseqs}'s shared 3D structural coordinate frame provides an ideal unified feature representation, unlocking pan-class prediction requires training a future architecture jointly on both closed-groove Class~I and open-groove Class~II binding registers.
\end{enumerate}
\end{caveatbox}

% ============================================================================
% SECTION 6: PRACTICAL GUIDANCE & REPRODUCIBILITY
% ============================================================================
\section{Practical Guidance, Ecosystem Data Hygiene, and Reproducibility}

\subsection{Checklist for Predicting on Newly Added Vertebrate Species}
\begin{enumerate}
  \item \textbf{Inspect groove alignment warnings:} Use \code{extract_class1_pseudosequence_with_alignment()} (or watch \texttt{mhcflurry} logs). Canonical grooves and small deletions transfer robustly across mammals and birds; multi-residue pocket insertions (e.g., bat \texttt{Ptal-N*01:01} 3-aa \texttt{MDL}) degrade fixed-39-aa accuracy and require retrained insertion-aware models.
  \item \textbf{Use mismatched-allele null distributions and within-peptide scrambled controls:} Rather than comparing against a single arbitrary human allele, benchmark new non-model datasets against a diverse panel of unrelated real alleles (size-matched $k$-allele draws for haplotypes) and residue-scrambled copies of the positive ligands.
  \item \textbf{Provide real host flanking sequences for MS/cellular tasks:} Real species proteome flanks consistently improve Presentation AUROC and PPV@N even across $\sim$160~Myr to marsupials ($+0.057$ AUROC on Tasmanian devil) and on audited bovine MS ($+4\%$ PPV@N).
  \item \textbf{Audit all three bundled training tables before held-out claims:} Check \code{affinity_predictor_train_data.csv.bz2} and both \code{processing_predictor_*_train_data.csv.bz2} files inside \code{models_class1_presentation/models/}—never \code{curated_training_data.affinity.csv.bz2} alone.
  \item \textbf{Do not apply frozen Class~I weights to Class~II alleles:} Use dedicated Class~II predictors (e.g., \texttt{NetMHCIIpan-4.3}) until a jointly trained pan-class MHCflurry architecture is released.
\end{enumerate}

\subsection{Upstream Nomenclature and Corpus Hygiene Resolved During 2.5.x}
Cross-species validation surfaced several ecosystem-wide issues that were filed and resolved cleanly upstream: \texttt{pirl-unc/mhcseqs\#141} (canonicalization of 4~trained alleles, fixed in \texttt{2.12.5}); \texttt{pirl-unc/mhcgnomes\#306} (blocking bare UniProt accessions such as medaka \texttt{H2LR16} from misparsing as mouse \texttt{H2-L}, fixed in \texttt{3.87.0}); \texttt{pirl-unc/mhcgnomes\#311}, \texttt{\#328}, and \texttt{mhcseqs\#150} (first-class \texttt{AccessionedAllele} identifiers preserving Class~I across 223~species); \texttt{pirl-unc/hitlist\#977} (9~misassigned UniProt proteome UPIDs); and \texttt{pirl-unc/hitlist\#989--\#993} (data-bundle organization and provenance improvements).

\subsection{Reproducible Experiment Scripts}
Every figure and table in this whitepaper is reproducible from version-controlled scripts in the MHCflurry repository:
\begin{itemize}
  \item \textbf{2.5.0 zero-shot, motifs \& pocket mutants:} \code{scripts/experiments/release_2_5_0/benchmark_cross_species_mhcseqs.py}
  \item \textbf{2.5.7--2.5.8 \texttt{hitlist} Presentation/Affinity \& training parity:} \code{scripts/experiments/release_2_5_0/benchmark_hitlist_cross_species.py} and \code{plot_hitlist_cross_species_figures.py}
  \item \textbf{2.5.9 Class~I $\rightarrow$ Class~II transfer vs.\ \texttt{NetMHCIIpan-4.3}:} \code{scripts/experiments/release_2_5_0/benchmark_pan_class_ii_transfer.py}
  \item \textbf{2.5.13 Audited Cattle \texttt{BoLA-I} \& Dog \texttt{DLA-88}:} \code{scripts/experiments/release_2_5_0/benchmark_cattle_dog_hitlist.py}
  \item \textbf{This whitepaper PDF:} \texttt{python} \code{docs/build_whitepaper_pdf.py}
\end{itemize}

% ============================================================================
% SECTION 7: REFERENCES
% ============================================================================
\section{References}
{\footnotesize
\begin{enumerate}[label={[\arabic*]}, itemsep=2pt]
  \item O'Donnell, T. J., Rubinsteyn, A., \& Laserson, U. (2020). MHCflurry 2.0: Improved pan-allele prediction of MHC class~I-presented peptides by incorporating antigen processing. \textit{Cell Systems}, 11(1), 42--48.e7. \href{https://doi.org/10.1016/j.cels.2020.06.010}{doi:10.1016/j.cels.2020.06.010}
  \item O'Donnell, T. J., Rubinsteyn, A., Bonsack, M., Riemer, A. B., Laserson, U., \& Hammerbacher, J. (2018). MHCflurry: Open-source Class~I MHC binding affinity prediction. \textit{Cell Systems}, 7(1), 129--132.e4. \href{https://doi.org/10.1016/j.cels.2018.05.014}{doi:10.1016/j.cels.2018.05.014}
  \item Nielsen, M., Connelley, T., \& Ternette, N. (2018). Improved prediction of bovine leucocyte antigens (BoLA) presented ligands by use of mass-spectrometry-determined ligand and in vitro binding data (\texttt{PXD008151}, PMID~29115832). \textit{Journal of Proteome Research}, 17(1), 559--567.
  \item Nilsson, J. B., Kaabinejadian, S., Yari, H., Kester, M. G. D., van Balen, P., Hildebrand, W. H., \& Nielsen, M. (2023). Accurate prediction of HLA class~II antigen presentation across all loci using tailored data acquisition and refined machine learning (\texttt{NetMHCIIpan-4.3}). \textit{Science Advances}, 9(47), eadj6367.
  \item Lefranc, M.-P., Pommi\'e, C., Kaas, Q., Duprat, E., Bosc, N., Guiraudou, D., Jean, C., Ruiz, M., Da Pi\'edade, I., Rouard, M., Foulquier, E., Thouvenin, V., \& Lefranc, G. (2005). IMGT unique numbering for immunoglobulin and T cell receptor constant domains and Ig superfamily C-like domains (and G-DOMAINs). \textit{Developmental \& Comparative Immunology}, 29(3), 185--203.
  \item Caldwell, A., Coleby, R., Tovar, C., et al.\ (2018). The newly-arisen Devil facial tumour disease~2 (DFT2) reveals a mechanism for the emergence of a contagious cancer (\textit{Sarcophilus harrisii} immunopeptidome). \textit{eLife}, 7, e35314.
  \item Wynne, J. W., Weyer, J. L., Shiell, B. J., et al.\ (2016). Characterization of the MHC class~I peptide repertoire of the black flying fox (\textit{Pteropus alecto}, \texttt{Ptal-N*01:01}). \textit{Journal of Immunology}, 196(11), 4445--4454.
  \item Rubinsteyn, A., et al.\ (2026). OpenVax / PIRL-UNC immunogenomics ecosystem: \texttt{mhcseqs} (\href{https://github.com/pirl-unc/mhcseqs}{github.com/pirl-unc/mhcseqs}), \texttt{mhcgnomes} (\href{https://github.com/pirl-unc/mhcgnomes}{github.com/pirl-unc/mhcgnomes}), and \texttt{hitlist} (\href{https://github.com/pirl-unc/hitlist}{github.com/pirl-unc/hitlist}).
\end{enumerate}
}

\end{document}
"""


def _require_tool(name: str, fallback_paths: tuple[str, ...] = ()) -> str:
    found = shutil.which(name)
    if found:
        return found
    for candidate in fallback_paths:
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError(f"Required tool '{name}' not found on PATH.")


def build_pdf(output_path: Path) -> None:
    rsvg = _require_tool("rsvg-convert", ("/opt/homebrew/bin/rsvg-convert",))
    xelatex = _require_tool("xelatex", ("/Library/TeX/texbin/xelatex",))

    for out_name, svg_path in FIGURES.items():
        if not svg_path.is_file():
            raise FileNotFoundError(f"Missing source figure SVG for {out_name}: {svg_path}")

    with tempfile.TemporaryDirectory(prefix="mhcflurry-whitepaper-") as tmp:
        work = Path(tmp)
        for out_name, svg_path in FIGURES.items():
            subprocess.run(
                [rsvg, "-f", "pdf", "-o", str(work / out_name), str(svg_path)],
                check=True,
            )

        tex_path = work / "whitepaper.tex"
        tex_path.write_text(TEX_SOURCE, encoding="utf-8")

        for pass_num in (1, 2):
            proc = subprocess.run(
                [
                    xelatex,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    tex_path.name,
                ],
                cwd=work,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                tail = "\n".join(proc.stdout.splitlines()[-60:])
                raise RuntimeError(
                    f"xelatex failed on pass {pass_num} (exit {proc.returncode}):\n{tail}"
                )

        built_pdf = work / "whitepaper.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_pdf, output_path)
        size_kb = output_path.stat().st_size / 1024.0
        print(f"Wrote {output_path} ({size_kb:.1f} KiB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination PDF path (default: %(default)s)",
    )
    args = parser.parse_args()
    build_pdf(args.out)


if __name__ == "__main__":
    main()
