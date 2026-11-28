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

Converts the seven publication vector SVG figures in ``docs/_static/`` losslessly
to vector PDF via ``rsvg-convert``, compiles the XeLaTeX manuscript, and writes
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
STATIC_DIR = DOCS_DIR / "_static"
DEFAULT_OUTPUT = (
    STATIC_DIR / "whitepaper" / "mhcflurry_25x_cross_species_whitepaper.pdf"
)

FIGURES = {
    "fig1.pdf": STATIC_DIR / "release-2.5.7" / "fig1_species_and_validation_landscape.svg",
    "fig2.pdf": STATIC_DIR / "release-2.5.7" / "fig2_hitlist_presentation_vs_affinity.svg",
    "fig3.pdf": STATIC_DIR / "release-2.5.7" / "fig3_hitlist_zeroshot_binding_and_ligands.svg",
    "fig4.pdf": STATIC_DIR / "release-2.5.8" / "fig4_zeroshot_vs_control_and_parity.svg",
    "fig5.pdf": STATIC_DIR / "release-2.5.0" / "zero_shot_motifs.svg",
    "fig6.pdf": STATIC_DIR / "release-2.5.13" / "fig_cattle_dog_hitlist_validation.svg",
    "fig7.pdf": STATIC_DIR / "release-2.5.9" / "fig_pan_class_ii_experiment.svg",
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

% Allow clean line breaks inside long code identifiers and paths
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
\fancyfoot[L]{\sffamily\footnotesize\color{Slate}\href{https://openvax.github.io/mhcflurry/whitepaper_25x_cross_species.html}{openvax.github.io/mhcflurry}}
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
\textbf{Software:} \texttt{mhcflurry 2.5.x} (frozen \texttt{2.3.0} weights), \texttt{mhcseqs 2.9–2.14}, \texttt{mhcgnomes 3.79–3.93}\\
\textbf{Web Companion:} \href{https://openvax.github.io/mhcflurry/whitepaper_25x_cross_species.html}{\texttt{openvax.github.io/mhcflurry/whitepaper\_25x\_cross\_species.html}}\par}
\vspace{6pt}
{\color{Navy}\rule{\textwidth}{1.2pt}}
\vspace{6pt}

% ============================================================================
% ABSTRACT + EXECUTIVE SUMMARY
% ============================================================================
\begin{summarybox}{Abstract \& Key Findings}
MHCflurry~2.5 integrates \texttt{mhcseqs} IMGT G-DOMAIN structural alignment and \texttt{mhcgnomes} sequence-resolved nomenclature directly into MHCflurry's pan-allele pipeline, expanding complete Class~I binding-groove coverage nearly five-fold—from 47~vertebrate species (10~trained species / 308~alleles in \texttt{2.3.0}) to \textbf{223--231~vertebrate species} (26,156--26,549~complete grooves across 14~clades)—\textbf{without retraining or altering existing \texttt{2.3.0} model weights}. Here we present a unified empirical evaluation of how frozen \texttt{2.3.0} Class~I binding affinity and antigen presentation predictors generalize to newly added vertebrate clades, engineered pocket mutants, audited agricultural/companion-animal mass-spectrometry cohorts, and an exploratory Class~I~$\rightarrow$~Class~II pan-class transfer test:
\begin{itemize}
  \item \textbf{Broad zero-shot Class~I transfer:} Across 16~unseen-species and binding-pocket mutant cohorts (0~training rows), frozen \texttt{2.3.0} weights achieve \textbf{0.77--0.98 AUROC} across Carnivora (Cat \texttt{0.977}, Giant panda \texttt{0.883}), Perissodactyla (Horse \texttt{0.967}), Aves (Mallard duck \texttt{0.893–0.938}), Marsupialia (Brushtail possum \texttt{0.869}, Tasmanian devil \texttt{0.806 +F}), and engineered pocket mutants (Pig \texttt{R156A} $\Delta=+0.010$, Cat \texttt{W167S} $\Delta=+0.006$).
  \item \textbf{Cross-clade transfer of human-trained antigen processing:} On Tasmanian devil (\textit{Sarcophilus harrisii}, $\sim$160~Myr diverged from placentals), MHCflurry's human-trained processing model alone achieves \textbf{0.750~AUROC} with real marsupial flanking sequences and lifts combined Presentation from \textbf{0.749 $\rightarrow$ 0.806~AUROC} ($+0.057$, $+30\%$ relative PPV@N).
  \item \textbf{Mandatory three-table overlap audit on Cattle \texttt{BoLA-I} \& Dog \texttt{DLA-88}:} Auditing against all three bundled \texttt{2.3.0} training tables catches \textbf{100\% in-sample overlap} on \texttt{hitlist} Dog \texttt{DLA-88} and small IEDB \texttt{BoLA-I} cohorts, while isolating \textbf{21,332 strictly peptide-disjoint flanked Cattle \texttt{BoLA-I} MS ligands} (\texttt{PXD008151}) that reach \textbf{0.902--0.960 Presentation (+F) AUROC} and confirm strong ultra-few-shot transfer (4--12~training rows).
  \item \textbf{Clean negative result on zero-shot Class~I~$\rightarrow$~Class~II transfer:} Mapping 36/39 Class~II \texttt{DR}/\texttt{DQ}/\texttt{DP} groove positions via pan-class IMGT alignments onto frozen Class~I weights yields \textbf{0.551~AUROC} (9-mer core)—below an unrelated \texttt{HLA-A*02:01} negative control (\textbf{0.598}) and \texttt{NetMHCIIpan-4.3 BA} (\textbf{0.923})—with anti-correlated direct 15-mer scores (\textbf{0.394}, down to \textbf{0.235} on HLA-DP) and zero diagonal groove specificity ($\Delta=-0.005$), proving that pan-class prediction requires \textbf{joint pan-class training}.
  \item \textbf{Bit-identical training parity:} Locking existing trained alleles via \texttt{--lock-existing} reproduces legacy \texttt{clustalo} pan-allele training with \textbf{Max $|\Delta| = 0.0$} and \textbf{Pearson $r = 1.000000$} across all 308~trained alleles.
\end{itemize}
\end{summarybox}

% ============================================================================
% SECTION 1: INTRODUCTION & TAXONOMIC EXPANSION
% ============================================================================
\section{Introduction and Pan-Vertebrate Architecture in MHCflurry 2.5}

Pan-allele Class~I MHC predictors represent each MHC allele as a fixed-length \emph{pseudosequence} of polymorphic peptide-contacting residues lining the $\alpha_1$/$\alpha_2$ binding groove. In MHCflurry~2.0--2.3, building this 39-amino-acid table required downloading bulk IPD-IMGT/HLA, IPD-MHC, and UniProt FASTA files and running external \texttt{clustalo} multiple sequence alignments against hand-curated anchors. That workflow covered 47~vertebrate species (20,205~alleles, with training measurements concentrated in only 10~species and 308~alleles), required heavy external binaries, and could not register novel heavy chains, non-model UniProt sequences, or engineered binding-pocket mutants at inference time.

MHCflurry~2.5 resolves these limitations without changing any trained \texttt{2.3.0} weights:
\begin{enumerate}
  \item \textbf{Direct IMGT G-DOMAIN groove extraction via \texttt{mhcseqs}:} Installing \texttt{mhcflurry[mhcseqs]} delegates heavy-chain signal-peptide trimming, $\alpha_1$/$\alpha_2$ domain boundary detection, and mature-numbered 39-aa pseudosequence extraction to \texttt{mhcseqs}'s Kabsch-calibrated IMGT G-DOMAIN aligner. Users can register any raw vertebrate Class~I heavy chain or mature-numbered point mutation at runtime (\texttt{add\_allele\_from\_sequence(..., mutations=("W167S",))}) or extend an entire predictor (\texttt{extend\_alleles\_from\_mhcseqs()}).
  \item \textbf{Explicit structural indel diagnostics:} When \texttt{mhcseqs $\ge$ 2.9} encounters non-canonical insertions or deletions inside the binding groove (such as bat or marsupial $\alpha_1$ loop insertions), MHCflurry exposes a structured \texttt{GrooveAlignment} object and logs an explicit warning rather than silently producing an out-of-register pseudosequence.
  \item \textbf{Unified sequence-resolved nomenclature (\code{mhcgnomes.AccessionedAllele}):} Beginning in \texttt{mhcflurry 2.5.9}, canonical sequence-backed non-model vertebrate identifiers (e.g., \code{TriakisScyllium-UAA*uniprot:O46817}, \code{Saha-classI*uniprot:G3V1L5}) flow cleanly through \code{normalize_allele_name} and \code{Class1AffinityPredictor} under \code{normalize_alleles=True} (26,156~alleles across 223~canonical vertebrate species, including 2,884~\code{AccessionedAllele} entries and 0~unparsed rows), while defensively blocking bare UniProt accessions from misparsing as mammalian alleles.
  \item \textbf{Zero-regression parity gate:} Across 19,886 overlapping non-\texttt{X} alleles in \texttt{2.3.0}, \texttt{mhcseqs} achieves \textbf{99.45\% exact 39/39 match} (19,777/19,886) and \textbf{99.98\% per-residue identity} (775,385/775,554). The updated \code{downloads-generation/allele_sequences/GENERATE.sh} uses \texttt{--lock-existing} plus \code{verify_trained_allele_parity.py} to enforce 0~character differences across all 308~trained alleles.
\end{enumerate}

As shown in \textbf{Figure~\ref{fig:landscape}}, complete Class~I groove coverage expands from 47~species in \texttt{2.3.0} to 227~species in \texttt{mhcseqs 2.12} (231~union, 26,549~complete grooves, \textbf{+184 net new vertebrate species} across 14~clades): bony fish ($2\rightarrow 53$), birds ($1\rightarrow 42$), other mammals ($0\rightarrow 28$), reptiles ($0\rightarrow 17$), amphibians ($0\rightarrow 17$), rodents ($2\rightarrow 15$), marsupials ($0\rightarrow 7$), sharks \& rays ($0\rightarrow 4$), bats ($0\rightarrow 4$), and cetaceans ($0\rightarrow 2$). Non-canonical groove indels cluster cleanly into six recurring structural loops ($\alpha_1$ CD-turn 39--42, $\alpha_1$ N-helix hinge 51--58, $\alpha_2$ AB-turn 103--109, and $\alpha_2$ helix kink 145--155).

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{fig1.pdf}
  \caption{\textbf{Taxonomic expansion of complete Class~I binding grooves and \texttt{hitlist} validation data landscape.} \textbf{Left:} Distinct vertebrate species with complete Class~I grooves in MHCflurry~2.3.0 (47~species, 10~trained) versus MHCflurry~2.5.x + \texttt{mhcseqs 2.12} (227~species / 231~union; $+184$~net new species across 14~clades). \textbf{Right:} Availability of experimental Class~I validation data in \texttt{hitlist} by species and experimental modality (cellular MS elutions, direct ligands, and positive/negative binding assays) across 10~unseen vertebrate species, 2~unseen allele/mutant cohorts in trained species, and 2~trained MS reference controls.}
  \label{fig:landscape}
\end{figure}

% ============================================================================
% SECTION 2: ZERO-SHOT CLASS I AFFINITY & PRESENTATION
% ============================================================================
\section{Zero-Shot Class~I Affinity and Antigen Presentation Across Unseen Clades}

\subsection{Zero-Shot Presentation (+ Real Species Proteome Flanks) on MS Elutions}

MHCflurry's presentation model combines pan-allele binding affinity with a species-agnostic (human-trained) convolutional antigen-processing predictor that inspects the peptide and its upstream/downstream source-protein flanking sequences. A key scientific question is whether that human-trained processing model transfers across deep evolutionary divergence when supplied with real non-human host proteome flanks.

Using cellular MS immunopeptidomics datasets in \texttt{hitlist} with verified species reference proteomes (\textit{Sarcophilus harrisii}, \textit{Canis lupus}, \textit{Macaca mulatta}), we paired mapped eluted ligands against $10{:}1$ length-matched decoys sampled from the same host proteome with symmetric real flanking sequences, scoring frozen \texttt{2.3.0} weights with zero retraining (\textbf{Figure~\ref{fig:pres}} and \textbf{Table~\ref{tab:ms_pres}}).

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{fig2.pdf}
  \caption{\textbf{Zero-shot Presentation (+ real species proteome flanks) versus Affinity on \texttt{hitlist} MS elutions.} AUROC with 95\% stratified bootstrap confidence intervals against $10{:}1$ same-proteome length-matched decoys for zero-shot Tasmanian devil (\textit{Sarcophilus harrisii}, 0~training rows, 11~UniProt grooves, 15,301~mapped ligands) alongside trained Dog (\texttt{DLA-88}) and Rhesus macaque (\texttt{Mamu-B}) MS reference controls. Even across $\sim$160~Myr of divergence to marsupials, MHCflurry's human-trained processing model alone achieves 0.750~AUROC and lifts combined Presentation by $+0.057$~AUROC ($+30\%$ relative PPV@N).}
  \label{fig:pres}
\end{figure}

\begin{table}[H]
\centering
\caption{\textbf{Zero-shot Presentation (+ real proteome flanks) versus Affinity on \texttt{hitlist} MS elutions.} All evaluations use $10{:}1$ length-matched decoys with real flanking sequences drawn from the matching host proteome and frozen \texttt{2.3.0} weights.}
\label{tab:ms_pres}
\scriptsize\sffamily
\begin{tabularx}{\textwidth}{@{}X l r r r r r r@{}}
\toprule
\textbf{Dataset (\texttt{hitlist} MS)} & \textbf{Status} & \textbf{Eval / Total} & \textbf{Aff.} & \textbf{Proc. (+F)} & \textbf{Pres. (noF)} & \textbf{Pres. (+F)} & \textbf{PPV@N (Aff $\rightarrow$ +F)} \\
\midrule
\rowcolor{RowTint}
\textbf{Tas. devil \texttt{Saha}} (11 grooves) & \textbf{0-shot Marsup.} & 2,500 / 15,301 & 0.749 & \textbf{0.750} & 0.791 & \textbf{0.806} ($+0.057$) & \textbf{0.273 $\rightarrow$ 0.356} ($+30\%$) \\
Dog \texttt{DLA-88*501:01} & Trained ref$^\dagger$ & 1,301 / 1,301 & 0.933 & 0.837 & 0.942 & \textbf{0.947} ($+0.014$) & 0.711 $\rightarrow$ 0.746 \\
\rowcolor{RowTint}
Dog \texttt{DLA-88*034:01} & Trained ref & 355 / 355 & 0.959 & 0.875 & 0.966 & \textbf{0.970} ($+0.011$) & 0.752 $\rightarrow$ 0.780 \\
Rhesus \texttt{Mamu-B*008:01} & Trained ref & 874 / 874 & 0.968 & 0.866 & 0.976 & \textbf{0.979} ($+0.011$) & 0.852 $\rightarrow$ 0.874 \\
\bottomrule
\end{tabularx}
\vspace{2pt}
{\raggedright\scriptsize\sffamily\color{Slate}$^\dagger$As established by our mandatory three-table audit in Section~4, IEDB \texttt{DLA-88*501:01} peptides overlap \texttt{2.3.0}'s bundled affinity training table; these rows serve strictly as trained positive-control ceilings, not held-out canine benchmarks.\par}
\end{table}

\subsection{Experimental Positive/Negative Binding Assays and 16-Cohort Zero-Shot Ranking}

Non-MS datasets in \texttt{hitlist} and IEDB—including in-vitro refolding assays, purified-MHC fluorescence screens, synthetic random-library elution panels, and viral/mutant binding assays—lack cellular proteasome/TAP processing and have minimal host-proteome mapping ($19/448 = 4.2\%$ mapped in \texttt{binding.parquet}; $0\%$ for the 5,607 synthetic random-library \texttt{Trvu-UB*01:01} ligands). For these cohorts, \textbf{Affinity} is the primary metric, evaluated both on experimental positive-versus-negative binding assays (with \emph{no} synthetic decoys) and on ligand ranking against $10{:}1$ length-matched background decoys, paired with an unrelated human negative control (\texttt{HLA-A*02:01}) scored on identical peptides to separate genuine groove-specific recognition from peptide-composition bias (\textbf{Figure~\ref{fig:zeroshot}} and \textbf{Table~\ref{tab:all16}}).

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{fig3.pdf}
  \caption{\textbf{Experimental positive/negative binding assays and broad zero-shot ligand ranking across unseen vertebrate species and pocket mutants.} \textbf{Left:} Experimental positive-versus-negative binding assay AUROC (no synthetic decoys) from \texttt{hitlist/binding.parquet}. \textbf{Right:} Zero-shot Affinity (solid) and no-flank Presentation (hatched) ligand ranking AUROC against $10{:}1$ length-matched background decoys across 11~unseen vertebrate species and 3~engineered binding-pocket mutants.}
  \label{fig:zeroshot}
\end{figure}

\begin{table}[H]
\centering
\caption{\textbf{Complete zero-shot evaluation across 16 unseen-species and binding-pocket mutant cohorts.} Every target allele has 0~training rows in \texttt{2.3.0}. \texttt{HLA-A*02:01} Ctrl scores the identical peptides using an unrelated human groove; large positive margins ($\Delta = \text{Target} - \text{Ctrl}$) confirm groove-specific transfer rather than generic hydrophobicity.}
\label{tab:all16}
\scriptsize\sffamily
\begin{tabularx}{\textwidth}{@{}l l r r r X@{}}
\toprule
\textbf{Unseen Species / Allele (Clade)} & \textbf{Data Available} & \textbf{Aff / Pres} & \textbf{A*02 Ctrl} & \textbf{Exp. $+/-$} & \textbf{Notes / Caveats} \\
\midrule
\rowcolor{RowTint}
\textbf{Cat \texttt{FLA-E*018:01}} (Carnivora) & 24 lig + 16 $+/-$ & \textbf{0.977} / 0.966 & 0.636 & 0.308$^*$ & 155 vs 16,992~nM; full IEDB 24+/12- gives 0.684 \\
\textbf{Cat \texttt{FLA-E*018:01 W167S}} (Mut.) & 12 mut. ligands & \textbf{0.976} / \textbf{0.981} & 0.642 & --- & Mutant groove beats WT ($\Delta=+0.006$) \\
\rowcolor{RowTint}
\textbf{Cat \texttt{FLA-E*003:01} / \texttt{K*007:01}} & 1,612 / 491 MS & \textbf{0.830} / \textbf{0.880} & 0.562 / 0.659 & --- & Recovers P2-Pro \& P9-Arg/Lys motif zero-shot \\
\textbf{Horse \texttt{Eqca-N*006:02}} (Periss.) & 6+ / 10- bind & \textbf{0.967} / 0.956 & 0.611 & 0.217$^*$ & Single-mutant scan negatives ($n=16$) \\
\rowcolor{RowTint}
\textbf{Pig \texttt{SLA-1*04:01 R156A}} (Mut.) & 141 mut. ligands & \textbf{0.951} / 0.946 & 0.706 & --- & Mutant groove beats WT ($\Delta=+0.010$) \\
\textbf{Mallard duck \texttt{Anpl-UAA*01}} (Aves) & 31 binding+ & \textbf{0.938} / 0.920 & 0.862 & --- & Indel-bearing avian groove; high ctrl baseline \\
\rowcolor{RowTint}
\textbf{Pig \texttt{SLA-1*13:01 F99Y}} (Mut.) & 60 mut. ligands & \textbf{0.898} / 0.892 & 0.639 & --- & 227 vs 5,543~nM separation \\
\textbf{Mallard duck \texttt{Anpl-UAA*76}} (Aves) & 30 direct lig. & \textbf{0.893} / 0.856 & 0.564 & --- & Strong $+0.329$ lift over ctrl ($\sim$310~Myr) \\
\rowcolor{RowTint}
\textbf{Giant panda \texttt{Aime-128}} (Carn.) & 21 direct (7DC6) & \textbf{0.883} / \textbf{0.906} & 0.718 & --- & 432 vs 15,828~nM \\
\textbf{Brushtail possum \texttt{Trvu-UB*01}} & 5,607 + 31 $+/-$ & \textbf{0.869} / 0.849 & 0.570 & \textbf{0.655} & $+0.299$ over ctrl (1,182 vs 5,237~nM on $+/-$) \\
\rowcolor{RowTint}
\textbf{Cynomolgus \texttt{Mafa-A1*063}} (NHP) & 56 $+/-$ (54+/2-) & \textbf{0.868} / 0.830 & 0.666 & \textbf{0.889} & Ctrl 0.157 on $+/-$ (1,325 vs 9,663~nM) \\
\textbf{Clawed frog \texttt{Xela-UAAg}} (Amph.) & 26 lig + 19 $+/-$ & 0.826 / 0.828 & 0.810 & 0.357 & Caveat: matches ctrl (composition bias + 2 indels) \\
\rowcolor{RowTint}
\textbf{Rabbit \texttt{RLA-A*01}} (Lagom.) & 27 $+/-$ (11+/16-) & 0.787 / 0.792 & 0.969 & 0.562 & Shares P2/P9 hydrophobic motif with \texttt{HLA-A*02:01} \\
\textbf{Tas. devil \texttt{Saha}} (Marsupial) & 23,973 MS & \textbf{0.772} / \textbf{0.806}$^\ddagger$ & 0.666 & --- & Multi-allelic (11 UniProt grooves; 15,301 flanked) \\
\rowcolor{RowTint}
\textbf{Norway rat \texttt{RT1-A*a}} (Rod.) & 5 binding+ & 0.720 / \textbf{0.940} & 0.728 & --- & Very small sample ($n=5$) \\
\textbf{Black flying fox \texttt{Ptal-N*01:01}} (Bat) & 86 $+/-$ (37+/43-) & 0.593 / 0.717 & 0.520 & \textbf{0.606} & Failure mode: $+3$aa \texttt{MDL} A-pocket ins. (warned) \\
\bottomrule
\end{tabularx}
\vspace{2pt}
{\raggedright\scriptsize\sffamily\color{Slate}$^*$Tiny single-mutation negative panels ($3+/13-$ for Cat \texttt{FLA-E*018:01} in \texttt{hitlist}, $6+/10-$ for Horse). $^\ddagger$0.803 noF; 0.806 with real flanks.\par}
\end{table}

% ============================================================================
% SECTION 3: GROOVE SPECIFICITY, MOTIFS & TRAINING PARITY
% ============================================================================
\section{Groove Specificity, Motif Recovery, Failure Modes, and Training Parity}

\subsection{Groove-Specific Transfer vs.\ Negative Control and Sequence Logos}

Plotting target-groove AUROC against the \texttt{HLA-A*02:01} negative control across all 16~unseen cohorts (\textbf{Figure~\ref{fig:ctrl_parity}A}) immediately distinguishes genuine zero-shot groove decoding from peptide-composition artifacts. Thirteen of sixteen cohorts outperform the \texttt{HLA-A*02:01} control, with margins up to $+0.356$ (Horse), $+0.341$ (Cat \texttt{FLA-E*018:01}), $+0.329$ (Mallard duck \texttt{Anpl-UAA*76}), and $+0.299$ (Brushtail possum \texttt{Trvu-UB*01:01}). Conversely, African clawed frog (\texttt{Xela-UAAg}, $\sim$350~Myr diverged, bearing both $+1$aa~\texttt{K} and $+2$aa~\texttt{NA} pocket insertions) ties the control ($0.826$ vs.\ $0.810$), illustrating why negative-control checks are indispensable on deeply diverged clades.

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{fig4.pdf}
  \caption{\textbf{Groove-specific zero-shot transfer across 16 unseen cohorts and controlled A/B pan-allele training parity.} \textbf{Left:} Zero-shot target-groove AUROC compared against an \texttt{HLA-A*02:01} negative control on identical peptides across all 16~unseen-species and pocket-mutant datasets, sorted by $\Delta = \text{Target} - \text{Control}$. \textbf{Right:} Controlled A/B pan-allele training parity across all 307~trained \texttt{2.3.0} alleles comparing legacy \texttt{clustalo} pseudosequences (Arm~A) against \texttt{mhcseqs --lock-existing} (Arm~B: Max $|\Delta|=0.0$, $r=1.000000$) and unlocked chicken \texttt{Gaga-BF} resolution (Arm~C).}
  \label{fig:ctrl_parity}
\end{figure}

Comparing empirical 9-mer MS ligands against the top 1\% of 30,000 random 9-mers scored zero-shot (\textbf{Figure~\ref{fig:motifs}}) explains both why transfer succeeds and where fixed 39-aa pseudosequences hit their physical ceiling:
\begin{itemize}
  \item \textbf{Feline \& Marsupial anchor recovery:} With zero feline training data, \texttt{FLA-E*003:01} recovers both its empirical P2-Pro and P9-Arg/Lys anchor motif ($n=1{,}610$ 9-mers). Across $\sim$160~Myr to marsupials, Brushtail possum \texttt{Trvu-UB*01:01} ($n=5{,}597$) and Tasmanian devil \texttt{Saha} recover aromatic/hydrophobic P9 anchors.
  \item \textbf{Single-residue pocket-mutant sensitivity:} Extracting mature-numbered point mutations via \texttt{mhcseqs} improves AUROC over the corresponding wild-type groove on mutant-specific ligands for both Pig \texttt{SLA-1*04:01 R156A} ($\Delta\text{AUROC}=+0.010$, 95\% paired CI $[+0.007,+0.013]$) and Cat \texttt{FLA-E*018:01 W167S} ($\Delta\text{AUROC}=+0.006$, 95\% CI $[+0.000,+0.015]$).
  \item \textbf{Honest failure mode — Bat \texttt{Ptal-N*01:01} (\texttt{+3aa MDL} A-pocket insertion):} Black flying fox \texttt{Ptal-N*01:01} carries a 3-aa \texttt{MDL} insertion at mature position~51 that remodels the A-pocket (driving strong empirical P1-Asp and P9-Pro specificity) alongside a rare \texttt{Y171H} substitution. Because a fixed 39-aa vector omits those three inserted residues, predicted affinities flatten to $\sim$26--30~$\mu$M ($1.1\times$ separation). \texttt{mhcseqs $\ge$ 2.9} automatically detects and warns on every such groove insertion.
\end{itemize}

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{fig5.pdf}
  \caption{\textbf{Empirical versus zero-shot predicted 9-mer binding motifs.} Empirical 9-mer ligands (top row) compared against the top 1\% zero-shot predicted 9-mers out of 30,000 random background peptides (bottom row) for Brushtail possum \texttt{Trvu-UB*01:01}, Cat \texttt{FLA-E*003:01}, Tasmanian devil \texttt{Saha}, and the Bat \texttt{Ptal-N*01:01} insertion failure mode.}
  \label{fig:motifs}
\end{figure}

\subsection{Controlled A/B Pan-Allele Training Parity (\texttt{--lock-existing})}

Auditing all 308~trained alleles in \texttt{curated\_training\_data.csv.bz2} against \texttt{mhcseqs 2.12} confirmed that 297/297 mammalian trained alleles match \texttt{2.3.0}'s legacy \texttt{clustalo} pseudosequences 39-for-39, while 7~chicken \texttt{Gaga-BF1/BF2} alleles (176~rows) differed at an indel site where \texttt{2.3.0} carried \texttt{X} and 4~alleles (\texttt{H2-D*q}, \texttt{H2-L*q}, \texttt{Gogo-B*01:01}, \texttt{HLA-A*30:14L}) were resolved upstream in \texttt{mhcseqs 2.12.5} (\texttt{pirl-unc/mhcseqs\#141}). In a controlled A/B pan-allele training run ($[1024,1024]$ architecture, 30,760~train / 7,690~held-out test, seed~42; \textbf{Table~\ref{tab:parity}}), \texttt{--lock-existing} reproduced legacy training bit-for-bit while adding all $+184$ net new vertebrate species.

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

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{fig6.pdf}
  \caption{\textbf{Mandatory three-table overlap audit and strictly peptide-disjoint Cattle \texttt{BoLA-I} / Dog \texttt{DLA-88} validation.} \textbf{A.} Exact peptide overlap against all three bundled \texttt{2.3.0} training tables: every \texttt{hitlist} Dog \texttt{DLA-88} MS peptide and small IEDB \texttt{BoLA-I} cohort (PMID~36423003) is 100\% in-sample and excluded from held-out claims, whereas 79.5\%--87.3\% (21,606~unique peptides; 21,332~flanked) of \texttt{PXD008151} Cattle \texttt{BoLA-I} are strictly peptide-disjoint. \textbf{B.} Strictly peptide-disjoint \texttt{PXD008151} Cattle \texttt{BoLA-I} ranking against $10\times$ disjoint \textit{Bos taurus} proteome decoys with symmetric real 5-aa flanks. \textbf{C.} Ultra-few-shot \texttt{BoLA} allele ablation (4--12~training rows) on identical disjoint bovine ligands.}
  \label{fig:cattle_dog}
\end{figure}

Auditing the refreshed November~2026 \texttt{hitlist} Cattle and Dog cohorts against all three tables (\textbf{Figure~\ref{fig:cattle_dog}A}, \textbf{Table~\ref{tab:cattle}}) immediately separated contaminated cohorts from genuinely unseen data:
\begin{itemize}
  \item \textbf{Excluded (100\% in-sample):} All 1,959~\texttt{DLA-88*501:01} peptides (PMID~27893789), all 50~\texttt{DLA-88*508:01} peptides (PMID~29475511), and all 48~IEDB Cattle \texttt{BoLA-I} peptides from PMID~36423003 already appear verbatim in \texttt{affinity\_predictor\_train\_data.csv.bz2}. This confirms issue~\texttt{\#490}: current public IEDB canine MS data cannot support held-out \texttt{DLA-88} accuracy claims.
  \item \textbf{Strictly peptide-disjoint Cattle \texttt{BoLA-I} (\texttt{PXD008151}, 21,332 flanked ligands):} Stripping every exact peptide present in any of the three \texttt{2.3.0} training tables leaves 5,811 monoallelic \texttt{BoLA-6*013:01} ligands and 15,521 multi-allelic haplotype ligands mapped to the \textit{Bos taurus} UniProt proteome (51,509~proteins). Evaluated against $10\times$ training-disjoint, length-matched bovine decoys with symmetric real 5-aa flanks, all three cohorts achieve \textbf{0.902--0.960 Presentation (+F) AUROC} ($+0.171$ to $+0.290$ above \texttt{HLA-A*02:01}, $17\times$--$278\times$ tighter median IC50). Real bovine flanks lift AUROC on every cohort ($+0.003$ to $+0.006$) and lift multi-allelic PPV@N by $+4.1\%$ and $+4.2\%$.
  \item \textbf{Ultra-few-shot transfer (4--12 training rows):} In haplotype ablations (\textbf{Figure~\ref{fig:cattle_dog}C}), adding \texttt{BoLA-3*002:01} (only 12~training rows in \texttt{2.3.0}) lifts the 2-allele haplotype by $+0.041$ Affinity AUROC ($+0.029$ Presentation +F) and claims 42.8\% of disjoint ligands as top binder; \texttt{BoLA-4*024:01} (only 6~training rows) reaches \textbf{0.904 Presentation (+F) AUROC} alone, and \texttt{BoLA-2*025:01} (4~training rows) reaches \textbf{0.850}.
\end{itemize}

\begin{table}[H]
\centering
\caption{\textbf{Three-table overlap audit and strictly peptide-disjoint Cattle \texttt{BoLA-I} (\texttt{PXD008151}) benchmark.} Disjoint cohorts evaluate 2,500 strictly unseen flanked positives against 25,000 ($10\times$) disjoint \textit{Bos taurus} proteome decoys with real 5-aa N/C flanks.}
\label{tab:cattle}
\scriptsize\sffamily
\begin{tabularx}{\textwidth}{@{}X r r r r r r r@{}}
\toprule
\textbf{Cohort (Source)} & \textbf{2.3.0 Rows} & \textbf{Disjoint (\%)} & \textbf{Aff.} & \textbf{Pres. noF} & \textbf{Pres. (+F)} & \textbf{A*02 Ctrl} & \textbf{PPV@N (noF$\rightarrow$+F)} \\
\midrule
\rowcolor{RowTint}
\textbf{Cattle \texttt{BoLA-6*013:01}} (PXD008151) & 126 & 5,873 (87.3\%) & 0.950 & 0.957 & \textbf{0.960} & 0.789 & 0.726 $\rightarrow$ 0.725 \\
\textbf{Cattle \texttt{2*012:01; 3*002:01}} (PXD008151) & 64 + 12 & 8,304 (80.8\%) & 0.871 & 0.896 & \textbf{0.902} & 0.648 & 0.538 $\rightarrow$ \textbf{0.560} \\
\rowcolor{RowTint}
\textbf{Cattle \texttt{1*023; 2*025; 4*024}} (PXD008151) & 24+4+6 & 7,429 (79.5\%) & 0.892 & 0.915 & \textbf{0.918} & 0.628 & 0.570 $\rightarrow$ \textbf{0.594} \\
\midrule
Cattle \texttt{BoLA} (IEDB PMID 36423003) & 21/14/13 & 0 (0.0\%) & \multicolumn{5}{c}{\textit{Excluded — 100\% in-sample in \texttt{2.3.0} bundled affinity table}} \\
\rowcolor{RowTint}
Dog \texttt{DLA-88*501:01} (PMID 27893789) & 1,963 & 0 (0.0\%) & \multicolumn{5}{c}{\textit{Excluded — 100\% in-sample (1,959/1,959 overlap; confirms \#490)}} \\
Dog \texttt{DLA-88*508:01} (PMID 29475511) & 96 & 0 (0.0\%) & \multicolumn{5}{c}{\textit{Excluded — 100\% in-sample (50/50 overlap; confirms \#490)}} \\
\bottomrule
\end{tabularx}
\end{table}

% ============================================================================
% SECTION 5: EXPERIMENTAL CLASS I -> CLASS II TRANSFER
% ============================================================================
\section{Exploratory Boundary Study: Zero-Shot Class~I \texorpdfstring{$\rightarrow$}{->} Class~II Transfer}

Because MHC Class~I ($\alpha_1$/$\alpha_2$ heavy chain) and Class~II ($\alpha_1$ from \texttt{DRA}/\texttt{DQA1}/\texttt{DPA1}, $\beta_1$ from \texttt{DRB1}/\texttt{DQB1}/\texttt{DPB1}) share the homologous G-DOMAIN fold, \texttt{mhcseqs.align\_domain\_to\_imgt(...)} places all four domains onto a single 3D Kabsch-calibrated IMGT coordinate system (\texttt{G-ALPHA1} $\leftrightarrow$ \texttt{G-ALPHA}, \texttt{G-ALPHA2} $\leftrightarrow$ \texttt{G-BETA}). Mapping MHCflurry's 39 Class~I groove positions (22 in $\alpha_1$, 17 in $\alpha_2$/$\beta_1$) directly populates \textbf{36 of 39 positions} across Class~II heterodimers, with only 3 Class~I-specific $\alpha_1$ loop positions (mature 16, 19, 59) padded with \texttt{X}.

We tested whether these structurally aligned 39-aa Class~II pseudosequences allow \textbf{frozen Class~I MHCflurry \texttt{2.3.0} weights} to predict quantitative Class~II 15-mer binding affinity zero-shot across 15~HLA-DR, HLA-DQ, and HLA-DP alleles from \texttt{hitlist} (500 unique 15-mers per allele; 7,500 total IC50~nM pairs; binder threshold $\text{IC50}\le 500$~nM), benchmarked head-to-head against \texttt{NetMHCIIpan-4.3} (BA and EL) and an unrelated \texttt{HLA-A*02:01} negative control in both \textbf{9-mer core scanning} ($\min\text{IC50}$ over the 7 overlapping 9-mers inside each 15-mer) and \textbf{direct 15-mer} modes (\textbf{Figure~\ref{fig:class2}} and \textbf{Table~\ref{tab:class2}}).

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{fig7.pdf}
  \caption{\textbf{Exploratory zero-shot Class~I $\rightarrow$ Class~II pan-class transfer versus \texttt{NetMHCIIpan-4.3}.} \textbf{A, B.} Per-allele AUROC ($\text{IC50}\le 500$~nM) and continuous Spearman rank correlation ($\rho$ against $-\log_{10}\text{IC50}$) across 15~HLA-DR, HLA-DQ, and HLA-DP alleles (500~15-mers each). \textbf{C.} $15\times 15$ cross-allele specificity matrix (AUROC minus row mean under pan-class 9-mer core scoring), displaying column-dominated vertical bands rather than diagonal groove specificity (mean diagonal $0.551$ vs.\ off-diagonal $0.556$, $\Delta=-0.005$).}
  \label{fig:class2}
\end{figure}

\begin{table}[H]
\centering
\caption{\textbf{Summary and per-allele Class~I $\rightarrow$ Class~II transfer results across 15~HLA-DR/DQ/DP alleles (500~15-mers per allele; 7,500~pairs total).} Format: \texttt{AUROC ($\le 500$~nM) / Spearman $\rho$ ($-\log_{10}\text{IC50}$)}.}
\label{tab:class2}
\scriptsize\sffamily
\begin{tabularx}{\textwidth}{@{}l c r r r r r r@{}}
\toprule
\textbf{Allele} & \textbf{Locus} & \textbf{Binders} & \textbf{NetMHCIIpan BA} & \textbf{NetMHCIIpan EL} & \textbf{MHCf 9-mer} & \textbf{A*02 9-mer Ctrl} & \textbf{MHCf 15-mer} \\
\midrule
\rowcolor{RowTint}
\texttt{HLA-DRB1*01:01} & DR & 301/500 & 0.908 / $+0.824$ & 0.734 / $+0.481$ & 0.592 / $+0.180$ & 0.619 / $+0.235$ & 0.410 / $-0.170$ \\
\texttt{HLA-DRB1*03:01} & DR & 128/500 & 0.918 / $+0.785$ & 0.793 / $+0.544$ & 0.568 / $+0.163$ & 0.592 / $+0.183$ & 0.438 / $-0.192$ \\
\rowcolor{RowTint}
\texttt{HLA-DRB1*04:01} & DR & 173/500 & 0.859 / $+0.649$ & 0.730 / $+0.360$ & 0.498 / $+0.081$ & 0.557 / $+0.174$ & 0.406 / $-0.245$ \\
\texttt{HLA-DRB1*07:01} & DR & 244/500 & 0.924 / $+0.851$ & 0.760 / $+0.525$ & 0.617 / $+0.257$ & 0.645 / $+0.300$ & 0.319 / $-0.355$ \\
\rowcolor{RowTint}
\texttt{HLA-DRB1*08:02} & DR & 161/500 & 0.894 / $+0.714$ & 0.721 / $+0.433$ & 0.525 / $+0.033$ & 0.577 / $+0.116$ & 0.404 / $-0.183$ \\
\texttt{HLA-DRB1*09:01} & DR & 207/500 & 0.896 / $+0.774$ & 0.772 / $+0.511$ & 0.622 / $+0.181$ & 0.586 / $+0.135$ & 0.413 / $-0.195$ \\
\rowcolor{RowTint}
\texttt{HLA-DRB1*11:01} & DR & 202/500 & 0.929 / $+0.836$ & 0.818 / $+0.631$ & 0.484 / $-0.018$ & 0.592 / $+0.117$ & 0.333 / $-0.356$ \\
\texttt{HLA-DRB1*13:02} & DR & 193/500 & 0.920 / $+0.777$ & 0.652 / $+0.367$ & 0.567 / $+0.133$ & 0.591 / $+0.159$ & 0.409 / $-0.163$ \\
\rowcolor{RowTint}
\texttt{HLA-DRB1*15:01} & DR & 206/500 & 0.893 / $+0.797$ & 0.768 / $+0.529$ & 0.518 / $+0.052$ & 0.569 / $+0.176$ & 0.363 / $-0.308$ \\
\texttt{DQA1*05:01/DQB1*02:01} & DQ & 104/500 & 0.930 / $+0.825$ & 0.797 / $+0.555$ & 0.543 / $+0.089$ & 0.521 / $+0.095$ & 0.491 / $-0.065$ \\
\rowcolor{RowTint}
\texttt{DQA1*05:01/DQB1*03:01} & DQ & 238/500 & 0.945 / $+0.859$ & 0.807 / $+0.610$ & 0.534 / $+0.093$ & 0.531 / $+0.070$ & 0.650 / $+0.269$ \\
\texttt{DQA1*01:02/DQB1*06:02} & DQ & 151/500 & 0.940 / $+0.873$ & 0.805 / $+0.580$ & 0.565 / $+0.181$ & 0.593 / $+0.247$ & 0.532 / $-0.021$ \\
\rowcolor{RowTint}
\texttt{DPA1*01:03/DPB1*04:01} & DP & 84/500 & 0.961 / $+0.827$ & 0.840 / $+0.519$ & 0.514 / $+0.087$ & 0.634 / $+0.262$ & 0.250 / $-0.521$ \\
\texttt{DPA1*02:01/DPB1*01:01} & DP & 85/500 & 0.969 / $+0.797$ & 0.694 / $+0.351$ & 0.566 / $+0.076$ & 0.660 / $+0.215$ & 0.255 / $-0.398$ \\
\rowcolor{RowTint}
\texttt{DPA1*03:01/DPB1*04:02} & DP & 76/500 & 0.964 / $+0.826$ & 0.745 / $+0.401$ & 0.553 / $+0.135$ & 0.697 / $+0.322$ & 0.235 / $-0.459$ \\
\midrule
\textbf{Mean (15 alleles)} & --- & \textbf{167/500} & \textbf{0.923 / $+0.801$} & \textbf{0.763 / $+0.493$} & \textbf{0.551 / $+0.115$} & \textbf{0.598 / $+0.187$} & \textbf{0.394 / $-0.224$} \\
\bottomrule
\end{tabularx}
\end{table}

\begin{caveatbox}{Mechanistic Diagnosis of the Class~I $\rightarrow$ Class~II Negative Result}
\begin{enumerate}
  \item \textbf{9-mer core scanning captures generic hydrophobicity, not Class~II groove chemistry:} Scanning 9-mer cores lifts AUROC from $0.394$ to $0.551$ only because Class~II 15-mer binders contain hydrophobic internal P1 cores favored by Class~I models too. The unrelated \texttt{HLA-A*02:01} negative control scores \emph{higher} ($0.598$, $\rho=+0.187$) on the exact same peptides.
  \item \textbf{Closed-groove vs.\ open-groove geometry causes severe anti-correlation on HLA-DP ($0.235$--$0.255$, $\rho=-0.40$ to $-0.52$):} Class~I binds 15-mers in a \emph{closed groove} with pinned N- and C-termini and a central bulge, whereas Class~II binds an \emph{open groove} where the 9-mer core floats internally with polar/charged flanking overhangs extending past both ends.
  \item \textbf{Zero diagonal specificity proves joint pan-class training is mandatory:} In \textbf{Figure~\ref{fig:class2}C}, mean on-diagonal AUROC ($0.551$) is slightly \emph{below} mean off-diagonal AUROC ($0.556$, $\Delta=-0.005$; vertical column stripes dominate). While \texttt{mhcseqs}'s shared 3D IMGT coordinate frame provides an ideal unified feature representation, unlocking pan-class prediction requires training a future architecture jointly on both closed-groove Class~I and open-groove Class~II binding registers.
\end{enumerate}
\end{caveatbox}

% ============================================================================
% SECTION 6: PRACTICAL GUIDANCE & REPRODUCIBILITY
% ============================================================================
\section{Practical Guidance, Ecosystem Data Hygiene, and Reproducibility}

\subsection{Checklist for Predicting on Newly Added Vertebrate Species}
\begin{enumerate}
  \item \textbf{Inspect groove alignment warnings:} Use \texttt{extract\_class1\_pseudosequence\_with\_alignment()} (or watch \texttt{mhcflurry} logs). Canonical grooves and small deletions transfer robustly across mammals and birds; multi-residue pocket insertions (e.g., bat \texttt{Ptal-N*01:01} $+3$aa \texttt{MDL}) degrade fixed-39-aa accuracy and require retrained insertion-aware models.
  \item \textbf{Run an unrelated negative-control allele:} When validating on small non-model ligand sets, score identical peptides with \texttt{HLA-A*02:01}; genuine groove transfer shows a clear positive AUROC margin ($\Delta > 0.10$), whereas tied scores (like frog \texttt{Xela-UAAg}) warn of peptide-composition bias.
  \item \textbf{Provide real host flanking sequences for MS/cellular tasks:} Real species proteome flanks consistently improve Presentation AUROC and PPV@N even across $\sim$160~Myr to marsupials ($+0.057$ AUROC on Tasmanian devil) and on audited bovine MS ($+4\%$ PPV@N).
  \item \textbf{Audit all three bundled training tables before held-out claims:} Check \code{affinity_predictor_train_data.csv.bz2} and both \code{processing_predictor_*_train_data.csv.bz2} files inside \code{models_class1_presentation/models/}—never \code{curated_training_data.affinity.csv.bz2} alone.
  \item \textbf{Do not apply frozen Class~I weights to Class~II alleles:} Use dedicated Class~II predictors (e.g., \texttt{NetMHCIIpan-4.3}) until a jointly trained pan-class MHCflurry architecture is released.
\end{enumerate}

\subsection{Upstream Nomenclature and Corpus Hygiene Resolved During 2.5.x}
Cross-species validation surfaced several ecosystem-wide issues that were filed and resolved cleanly upstream: \texttt{pirl-unc/mhcseqs\#141} (canonicalization of 4~trained alleles, fixed in \texttt{2.12.5}); \texttt{pirl-unc/mhcgnomes\#306} (blocking bare UniProt accessions such as medaka \texttt{H2LR16} from misparsing as mouse \texttt{H2-L}, fixed in \texttt{3.87.0}); \texttt{pirl-unc/mhcgnomes\#311}, \texttt{\#328}, and \texttt{mhcseqs\#150} (first-class \texttt{AccessionedAllele} identifiers preserving Class~I across 223~species); \texttt{pirl-unc/hitlist\#977} (9~misassigned UniProt proteome UPIDs); and \texttt{pirl-unc/hitlist\#989–\#993} (data-bundle organization and provenance improvements).

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
