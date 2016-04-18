

bdata.2009.mhci.public.1.txt
  #Original MHC-I binding data prepared in year 2009, from which the following cross-validation data sets were prepared.


Three types of cross-validation data sets used in the manuscript:
=================================================================
bdata.2009.mhci.public.1.cv_rnd.txt
  # cross-validation strategy = cv_rnd
  # Randomly partitioning of data points into 5 folds.

bdata.2009.mhci.public.1.cv_sr.txt
  # cross-validation strategy = cv_sr
  # 1) Similarity-Reduced was applied.
  # 2) Then, random partitioning of data points into 5 folds.

bdata.2009.mhci.public.1.cv_gs.txt
  # cross-validation strategy = cv_gs
  # 1) Top scoring peptides based on 'cluster_size_reduced' were removed (13 peptides removed).
  # 2) Distributed peptides among 5 folds such that there are no similar peptides shared between any two cv-partitions.
  # 3) Each peptide is assigned to the same cv-partition index across all (mhc,length) data sets.



Comparison of average percentage of peptides in cv-testing that have similar peptides in cv-training:
===================
rnd  sr  gs
 8%  1% 0%

