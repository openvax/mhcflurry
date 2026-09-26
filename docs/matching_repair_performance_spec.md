# Preserve seeded assignments while reducing repair edge visits

The real CPH-08-TISSUE evaluation pool spends minutes revisiting dense
augmenting-path neighborhoods (#439). Filter neighbors whose owners are
already queued using a NumPy mask, after generating the original permutations.
Retain random-number consumption, BFS order, assignments and matching policy.
Verify golden seeded replay and subsequent RNG state, existing independent
maximum-matching tests, and an empirical timed replay of the saved real pool.
No affinity scores, calipers, hit identities or trained weights change.

The regression fixture freezes 24 cases from commit `2c0736d61`, including
28 repair calls. It checks both assignments and the next eight RNG outputs.
The 159-test matching suite also retains the independent maximum-matching
oracle; exact seeded replay complements that feasibility check.

A native profile after vectorizing edge visits showed repeated object-string
comparisons and neighborhood materialization. Factorize protein identities
once and use a bounded 128-hit neighborhood cache. Missing-protein semantics,
free-candidate masks and all permutation calls remain unchanged. The cache
has a fixed entry bound rather than retaining the full bipartite graph.
