# Preserve seeded assignments while reducing repair edge visits

The real CPH-08-TISSUE evaluation pool spends minutes revisiting dense
augmenting-path neighborhoods (#439). Filter neighbors whose owners are
already queued using a NumPy mask, after generating the original permutations.
Retain random-number consumption, BFS order, assignments and matching policy.
Verify golden seeded replay and subsequent RNG state, existing independent
maximum-matching tests, and an empirical timed replay of the saved real pool.
No affinity scores, calipers, hit identities or trained weights change.
