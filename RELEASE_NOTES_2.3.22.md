# MHCflurry 2.3.22

Test-only fix. `test_calibrate_fast_parity_with_legacy_path[compact]` failed
on the Python 3.13 CI leg.

- Two tests call `worker_init` in the test process. It reseeds numpy's and
  Python's global random generators from OS entropy, as each pool worker
  needs, so every later test drew different random numbers on each run. A
  new `preserve_global_random_state` fixture restores both generators after
  those tests. The parity test also draws its peptides from its own seeded
  generator.
- For compact calibration, the parity test asserted that the legacy and fast
  curves agree within 0.1 percentile at 100 probe affinities. The greedy knot
  selection is discontinuous in its input: in 200 refits of real predictions
  perturbed by 1e-5 relative noise, the mapped percentile moved by up to 0.48.
  CI showed 0.16 and 0.22. The test now checks two things: the IC50
  predictions that the two paths fit agree to `rtol=1e-5`, and the fast curve
  fits those predictions' empirical CDF as well as the legacy curve does,
  within two peptides (0.1 percentile). In the same 200 refits, that error
  moved by at most 0.053.

The prediction models and numerical prediction behavior are unchanged.
