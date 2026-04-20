# Training loop: no DataLoader, every batch encoded synchronously on main thread

## Observation

In `mhcflurry/class1_neural_network.py` around line 2045, the per-epoch batch
loop does CPU data prep on the main thread right before dispatching each batch
to the GPU:

```python
for epoch in range(max_epochs):
    # per-epoch CPU work: generate random negatives, encode peptides, shuffle
    ...
    for batch_start in range(0, n_train, batch_size):
        batch_idx = train_indices[batch_start:batch_start + batch_size]
        peptide_batch = torch.from_numpy(x_peptide[batch_idx]).float().to(device)
        y_batch       = torch.from_numpy(y_encoded[batch_idx].astype(numpy.float32)).to(device)
        ...
        predictions = network(inputs)
        loss = loss_obj(predictions, y_batch, ...)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())   # forces GPU → CPU sync every batch
```

No `torch.utils.data.DataLoader` with `num_workers>0`, no pinned memory,
no `non_blocking=True` on `.to(device)`, and `.item()` per batch forces a
GPU→CPU sync that serializes kernels.

## Impact

Observed empirically on a Brev 8×A100 box running the public GENERATE.sh
recipe: per-epoch wall time 200–450 s for a 1024×512 dense model that should
be <10 s on an A100. GPU memory utilization 1–5% across all 8 GPUs, GPU util
in single digits. Both OpenMP oversubscription (separate issue) and this
synchronous batch loop contribute.

## Suggested fix

Opt-in behind a hyperparameter / env var so default reproducibility is
preserved:

```python
dataset = TensorDataset(x_peptide_tensor, x_allele_tensor, y_tensor)
loader = DataLoader(
    dataset, batch_size=self.hyperparameters["minibatch_size"], shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True,
)
for batch in loader:
    peptide_batch = batch[0].to(device, non_blocking=True)
    allele_batch  = batch[1].to(device, non_blocking=True)
    y_batch       = batch[2].to(device, non_blocking=True)
    ...
```

Plus: accumulate losses as a running tensor sum and materialize with `.item()`
once at epoch end instead of per batch.

## Complication

`random_negatives_planner.get_peptides()` is called once per epoch at
line 1995. Moving that into a DataLoader-compatible sampler requires
restructuring — not hard but more than a trivial diff.

## Expected speedup

2–5× on top of the OMP fix (see sibling issues) for this workload.
