# Training loop: .to(device) is blocking + no pinned memory

## Observation

In `mhcflurry/class1_neural_network.py` around line 2048:

```python
peptide_batch = torch.from_numpy(x_peptide[batch_idx]).float().to(device)
y_batch       = torch.from_numpy(y_encoded[batch_idx].astype(numpy.float32)).to(device)
...
if x_allele is not None:
    allele_batch = torch.from_numpy(x_allele[batch_idx]).float().to(device)
```

Each `.to(device)` call is synchronous by default and the source numpy
arrays aren't pinned, so every transfer stalls the main thread for the
duration of the PCIe copy.

## Suggested fix

Pin the source tensors once, then pass `non_blocking=True` on each
`.to(device)`:

```python
# one-time at start of fit():
x_peptide_t = torch.from_numpy(x_peptide).float().pin_memory()
y_t         = torch.from_numpy(y_encoded.astype(numpy.float32)).pin_memory()

# per batch:
peptide_batch = x_peptide_t[batch_idx].to(device, non_blocking=True)
y_batch       = y_t[batch_idx].to(device, non_blocking=True)
```

Prerequisite for effective DataLoader-based async data loading (sibling
issue `01-training-no-dataloader.md`).
