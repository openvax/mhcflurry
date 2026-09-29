# Legacy minimal allele-specific affinity models

This historical bundle contains one affinity network per supported allele.
It is small and useful for compatibility tests; use `models_class1_presentation`
for current prediction work.

To use it explicitly without changing the default weights:

```bash
mhcflurry downloads fetch models_class1_minimal
mhcflurry predict INPUT.csv --affinity-only \
  --models "$(mhcflurry downloads path models_class1_minimal)/models" \
  --out predictions.csv
```
