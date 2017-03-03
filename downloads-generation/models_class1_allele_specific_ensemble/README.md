# Class I allele-specific models (ensemble)

This download contains trained MHC Class I allele-specific MHCflurry models. For each allele, an ensemble of predictors is trained on random halves of the training data. Model architectures are selected based on performance on the other half of the dataset, so in general each ensemble contains predictors of different architectures. At prediction time the geometric mean IC50 is taken over the trained models. The training data used is in the [data_combined_iedb_kim2014](../data_combined_iedb_kim2014) MHCflurry download.

The training script supports multi-node parallel execution using the [kubeface](https://github.com/hammerlab/kubeface) library.

To use kubeface, you should make a google storage bucket and pass it below with the --storage-prefix argument. 

To generate this download we run:

```
./GENERATE.sh \
    --parallel-backend kubeface \
    --target-tasks 200 \
    --kubeface-backend kubernetes \
    --kubeface-storage gs://kubeface-tim \
    --kubeface-worker-image hammerlab/mhcflurry-misc:latest \
    --kubeface-kubernetes-task-resources-memory-mb 10000 \
    --kubeface-worker-path-prefix venv-py3/bin \
    --kubeface-max-simultaneous-tasks 200 \
```

To debug locally:
```
./GENERATE.sh \
    --parallel-backend local-threads \
    --target-tasks 1
```
