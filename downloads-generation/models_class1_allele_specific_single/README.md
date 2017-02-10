# Class I allele-specific models (single)

This download contains trained MHC Class I allele-specific MHCflurry models. The training data used is in the [data_combined_iedb_kim2014](../data_combined_iedb_kim2014) MHCflurry download. We first select network hyperparameters for each allele individually using cross validation over the models enumerated in [models.py](models.py). The best hyperparameter settings are selected via average of AUC (at 500nm), F1, and Kendall's Tau over the training folds. We then train the production models over the full training set using the selected hyperparameters.

The training script supports multi-node parallel execution using the [kubeface](https://github.com/hammerlab/kubeface) librarie.

To use kubeface, you should make a google storage bucket and pass it below with the --storage-prefix argument. 

To generate this download we run:

```
./GENERATE.sh \
    --cv-folds-per-task 10 \
    --backend kubernetes \
    --storage-prefix gs://kubeface \
    --worker-image hammerlab/mhcflurry:latest \
    --kubernetes-task-resources-memory-mb 10000 \
    --worker-path-prefix venv-py3/bin \
    --max-simultaneous-tasks 200 \

```
