# Class I allele-specific models (single)

This download contains trained MHC Class I allele-specific MHCflurry models. The training data used is in the [data_combined_iedb_kim2014](../data_combined_iedb_kim2014) MHCflurry download. We first select network hyperparameters for each allele individually using cross validation over the models enumerated in [models.py](models.py). The best hyperparameter settings are selected via average of AUC (at 500nm), F1, and Kendall's Tau over the training folds. We then train the production models over the full training set using the selected hyperparameters.

The training script supports multi-node parallel execution using the [dask-distributed](https://distributed.readthedocs.io/en/latest/) library. To enable this, pass the IP and port of the dask scheduler to the training script with the '--dask-scheduler' option. The GENERATE.sh script passes all arguments to the training script so you can just give it as an argument to GENERATE.sh.

We run dask distributed on Google Container Engine using Kubernetes as described [here](https://github.com/hammerlab/dask-distributed-on-kubernetes).

To generate this download we run:

```
# If you are running dask distributed using our kubernetes config, you can use the DASK_IP one liner below.
# Otherwise, just set it to the IP of the dask scheduler.
./GENERATE.sh \
    --cv-folds-per-task 10 \
    --backend kubernetes \
    --storage-prefix gs://kubeface \
    --worker-image hammerlab/mhcflurry:latest \
    --kubernetes-task-resources-memory-mb 10000 \
    --worker-path-prefix venv-py3/bin \
    --max-simultaneous-tasks 200 \

```
