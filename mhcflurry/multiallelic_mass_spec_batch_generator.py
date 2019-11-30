import collections

import numpy
import pandas


from .hyperparameters import HyperparameterDefaults


class BatchPlan(object):
    def __init__(self, equivalence_classes, batch_compositions):
        # batch_compositions is (num batches_generator, batch size)

        self.equivalence_classes = equivalence_classes # indices into points
        self.batch_compositions = batch_compositions # indices into equivalence_classes
        indices_into_equivalence_classes = []
        next_index = collections.defaultdict(int)
        for batch_composition in batch_compositions:
            indices = []
            for equivalence_class in batch_composition:
                indices.append(next_index[equivalence_class])
                next_index[equivalence_class] += 1
            indices_into_equivalence_classes.append(
                numpy.array(indices, dtype=int))
        self.indices_into_equivalence_classes = indices_into_equivalence_classes

    def batch_indices_generator(self, epochs=1):
        batch_nums = numpy.arange(len(self.batch_compositions))
        for epoch in range(epochs):
            # Shuffle equivalence classes
            for arr in self.equivalence_classes:
                numpy.random.shuffle(arr)
            numpy.random.shuffle(batch_nums)
            for batch_num in batch_nums:
                class_indices = self.batch_compositions[batch_num]
                indices_into_classes = self.indices_into_equivalence_classes[
                    batch_num
                ]
                batch_indices = [
                    self.equivalence_classes[i][j]
                    for (i, j) in zip(class_indices, indices_into_classes)
                ]
                yield batch_indices

    def batches_generator(self, x_dict, y_list, epochs=1):
        for indices in self.batch_indices_generator(epochs=epochs):
            batch_x_dict = {}
            for (item, value) in x_dict.items():
                batch_x_dict[item] = value[indices]
            batch_y_list = []
            for value in y_list:
                batch_y_list.append(value[indices])
            yield (batch_x_dict, batch_y_list)

    def summary(self, indent=0):
        lines = []
        lines.append("Equivalence class sizes: ")
        lines.append(pandas.Series(
            [len(c) for c in self.equivalence_classes]))
        lines.append("Batch compositions: ")
        lines.append(self.batch_compositions)
        indent_spaces = "    " * indent
        return "\n".join([indent_spaces + str(line) for line in lines])

    @property
    def num_batches(self):
        return self.batch_compositions.shape[0]

    @property
    def batch_size(self):
        return self.batch_compositions.shape[1]


class MultiallelicMassSpecBatchGenerator(object):
    hyperparameter_defaults = HyperparameterDefaults(
        batch_generator_validation_split=0.1,
        batch_generator_batch_size=128,
        batch_generator_affinity_fraction=0.5)
    """
    Hyperperameters for batch generation for the ligandome predictor.
    """

    def __init__(self, hyperparameters):
        self.hyperparameters = self.hyperparameter_defaults.with_defaults(
            hyperparameters)
        self.equivalence_classes = None
        self.batch_indices = None

    @staticmethod
    def plan_from_dataframe(df, hyperparameters):
        affinity_fraction = hyperparameters["batch_generator_affinity_fraction"]
        batch_size = hyperparameters["batch_generator_batch_size"]
        classes = {}
        df["equivalence_class"] = [
            classes.setdefault(
                tuple(row[["is_affinity", "is_binder", "experiment_name"]]),
                len(classes))
            for _, row in df.iterrows()
        ]
        df["first_allele"] = df.alleles.str.get(0)
        df["unused"] = True
        df["idx"] = df.index
        df = df.sample(frac=1.0)
        #df["key"] = df.is_binder ^ (numpy.arange(len(df)) % 2).astype(bool)
        #df = df.sort_values("key")
        #del df["key"]

        affinities_per_batch = int(affinity_fraction * batch_size)

        # First do mixed affinity / multiallelic ms batches_generator.
        batch_compositions = []
        for experiment in df.loc[~df.is_affinity].experiment_name.unique():
            if experiment is None:
                continue
            while True:
                experiment_df = df.loc[
                    df.unused & (df.experiment_name == experiment)]
                if len(experiment_df) == 0:
                    break
                (experiment_alleles,) = experiment_df.alleles.unique()
                affinities_df = df.loc[df.unused & df.is_affinity].copy()
                affinities_df["matches_allele"] = (
                    affinities_df.first_allele.isin(experiment_alleles))

                # Whenever possible we try to use affinities with the same
                # alleles as the mass spec experiment
                affinities_df = affinities_df.sort_values(
                    "matches_allele", ascending=False)

                affinities_for_this_batch = min(
                    affinities_per_batch, len(affinities_df))
                mass_spec_for_this_batch = (
                    batch_size - affinities_for_this_batch)
                if len(experiment_df) < mass_spec_for_this_batch:
                    mass_spec_for_this_batch = len(experiment_df)
                    affinities_for_this_batch = (
                            batch_size - mass_spec_for_this_batch)
                    if affinities_for_this_batch < len(affinities_df):
                        # For mass spec, we only do whole batches_generator, since it's
                        # unclear how our pairwise loss would interact with
                        # a smaller batch.
                        break

                to_use_list = []

                # sample mass spec
                to_use = experiment_df.head(mass_spec_for_this_batch)
                to_use_list.append(to_use.index.values)

                # sample affinities
                to_use = affinities_df.head(affinities_for_this_batch)
                to_use_list.append(to_use.index.values)

                to_use_indices = numpy.concatenate(to_use_list)
                df.loc[to_use_indices, "unused"] = False
                batch_compositions.append(
                    df.loc[to_use_indices].equivalence_class.values)

        # Affinities-only batches
        affinities_df = df.loc[df.unused & df.is_affinity]
        while len(affinities_df) > 0:
            to_use = affinities_df.head(batch_size)
            df.loc[to_use.index, "unused"] = False
            batch_compositions.append(to_use.equivalence_class.values)
            affinities_df = df.loc[df.unused & df.is_affinity]

        class_to_indices = df.groupby("equivalence_class").idx.unique()
        equivalence_classes = [
            class_to_indices[i]
            for i in range(len(class_to_indices))
        ]
        return BatchPlan(
            equivalence_classes=equivalence_classes,
            batch_compositions=batch_compositions)

    def plan(
            self,
            affinities_mask,
            experiment_names,
            alleles_matrix,
            is_binder):
        affinities_mask = numpy.array(affinities_mask, copy=False, dtype=bool)
        experiment_names = numpy.array(experiment_names, copy=False)
        alleles_matrix = numpy.array(alleles_matrix, copy=False)
        is_binder = numpy.array(is_binder, copy=False, dtype=bool)
        n = len(experiment_names)

        numpy.testing.assert_equal(len(affinities_mask), n)
        numpy.testing.assert_equal(len(alleles_matrix), n)
        numpy.testing.assert_equal(len(is_binder), n)
        numpy.testing.assert_equal(
            affinities_mask, pandas.isnull(experiment_names))

        validation_items = numpy.random.choice(
            n, int(
                self.hyperparameters['batch_generator_validation_split'] * n))
        validation_mask = numpy.zeros(n, dtype=bool)
        validation_mask[validation_items] = True

        df = pandas.DataFrame({
            "is_affinity": affinities_mask,
            "experiment_name": experiment_names,
            "is_binder": is_binder,
            "is_validation": validation_mask,
            "alleles": [tuple(row[row != None]) for row in alleles_matrix],
        })
        df.loc[df.is_affinity, "experiment_name"] = None

        train_df = df.loc[~df.is_validation].copy()
        test_df = df.loc[df.is_validation].copy()

        self.train_batch_plan = self.plan_from_dataframe(
            train_df, self.hyperparameters)
        self.test_batch_plan = self.plan_from_dataframe(
            test_df, self.hyperparameters)

    def summary(self):
        return (
            "Train: " + self.train_batch_plan.summary(indent=1) +
            "\n***\nTest: " + self.test_batch_plan.summary(indent=1))

    def get_train_and_test_generators(self, x_dict, y_list, epochs=1):
        train_generator = self.train_batch_plan.batches_generator(
            x_dict, y_list, epochs=epochs)
        test_generator = self.test_batch_plan.batches_generator(
            x_dict, y_list, epochs=epochs)
        return (train_generator, test_generator)
