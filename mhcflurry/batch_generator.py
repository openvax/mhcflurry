import collections

import numpy
import pandas


from .hyperparameters import HyperparameterDefaults


class BatchPlan(object):
    def __init__(self, equivalence_classes, batch_compositions, equivalence_class_labels=None):
        """

        Parameters
        ----------
        equivalence_classes
        batch_compositions
        equivalence_class_labels : list of string, optional
            Used only for summary().
        """
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
        self.equivalence_class_labels = (
            numpy.array(equivalence_class_labels)
            if equivalence_class_labels is not None else None)

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
        equivalence_class_labels = self.equivalence_class_labels
        if equivalence_class_labels is None:
            equivalence_class_labels = (
                "class-" + numpy.arange(self.equivalence_classes).astype("str"))

        i = 0
        while i < len(self.batch_compositions):
            composition = self.batch_compositions[i]
            label_counts = pandas.Series(
                equivalence_class_labels[composition]).value_counts()
            lines.append(
                ("Batch %5d: " % i) + ", ".join(
                    "{key}[{value}]".format(key=key, value=value)
                    for (key, value) in label_counts.iteritems()))
            if i == 5 and len(self.batch_compositions) > i + 3:
                lines.append("...")
                i = len(self.batch_compositions) - i + 1
            i += 1

        indent_spaces = "    " * indent
        return "\n".join([indent_spaces + str(line) for line in lines])

    @property
    def num_batches(self):
        return len(self.batch_compositions)

    @property
    def batch_size(self):
        return max(len(b) for b in self.batch_compositions)


class MultiallelicMassSpecBatchGenerator(object):
    hyperparameter_defaults = HyperparameterDefaults(
        batch_generator_validation_split=0.1,
        batch_generator_batch_size=128,
        batch_generator_affinity_fraction=0.5)
    """
    Hyperperameters for batch generation for the presentation predictor.
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
        df["first_allele"] = df.alleles.str.get(0)
        df["equivalence_key"] = numpy.where(
            df.is_affinity,
            df.first_allele,
            df.experiment_name,
        ) + " " + df.is_binder.map({True: "binder", False: "nonbinder"})
        (df["equivalence_class"], equivalence_class_labels) = (
            df.equivalence_key.factorize())
        df["idx"] = df.index
        df = df.sample(frac=1.0)

        affinities_per_batch = int(affinity_fraction * batch_size)

        remaining_affinities_df = df.loc[df.is_affinity].copy()

        # First do mixed affinity / multiallelic ms batches_generator.
        batch_compositions = []
        for (experiment, experiment_df) in df.loc[~df.is_affinity].groupby(
                "experiment_name"):
            (experiment_alleles,) = experiment_df.alleles.unique()
            remaining_affinities_df["matches_allele"] = (
                remaining_affinities_df.first_allele.isin(experiment_alleles))
            # Whenever possible we try to use affinities with the same
            # alleles as the mass spec experiment
            remaining_affinities_df = remaining_affinities_df.sort_values(
                "matches_allele", ascending=False)
            while len(experiment_df) > 0:
                affinities_for_this_batch = min(
                    affinities_per_batch, len(remaining_affinities_df))
                mass_spec_for_this_batch = (
                    batch_size - affinities_for_this_batch)
                if len(experiment_df) < mass_spec_for_this_batch:
                    mass_spec_for_this_batch = len(experiment_df)
                    affinities_for_this_batch = (
                            batch_size - mass_spec_for_this_batch)

                batch_composition = []

                # take mass spec
                to_use = experiment_df.iloc[:mass_spec_for_this_batch]
                experiment_df = experiment_df.iloc[mass_spec_for_this_batch:]
                batch_composition.extend(to_use.equivalence_class.values)

                # take affinities
                to_use = remaining_affinities_df.iloc[
                    :affinities_for_this_batch
                ]
                remaining_affinities_df = remaining_affinities_df.iloc[
                    affinities_for_this_batch:
                ]
                batch_composition.extend(to_use.equivalence_class.values)
                batch_compositions.append(batch_composition)

        # Affinities-only batches
        while len(remaining_affinities_df) > 0:
            to_use = remaining_affinities_df.iloc[:batch_size]
            remaining_affinities_df = remaining_affinities_df.iloc[batch_size:]
            batch_compositions.append(to_use.equivalence_class.values)

        class_to_indices = df.groupby("equivalence_class").idx.unique()
        equivalence_classes = [
            class_to_indices[i]
            for i in range(len(class_to_indices))
        ]
        return BatchPlan(
            equivalence_classes=equivalence_classes,
            batch_compositions=batch_compositions,
            equivalence_class_labels=equivalence_class_labels)

    def plan(
            self,
            affinities_mask,
            experiment_names,
            alleles_matrix,
            is_binder,
            potential_validation_mask=None):
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
        if potential_validation_mask is not None:
            numpy.testing.assert_equal(len(potential_validation_mask), n)

        validation_items = numpy.random.choice(
            n if potential_validation_mask is None
                else numpy.where(potential_validation_mask)[0],
            int(self.hyperparameters['batch_generator_validation_split'] * n),
            replace=False)
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
            "Train:\n" + self.train_batch_plan.summary(indent=1) +
            "\n***\nTest: " + self.test_batch_plan.summary(indent=1))

    def get_train_and_test_generators(self, x_dict, y_list, epochs=1):
        train_generator = self.train_batch_plan.batches_generator(
            x_dict, y_list, epochs=epochs)
        test_generator = self.test_batch_plan.batches_generator(
            x_dict, y_list, epochs=epochs)
        return (train_generator, test_generator)

    @property
    def num_train_batches(self):
        return self.train_batch_plan.num_batches

    @property
    def num_test_batches(self):
        return self.test_batch_plan.num_batches
