
from .hyperparameters import HyperparameterDefaults
from .class1_neural_network import Class1NeuralNetwork

class Class1LigandomePredictor(object):
    network_hyperparameter_defaults = HyperparameterDefaults(
        retrain_mode="all",
    )

    def __init__(self, class1_affinity_predictor):
        if not class1_affinity_predictor.pan_allele_models:
            raise NotImplementedError("Pan allele models required")
        if class1_affinity_predictor.allele_to_allele_specific_models:
            raise NotImplementedError("Only pan allele models are supported")
        self.binding_predictors = class1_affinity_predictor.pan_allele_models
        self.network = None

        self.network = Class1NeuralNetwork.merge(
            self.binding_predictors, merge_method="sum")

    def make_network(self):
        import keras
        import keras.backend as K
        from keras.layers import Input
        from keras.models import Model

        models = self.binding_predictors

        if len(models) == 1:
            return models[0]
        assert len(models) > 1

        result = Class1NeuralNetwork(**dict(models[0].hyperparameters))

        # Remove hyperparameters that are not shared by all models.
        for model in models:
            for (key, value) in model.hyperparameters.items():
                if result.hyperparameters.get(key, value) != value:
                    del result.hyperparameters[key]

        assert result._network is None

        networks = [model.network() for model in models]

        layer_names = [[layer.name for layer in network.layers] for network in
            networks]

        pan_allele_layer_names = ['allele', 'peptide', 'allele_representation',
            'flattened_0', 'allele_flat', 'allele_peptide_merged', 'dense_0',
            'dropout_0', 'dense_1', 'dropout_1', 'output', ]

        if all(names == pan_allele_layer_names for names in layer_names):
            # Merging an ensemble of pan-allele architectures
            network = networks[0]
            peptide_input = Input(
                shape=tuple(int(x) for x in K.int_shape(network.inputs[0])[1:]),
                dtype='float32', name='peptide')
            allele_input = Input(shape=(1,), dtype='float32', name='allele')

            allele_embedding = network.get_layer("allele_representation")(
                allele_input)
            peptide_flat = network.get_layer("flattened_0")(peptide_input)
            allele_flat = network.get_layer("allele_flat")(allele_embedding)
            allele_peptide_merged = network.get_layer("allele_peptide_merged")(
                [peptide_flat, allele_flat])

            sub_networks = []
            for (i, network) in enumerate(networks):
                layers = network.layers[
                pan_allele_layer_names.index("allele_peptide_merged") + 1:]
                node = allele_peptide_merged
                for layer in layers:
                    layer.name += "_%d" % i
                    node = layer(node)
                sub_networks.append(node)

            if merge_method == 'average':
                output = keras.layers.average(sub_networks)
            elif merge_method == 'sum':
                output = keras.layers.add(sub_networks)
            elif merge_method == 'concatenate':
                output = keras.layers.concatenate(sub_networks)
            else:
                raise NotImplementedError("Unsupported merge method",
                    merge_method)

            result._network = Model(inputs=[peptide_input, allele_input],
                outputs=[output], name="merged_predictor")
            result.update_network_description()
        else:
            raise NotImplementedError(
                "Don't know merge_method to merge networks with layer names: ",
                layer_names)
        return result


    def fit(self, peptides, labels, experiment_names,
            experiment_name_to_alleles):


        pass

    def predict(self, allele_lists, peptides):
        pass
