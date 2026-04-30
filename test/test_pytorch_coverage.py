"""
Unit tests to increase coverage of PyTorch migration code.

Covers:
- pytorch_losses: get_pytorch_loss registry, StandardLoss, sample weights
- pytorch_layers: get_activation, LocallyConnected1D numerics
- ensemble_centrality: edge cases (<=3 cols, all-NaN, median)
- class1_neural_network: weight init variants, MergedClass1NeuralNetwork,
  skip-connections topology
- class1_affinity_predictor: canonicalize_allele_name round-trip
"""
import warnings
import numpy as np
import pytest
import torch

from mhcflurry.testing_utils import startup, cleanup


@pytest.fixture(autouse=True)
def setup_teardown():
    startup()
    yield
    cleanup()


# ── pytorch_losses ──────────────────────────────────────────────────────────


class TestGetPytorchLoss:
    def test_standard_mse(self):
        from mhcflurry.pytorch_losses import get_pytorch_loss
        loss = get_pytorch_loss("mse")
        assert not loss.supports_inequalities
        pred = torch.tensor([0.5])
        target = torch.tensor([0.3])
        val = loss(pred, target).item()
        assert abs(val - 0.04) < 1e-6

    def test_standard_mae(self):
        from mhcflurry.pytorch_losses import get_pytorch_loss
        loss = get_pytorch_loss("mae")
        pred = torch.tensor([0.5])
        target = torch.tensor([0.3])
        val = loss(pred, target).item()
        assert abs(val - 0.2) < 1e-6

    def test_custom_loss_lookup(self):
        from mhcflurry.pytorch_losses import get_pytorch_loss
        loss = get_pytorch_loss("custom:mse_with_inequalities")
        assert loss.supports_inequalities
        assert not loss.supports_multiple_outputs

    def test_custom_multi_output_lookup(self):
        from mhcflurry.pytorch_losses import get_pytorch_loss
        loss = get_pytorch_loss("custom:mse_with_inequalities_and_multiple_outputs")
        assert loss.supports_inequalities
        assert loss.supports_multiple_outputs

    def test_custom_mass_spec_lookup(self):
        from mhcflurry.pytorch_losses import get_pytorch_loss
        loss = get_pytorch_loss("custom:multiallelic_mass_spec_loss")
        assert loss.supports_inequalities

    def test_unknown_standard_loss_raises(self):
        from mhcflurry.pytorch_losses import get_pytorch_loss
        with pytest.raises(ValueError, match="Unknown standard loss"):
            get_pytorch_loss("huber")

    def test_unknown_custom_loss_raises(self):
        from mhcflurry.pytorch_losses import get_pytorch_loss
        with pytest.raises(ValueError, match="No such custom loss"):
            get_pytorch_loss("custom:nonexistent")


class TestStandardLossWeighted:
    def test_mse_with_sample_weights(self):
        from mhcflurry.pytorch_losses import StandardLoss
        loss = StandardLoss("mse")
        pred = torch.tensor([0.5, 0.3])
        target = torch.tensor([0.5, 0.0])
        weights = torch.tensor([0.0, 1.0])
        val = loss(pred, target, sample_weights=weights).item()
        # Only second sample contributes: 0.3^2 * 1.0 / 1.0
        assert abs(val - 0.09) < 1e-6

    def test_mae_with_sample_weights(self):
        from mhcflurry.pytorch_losses import StandardLoss
        loss = StandardLoss("mae")
        pred = torch.tensor([0.5, 0.3])
        target = torch.tensor([0.5, 0.0])
        weights = torch.tensor([0.0, 1.0])
        val = loss(pred, target, sample_weights=weights).item()
        assert abs(val - 0.3) < 1e-6

    def test_mse_column_vector_predictions_do_not_warn(self):
        from mhcflurry.pytorch_losses import StandardLoss
        loss = StandardLoss("mse")
        pred = torch.tensor([[0.5], [0.3]])
        target = torch.tensor([0.5, 0.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            val = loss(pred, target).item()
        assert abs(val - 0.045) < 1e-6
        assert not caught


class TestMSEWithInequalitiesSampleWeights:
    def test_weighted_equality_loss(self):
        from mhcflurry.pytorch_losses import MSEWithInequalities
        loss_fn = MSEWithInequalities()
        # Two equality targets, weight only the first
        encoded = MSEWithInequalities.encode_y([0.5, 0.5], ["=", "="])
        y_pred = torch.tensor([0.7, 0.9])
        y_true = torch.tensor(encoded)
        weights = torch.tensor([1.0, 0.0])
        val = loss_fn(y_pred, y_true, sample_weights=weights).item()
        # Only first sample: (0.7 - 0.5)^2 * 1.0 / 1.0 = 0.04
        assert abs(val - 0.04) < 1e-6

    def test_encode_y_nan_raises(self):
        from mhcflurry.pytorch_losses import MSEWithInequalities
        with pytest.raises(ValueError, match="NaN"):
            MSEWithInequalities.encode_y([float("nan")])

    def test_encode_y_length_mismatch_raises(self):
        from mhcflurry.pytorch_losses import MSEWithInequalities
        with pytest.raises(ValueError, match="same length"):
            MSEWithInequalities.encode_y([0.5, 0.5], ["="])


class TestMSEMultiOutputSampleWeights:
    def test_weighted_multi_output(self):
        from mhcflurry.pytorch_losses import MSEWithInequalitiesAndMultipleOutputs
        loss_fn = MSEWithInequalitiesAndMultipleOutputs()
        encoded = loss_fn.encode_y([0.5, 0.5], output_indices=[0, 1])
        y_pred = torch.tensor([[0.7, 999.0], [999.0, 0.5]])
        y_true = torch.tensor(encoded)
        weights = torch.tensor([1.0, 0.0])
        val = loss_fn(y_pred, y_true, sample_weights=weights).item()
        # Only first sample, output 0: (0.7 - 0.5)^2 = 0.04
        assert abs(val - 0.04) < 1e-6

    def test_encode_y_negative_output_indices_raises(self):
        from mhcflurry.pytorch_losses import MSEWithInequalitiesAndMultipleOutputs
        with pytest.raises(ValueError, match="Invalid output indices"):
            MSEWithInequalitiesAndMultipleOutputs.encode_y(
                [0.5], output_indices=[-1])

    def test_encode_y_output_indices_shape_mismatch_raises(self):
        from mhcflurry.pytorch_losses import MSEWithInequalitiesAndMultipleOutputs
        with pytest.raises(ValueError, match="Expected output_indices"):
            MSEWithInequalitiesAndMultipleOutputs.encode_y(
                [0.5, 0.5], output_indices=[0])


class TestMultiallelicMassSpecEdgeCases:
    def test_no_hits_returns_zero(self):
        from mhcflurry.pytorch_losses import MultiallelicMassSpecLoss
        loss_fn = MultiallelicMassSpecLoss(delta=0.2)
        y_pred = torch.tensor([[0.5, 0.3]], requires_grad=True)
        y_true = torch.tensor([0.0])  # only decoys
        val = loss_fn(y_pred, y_true)
        assert val.item() == 0.0
        val.backward()  # should not error

    def test_no_decoys_returns_zero(self):
        from mhcflurry.pytorch_losses import MultiallelicMassSpecLoss
        loss_fn = MultiallelicMassSpecLoss(delta=0.2)
        y_pred = torch.tensor([[0.5, 0.3]], requires_grad=True)
        y_true = torch.tensor([1.0])  # only hits
        val = loss_fn(y_pred, y_true)
        assert val.item() == 0.0
        val.backward()


# ── pytorch_layers ──────────────────────────────────────────────────────────


class TestGetActivation:
    def test_tanh(self):
        from mhcflurry.pytorch_layers import get_activation
        act = get_activation("tanh")
        x = torch.tensor([0.0])
        assert act(x).item() == 0.0

    def test_sigmoid(self):
        from mhcflurry.pytorch_layers import get_activation
        act = get_activation("sigmoid")
        assert abs(act(torch.tensor([0.0])).item() - 0.5) < 1e-6

    def test_relu(self):
        from mhcflurry.pytorch_layers import get_activation
        act = get_activation("relu")
        assert act(torch.tensor([-1.0])).item() == 0.0
        assert act(torch.tensor([1.0])).item() == 1.0

    def test_linear_returns_none(self):
        from mhcflurry.pytorch_layers import get_activation
        assert get_activation("linear") is None
        assert get_activation("") is None

    def test_unknown_raises(self):
        from mhcflurry.pytorch_layers import get_activation
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("swish")


class TestLocallyConnected1D:
    def test_output_shape(self):
        from mhcflurry.pytorch_layers import LocallyConnected1D
        lc = LocallyConnected1D(
            in_channels=3, out_channels=5, input_length=10, kernel_size=3
        )
        x = torch.randn(2, 10, 3)
        out = lc(x)
        assert out.shape == (2, 8, 5)

    def test_deterministic_forward(self):
        from mhcflurry.pytorch_layers import LocallyConnected1D
        torch.manual_seed(42)
        lc = LocallyConnected1D(
            in_channels=2, out_channels=1, input_length=4, kernel_size=2,
            activation="linear",
        )
        x = torch.ones(1, 4, 2)
        out = lc(x)
        # With linear activation, output = einsum + bias, should be deterministic
        out2 = lc(x)
        assert torch.allclose(out, out2)


# ── ensemble_centrality ─────────────────────────────────────────────────────


class TestEnsembleCentralityEdgeCases:
    def test_robust_mean_falls_back_to_nanmean_for_few_columns(self):
        from mhcflurry.ensemble_centrality import robust_mean, _nanmean_no_warnings
        arr = np.array([[1.0, 2.0, 3.0]])  # 3 columns => fallback
        result = robust_mean(arr)
        expected = _nanmean_no_warnings(arr)
        np.testing.assert_array_equal(result, expected)

    def test_robust_mean_two_columns(self):
        from mhcflurry.ensemble_centrality import robust_mean
        arr = np.array([[10.0, 20.0]])
        result = robust_mean(arr)
        assert result[0] == 15.0

    def test_robust_mean_all_nan_many_columns(self):
        from mhcflurry.ensemble_centrality import robust_mean
        arr = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]])
        result = robust_mean(arr)
        assert np.isnan(result[0])

    def test_nanmedian_mixed_nans(self):
        from mhcflurry.ensemble_centrality import _nanmedian_no_warnings
        arr = np.array([
            [1.0, 3.0, np.nan],
            [np.nan, np.nan, np.nan],
            [5.0, 5.0, 5.0],
        ])
        result = _nanmedian_no_warnings(arr)
        assert result[0] == 2.0
        assert np.isnan(result[1])
        assert result[2] == 5.0

    def test_nanmean_single_value_per_row(self):
        from mhcflurry.ensemble_centrality import _nanmean_no_warnings
        arr = np.array([
            [np.nan, 7.0, np.nan],
        ])
        result = _nanmean_no_warnings(arr)
        assert result[0] == 7.0

    def test_centrality_measures_dict(self):
        from mhcflurry.ensemble_centrality import CENTRALITY_MEASURES
        assert set(CENTRALITY_MEASURES.keys()) == {"mean", "median", "robust_mean"}


# ── class1_neural_network: weight initialization ────────────────────────────


class TestWeightInitialization:
    def _make_model(self, init):
        from mhcflurry.class1_neural_network import (
            Class1NeuralNetwork,
            Class1NeuralNetworkModel,
        )
        nn_obj = Class1NeuralNetwork(peptide_amino_acid_encoding_torch=False)
        peptide_shape = nn_obj.peptides_to_network_input([]).shape[1:]
        return Class1NeuralNetworkModel(
            peptide_encoding_shape=peptide_shape,
            layer_sizes=[16],
            init=init,
        )

    def test_glorot_uniform(self):
        model = self._make_model("glorot_uniform")
        assert model.output_layer.weight.shape[0] == 1

    def test_glorot_normal(self):
        model = self._make_model("glorot_normal")
        assert model.output_layer.weight.shape[0] == 1

    def test_he_uniform(self):
        model = self._make_model("he_uniform")
        assert model.output_layer.weight.shape[0] == 1

    def test_he_normal(self):
        model = self._make_model("he_normal")
        assert model.output_layer.weight.shape[0] == 1

    def test_biases_are_zero(self):
        model = self._make_model("glorot_uniform")
        for name, param in model.named_parameters():
            if "bias" in name:
                assert torch.all(param == 0), f"Non-zero bias in {name}"


# ── class1_neural_network: MergedClass1NeuralNetwork ────────────────────────


class TestMergedClass1NeuralNetwork:
    def _make_merged(self, merge_method, n_networks=2):
        from mhcflurry.class1_neural_network import (
            Class1NeuralNetwork,
            Class1NeuralNetworkModel,
            MergedClass1NeuralNetwork,
        )
        nn_obj = Class1NeuralNetwork(peptide_amino_acid_encoding_torch=False)
        peptide_shape = nn_obj.peptides_to_network_input([]).shape[1:]
        torch.manual_seed(0)
        networks = []
        for _ in range(n_networks):
            net = Class1NeuralNetworkModel(
                peptide_encoding_shape=peptide_shape,
                layer_sizes=[4],
            )
            networks.append(net)
        return MergedClass1NeuralNetwork(networks, merge_method=merge_method)

    def test_average(self):
        merged = self._make_merged("average")
        inp = {"peptide": torch.randn(3, *merged.networks[0].peptide_encoding_shape)}
        out = merged(inp)
        # Average of 2 networks
        individual = [net(inp) for net in merged.networks]
        expected = torch.stack(individual, dim=-1).mean(dim=-1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_sum(self):
        merged = self._make_merged("sum")
        inp = {"peptide": torch.randn(3, *merged.networks[0].peptide_encoding_shape)}
        out = merged(inp)
        individual = [net(inp) for net in merged.networks]
        expected = torch.stack(individual, dim=-1).sum(dim=-1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_concatenate(self):
        merged = self._make_merged("concatenate")
        inp = {"peptide": torch.randn(3, *merged.networks[0].peptide_encoding_shape)}
        out = merged(inp)
        individual = [net(inp) for net in merged.networks]
        expected = torch.cat(individual, dim=-1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_unknown_merge_method_raises(self):
        merged = self._make_merged("average")
        merged.merge_method = "bad"
        inp = {"peptide": torch.randn(1, *merged.networks[0].peptide_encoding_shape)}
        with pytest.raises(ValueError, match="Unknown merge method"):
            merged(inp)

    def test_get_set_weights_roundtrip(self):
        merged = self._make_merged("average")
        weights = merged.get_weights_list()
        assert len(weights) > 0
        # Setting the same weights back should not change outputs
        inp = {"peptide": torch.randn(2, *merged.networks[0].peptide_encoding_shape)}
        out_before = merged(inp).detach().clone()
        merged.set_weights_list(weights)
        out_after = merged(inp)
        assert torch.allclose(out_before, out_after, atol=1e-6)


# ── class1_neural_network: skip-connections topology ────────────────────────


class TestSkipConnectionsTopology:
    def test_forward_pass(self):
        from mhcflurry.class1_neural_network import (
            Class1NeuralNetwork,
            Class1NeuralNetworkModel,
        )
        nn_obj = Class1NeuralNetwork(peptide_amino_acid_encoding_torch=False)
        peptide_shape = nn_obj.peptides_to_network_input([]).shape[1:]
        torch.manual_seed(7)
        model = Class1NeuralNetworkModel(
            peptide_encoding_shape=peptide_shape,
            layer_sizes=[8, 8, 4],
            topology="with-skip-connections",
        )
        inp = {"peptide": torch.randn(2, *peptide_shape)}
        out = model(inp)
        assert out.shape == (2, 1)

    def test_different_from_feedforward(self):
        from mhcflurry.class1_neural_network import (
            Class1NeuralNetwork,
            Class1NeuralNetworkModel,
        )
        nn_obj = Class1NeuralNetwork(peptide_amino_acid_encoding_torch=False)
        peptide_shape = nn_obj.peptides_to_network_input([]).shape[1:]

        torch.manual_seed(99)
        skip_model = Class1NeuralNetworkModel(
            peptide_encoding_shape=peptide_shape,
            layer_sizes=[8, 8],
            topology="with-skip-connections",
        )
        torch.manual_seed(99)
        ff_model = Class1NeuralNetworkModel(
            peptide_encoding_shape=peptide_shape,
            layer_sizes=[8, 8],
            topology="feedforward",
        )
        inp = {"peptide": torch.randn(2, *peptide_shape)}
        out_skip = skip_model(inp)
        out_ff = ff_model(inp)
        # Different topologies should give different outputs (skip model has
        # different second-layer input dim so weights differ)
        assert not torch.allclose(out_skip, out_ff, atol=1e-4)


# ── class1_affinity_predictor: canonicalize_allele_name ─────────────────────


class TestCanonicalizeAlleleName:
    def test_common_alleles_roundtrip(self):
        """Known HLA alleles should round-trip through canonicalize_allele_name."""
        from mhcflurry.common import normalize_allele_name
        alleles = [
            "HLA-A*02:01", "HLA-A*01:01", "HLA-B*07:02",
            "HLA-B*44:02", "HLA-C*07:01",
        ]
        for allele in alleles:
            result = normalize_allele_name(allele, use_allele_aliases=False)
            assert result == allele, f"{allele} -> {result}"

    def test_aliases_false_avoids_remapping(self):
        """With aliases=False, HLA-C*01:01 should not remap to C*01:02."""
        from mhcflurry.common import normalize_allele_name
        result = normalize_allele_name(
            "HLA-C*01:01", use_allele_aliases=False)
        assert result == "HLA-C*01:01"

    def test_normalize_raises_on_invalid(self):
        from mhcflurry.common import normalize_allele_name
        with pytest.raises(ValueError, match="Invalid MHC allele name"):
            normalize_allele_name("INVALID_ALLELE_NAME")

    def test_normalize_returns_default_on_invalid(self):
        from mhcflurry.common import normalize_allele_name
        result = normalize_allele_name(
            "INVALID_ALLELE_NAME", raise_on_error=False, default_value="NONE")
        assert result == "NONE"

    def test_forbidden_substring_raises(self):
        from mhcflurry.common import normalize_allele_name
        with pytest.raises(ValueError, match="Unsupported gene"):
            normalize_allele_name("HLA-MIC-A")

    def test_forbidden_substring_returns_default(self):
        from mhcflurry.common import normalize_allele_name
        result = normalize_allele_name(
            "HLA-MIC-A", raise_on_error=False, default_value="SKIP")
        assert result == "SKIP"


# ── common.py: configure_pytorch ────────────────────────────────────────────


class TestConfigurePyTorch:
    def test_reconfigure_backend(self):
        from mhcflurry import common
        old_backend = common._pytorch_backend
        common.configure_pytorch(backend="cpu")
        assert common._pytorch_backend == "cpu"
        common.configure_pytorch(backend="auto")
        assert common._pytorch_backend == "auto"
        common._pytorch_backend = old_backend

    def test_invalid_backend_raises(self):
        from mhcflurry import common
        with pytest.raises(ValueError, match="Invalid backend"):
            common.configure_pytorch(backend="gpuu")

    def test_gpu_visibility_does_not_import_torch(self, monkeypatch):
        import builtins
        from mhcflurry import common

        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("configure_pytorch imported torch")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        common.configure_pytorch(gpu_device_nums=[2])
        assert common.os.environ["CUDA_VISIBLE_DEVICES"] == "2"

    def test_default_backend_alias_maps_to_auto(self):
        from mhcflurry import common
        old_backend = common._pytorch_backend
        try:
            common.configure_pytorch(backend="default")
            assert common._pytorch_backend == "auto"
        finally:
            common._pytorch_backend = old_backend

    def test_configure_tensorflow_cpu_backend_maps_to_cpu(self):
        from mhcflurry import common
        old_backend = common._pytorch_backend
        try:
            with pytest.warns(FutureWarning, match="configure_tensorflow"):
                common.configure_tensorflow(backend="tensorflow-cpu")
            assert common._pytorch_backend == "cpu"
            assert str(common.get_pytorch_device()) == "cpu"
        finally:
            common._pytorch_backend = old_backend

    def test_configure_tensorflow_default_alias_maps_to_auto(self):
        from mhcflurry import common
        old_backend = common._pytorch_backend
        try:
            common.configure_pytorch(backend="cpu")
            with pytest.warns(FutureWarning, match="configure_tensorflow"):
                common.configure_tensorflow(backend="tensorflow-default")
            assert common._pytorch_backend == "auto"
        finally:
            common._pytorch_backend = old_backend

    def test_configure_tensorflow_gpu_backend_maps_to_gpu(self):
        from mhcflurry import common
        old_backend = common._pytorch_backend
        try:
            with pytest.warns(FutureWarning, match="configure_tensorflow"):
                common.configure_tensorflow(backend="tensorflow-gpu")
            assert common._pytorch_backend == "gpu"
            if not torch.cuda.is_available():
                with pytest.raises(RuntimeError, match="CUDA is not available"):
                    common.get_pytorch_device()
        finally:
            common._pytorch_backend = old_backend
