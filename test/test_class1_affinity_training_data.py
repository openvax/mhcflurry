"""Tests for affinity-model device-resident training data."""

import numpy
import torch

from mhcflurry.class1_affinity_training_data import AffinityDeviceTrainingData


def test_affinity_device_training_data_combined_buffer_layout():
    num_rn = 5
    num_real = 7
    encoded_length = 3 * 15

    x_peptide = numpy.arange(num_real * encoded_length, dtype=numpy.int8).reshape(
        num_real, encoded_length
    )
    rn_template = numpy.zeros((num_rn, encoded_length), dtype=numpy.int8)
    x_allele = numpy.ones((num_real, 39), dtype=numpy.float32) * 2.0
    rn_allele = numpy.ones((num_rn, 39), dtype=numpy.float32) * 3.0
    y_encoded = numpy.linspace(0.0, 1.0, num_rn + num_real, dtype=numpy.float32)
    weights = numpy.ones(num_rn + num_real, dtype=numpy.float32)

    data = AffinityDeviceTrainingData.from_arrays(
        x_peptide=x_peptide,
        x_allele=x_allele,
        y_encoded=y_encoded,
        sample_weights=weights,
        random_negative_x_peptide_template=rn_template,
        random_negative_x_allele=rn_allele,
        device="cpu",
    )

    assert isinstance(data.combined_peptide, torch.Tensor)
    assert data.combined_peptide.shape == (num_rn + num_real, encoded_length)
    assert data.row_count == num_rn + num_real
    assert data.random_negative_count == num_rn
    assert data.device.type == "cpu"
    numpy.testing.assert_array_equal(
        data.combined_peptide[num_rn:].numpy(), x_peptide
    )
    numpy.testing.assert_array_equal(data.x_peptide.numpy(), x_peptide)
    assert (data.combined_peptide[:num_rn] == 0).all()

    data.refill_random_negative_peptides(
        torch.full((num_rn, encoded_length), 17, dtype=torch.int8)
    )
    assert (data.combined_peptide[:num_rn] == 17).all()
    numpy.testing.assert_array_equal(
        data.combined_peptide[num_rn:].numpy(), x_peptide
    )

    assert data.combined_allele.shape == (num_rn + num_real, 39)
    assert (data.combined_allele[:num_rn] == 3.0).all()
    assert (data.combined_allele[num_rn:] == 2.0).all()


def test_affinity_device_training_data_no_random_negatives():
    x_peptide = numpy.arange(7 * 45, dtype=numpy.int8).reshape(7, 45)
    data = AffinityDeviceTrainingData.from_arrays(
        x_peptide=x_peptide,
        x_allele=None,
        y_encoded=numpy.zeros(7, dtype=numpy.float32),
        sample_weights=None,
        random_negative_x_peptide_template=None,
        random_negative_x_allele=None,
        device="cpu",
    )
    assert data.combined_peptide is None
    assert data.random_negative_x_peptide is None
    assert data.random_negative_count == 0
    numpy.testing.assert_array_equal(data.x_peptide.numpy(), x_peptide)


def test_affinity_device_training_data_batch_for_indices():
    data = AffinityDeviceTrainingData.from_arrays(
        x_peptide=numpy.arange(5 * 3, dtype=numpy.int8).reshape(5, 3),
        x_allele=None,
        y_encoded=numpy.arange(5, dtype=numpy.float32),
        sample_weights=numpy.ones(5, dtype=numpy.float32) * 2.0,
        random_negative_x_peptide_template=None,
        random_negative_x_allele=None,
        device="cpu",
    )

    inputs, y_batch, weights_batch = data.batch_for_indices(
        torch.tensor([3, 1], dtype=torch.long)
    )

    numpy.testing.assert_array_equal(
        inputs["peptide"].numpy(), [[9, 10, 11], [3, 4, 5]]
    )
    numpy.testing.assert_array_equal(y_batch.numpy(), [3.0, 1.0])
    numpy.testing.assert_array_equal(weights_batch.numpy(), [2.0, 2.0])


def test_affinity_device_training_data_validates_row_counts():
    x_peptide = numpy.arange(5 * 3, dtype=numpy.int8).reshape(5, 3)

    try:
        AffinityDeviceTrainingData.from_arrays(
            x_peptide=x_peptide,
            x_allele=None,
            y_encoded=numpy.zeros(4, dtype=numpy.float32),
            sample_weights=None,
            random_negative_x_peptide_template=None,
            random_negative_x_allele=None,
            device="cpu",
        )
    except ValueError as e:
        assert "y_encoded has 4 rows" in str(e)
    else:
        raise AssertionError("expected y_encoded row-count mismatch")

    try:
        AffinityDeviceTrainingData.from_arrays(
            x_peptide=x_peptide,
            x_allele=None,
            y_encoded=numpy.zeros(5, dtype=numpy.float32),
            sample_weights=numpy.zeros(4, dtype=numpy.float32),
            random_negative_x_peptide_template=None,
            random_negative_x_allele=None,
            device="cpu",
        )
    except ValueError as e:
        assert "sample_weights has 4 rows" in str(e)
    else:
        raise AssertionError("expected sample_weights row-count mismatch")


def test_affinity_device_training_data_validates_allele_row_space():
    x_peptide = numpy.arange(5 * 3, dtype=numpy.int8).reshape(5, 3)

    try:
        AffinityDeviceTrainingData.from_arrays(
            x_peptide=x_peptide,
            x_allele=numpy.zeros((4, 2), dtype=numpy.float32),
            y_encoded=numpy.zeros(5, dtype=numpy.float32),
            sample_weights=None,
            random_negative_x_peptide_template=None,
            random_negative_x_allele=None,
            device="cpu",
        )
    except ValueError as e:
        assert "x_allele has 4 rows" in str(e)
    else:
        raise AssertionError("expected x_allele row-count mismatch")

    try:
        AffinityDeviceTrainingData.from_arrays(
            x_peptide=x_peptide,
            x_allele=numpy.zeros((5, 2), dtype=numpy.float32),
            y_encoded=numpy.zeros(7, dtype=numpy.float32),
            sample_weights=None,
            random_negative_x_peptide_template=numpy.zeros((2, 3), dtype=numpy.int8),
            random_negative_x_allele=None,
            device="cpu",
        )
    except ValueError as e:
        assert "allele tensors must either both be present" in str(e)
    else:
        raise AssertionError("expected missing random-negative allele mismatch")

    try:
        AffinityDeviceTrainingData.from_arrays(
            x_peptide=x_peptide,
            x_allele=numpy.zeros((5, 2), dtype=numpy.float32),
            y_encoded=numpy.zeros(7, dtype=numpy.float32),
            sample_weights=None,
            random_negative_x_peptide_template=numpy.zeros((2, 3), dtype=numpy.int8),
            random_negative_x_allele=numpy.zeros((3, 2), dtype=numpy.float32),
            device="cpu",
        )
    except ValueError as e:
        assert "random_negative_x_allele has 3 rows" in str(e)
    else:
        raise AssertionError("expected random-negative allele row-count mismatch")
