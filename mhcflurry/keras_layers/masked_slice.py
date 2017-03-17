import keras.layers

class MaskedSlice(keras.layers.Lambda):
    """
    Takes an embedded representation of a sentence with dims
    (n_samples, max_length, n_dims)
    where each sample is masked to allow for variable-length inputs.
    Returns a tensor of shape (n_samples, n_dims) which are the first
    and last vectors in each sentence.
    """
    supports_masking = True

    def __init__(
            self,
            time_start,
            time_end,
            *args,
            **kwargs):
        assert time_start >= 0
        assert time_end >= 0
        self.time_start = time_start
        self.time_end = time_end
        super(MaskedSlice, self).__init__(*args, **kwargs)

    def call(self, x, mask):
        return x[:, self.time_start:self.time_end, :]

    def compute_mask(self, x, mask):
        return mask[:, self.time_start:self.time_end, :]

    def get_output_shape_for(self, input_shape):
        assert len(input_shape) == 3
        output_shape = (
            input_shape[0],
            self.time_end - self.time_start + 1,
            input_shape[2])
        return output_shape
