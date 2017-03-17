import keras.layers
import keras.backend as K

class MaskedGlobalAveragePooling1D(keras.layers.pooling._GlobalPooling1D):
    """
    Takes an embedded representation of a sentence with dims
    (n_samples, max_length, n_dims)
    where each sample is masked to allow for variable-length inputs.
    Returns a tensor of shape (n_samples, n_dims) after averaging across
    time in a mask-sensitive fashion.
    """
    supports_masking = True

    def call(self, x, mask):
        expanded_mask = K.expand_dims(mask)
        # zero embedded vectors which come from masked characters
        x_masked = x * expanded_mask
        # how many non-masked characters are in each row?
        mask_counts = K.sum(mask, axis=-1)
        # add up the vector representations along the time dimension
        # the result should have dimension (n_samples, n_embedding_dims)
        x_sums = K.sum(x_masked, axis=1)
        # cast the number of non-zero elements to float32 and
        # give it an extra dimension so it can broadcast properly in
        # an elementwise divsion
        counts_cast = K.expand_dims(K.cast(mask_counts, "float32"))
        return x_sums / counts_cast

    def compute_mask(self, x, mask):
        return None

