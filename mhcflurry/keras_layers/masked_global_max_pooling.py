import keras.layers
import keras.backend as K

class MaskedGlobalMaxPooling1D(keras.layers.pooling._GlobalPooling1D):
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

        # one flaw here is that we're returning max(0, max(x[:, i])) instead of
        # max(x[:, i])
        return K.max(x_masked, axis=1)

    def compute_mask(self, x, mask):
        return None
