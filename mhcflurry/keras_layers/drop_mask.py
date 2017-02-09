from keras.layers import Layer

class DropMask(Layer):
    """
    Sometimes we know that a mask is always going to contain 1s (and never 0s)
    due to e.g. slicing the beginning of a sequence with a known min length.
    In that case it can be useful to drop the sequence mask and feed the
    activations to a layer which does not support masking (e.g. Dense).
    """
    supports_masking = True

    def call(self, x, mask):
        return x

    def compute_mask(self, x, mask):
        return None
