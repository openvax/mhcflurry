from keras.layers import Layer

class DropMask(Layer):
    supports_masking = True

    def call(self, x, mask):
        return x

    def compute_mask(self, x, mask):
        return None
