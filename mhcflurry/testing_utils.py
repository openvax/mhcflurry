from . import Class1NeuralNetwork


def module_cleanup():
    import keras.backend as K
    Class1NeuralNetwork.KERAS_MODELS_CACHE.clear()
    K.clear_session()
