from . import Class1NeuralNetwork


def module_cleanup():
    import keras.backend as K
    Class1NeuralNetwork.clear_model_cache()
    K.clear_session()
