

def module_cleanup():
    import keras.backend as K
    K.clear_session()
