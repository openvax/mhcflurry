from os.path import join
from appdirs import user_data_dir

BASE_DIRECTORY = user_data_dir("mhcflurry", version="0.1")
CLASS1_DATA_DIRECTORY = join(BASE_DIRECTORY, "class1_data")
CLASS1_MODEL_DIRECTORY = join(BASE_DIRECTORY, "class1_models")