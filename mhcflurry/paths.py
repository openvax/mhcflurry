# Copyright (c) 2015. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import join
from appdirs import user_data_dir

BASE_DIRECTORY = user_data_dir("mhcflurry", version="0.1")
CLASS1_DATA_DIRECTORY = join(BASE_DIRECTORY, "class1_data")
CLASS1_MODEL_DIRECTORY = join(BASE_DIRECTORY, "class1_models")

CLASS1_DATA_CSV_FILENAME = "combined_human_class1_dataset.csv"
CLASS1_DATA_CSV_PATH = join(CLASS1_DATA_DIRECTORY, CLASS1_DATA_CSV_FILENAME)
