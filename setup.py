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

import os
import logging
import re
import sys

from setuptools import setup

# normally we would import six.PY2 but can't yet assume that six
# is installed here
PY2 = (sys.version_info.major == 2)

readme_dir = os.path.dirname(__file__)
readme_filename = os.path.join(readme_dir, 'README.md')

try:
    with open(readme_filename, 'r') as f:
        readme = f.read()
except:
    logging.warning("Failed to load %s" % readme_filename)
    readme = ""

try:
    import pypandoc
    readme = pypandoc.convert(readme, to='rst', format='md')
except:
    logging.warning("Conversion of long_description from MD to RST failed")
    pass

with open('mhcflurry/version.py', 'r') as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(),
        re.MULTILINE).group(1)

if __name__ == '__main__':
    required_packages = [
        'six',
        'numpy>=1.11',
        'pandas>=0.20.3',
        'Keras>=2.3.0',
        'appdirs',
        'tensorflow>=1.15.0,<2.0.0',
        'scikit-learn',
        'mhcnames',
        'pyyaml',
        'tqdm',
        'np_utils',
    ]

    setup(
        name='mhcflurry',
        version=version,
        description="MHC Binding Predictor",
        author="Tim O'Donnell and Alex Rubinsteyn",
        author_email="timodonnell@gmail.com",
        url="https://github.com/hammerlab/mhcflurry",
        license="http://www.apache.org/licenses/LICENSE-2.0.html",
        entry_points={
            'console_scripts': [
                'mhcflurry-downloads = mhcflurry.downloads_command:run',
                'mhcflurry-predict = mhcflurry.predict_command:run',
                'mhcflurry-predict-scan = mhcflurry.predict_scan_command:run',
                'mhcflurry-class1-train-allele-specific-models = '
                    'mhcflurry.train_allele_specific_models_command:run',
                'mhcflurry-class1-train-pan-allele-models = '
                    'mhcflurry.train_pan_allele_models_command:run',
                'mhcflurry-class1-train-processing-models = '
                    'mhcflurry.train_processing_models_command:run',
                'mhcflurry-class1-select-allele-specific-models = '
                    'mhcflurry.select_allele_specific_models_command:run',
                'mhcflurry-class1-select-pan-allele-models = '
                    'mhcflurry.select_pan_allele_models_command:run',
                'mhcflurry-class1-select-processing-models = '
                    'mhcflurry.select_processing_models_command:run',
                'mhcflurry-calibrate-percentile-ranks = '
                    'mhcflurry.calibrate_percentile_ranks_command:run',
                'mhcflurry-class1-train-presentation-models = '
                    'mhcflurry.train_presentation_models_command:run',
                '_mhcflurry-cluster-worker-entry-point = '
                    'mhcflurry.cluster_parallelism:worker_entry_point',
            ]
        },
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
        ],
        package_data={
            'mhcflurry': ['downloads.yml'],
        },
        install_requires=required_packages,
        long_description=readme,
        packages=[
            'mhcflurry',
        ],
    )
