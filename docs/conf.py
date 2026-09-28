#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import sys
import os
import re
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)

sys.path.insert(0, os.path.abspath('.'))

extensions = [
    'api_coverage',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'myst_parser',
    'numpydoc',
    'sphinxcontrib.programoutput',
    'sphinxcontrib.autoprogram',
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

doctest_global_setup = '''
import logging
logging.getLogger('matplotlib').disabled = True
import numpy
import pandas
import mhcflurry
pandas.set_option('display.max_columns', 20)
pandas.set_option('display.expand_frame_repr', False)
'''

doctest_test_doctest_blocks = ''

templates_path = ['_templates']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'

project = 'MHCflurry'
copyright = '2017–2026, MHCflurry contributors'
author = 'MHCflurry contributors'

with open('../mhcflurry/version.py', 'r') as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(),
        re.MULTILINE).group(1)

release = version

autodoc_member_order = 'bysource'
autoclass_content = 'both'

suppress_warnings = ['image.nonlocal_uri']

language = 'en'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

default_role = 'py:obj'

pygments_style = 'sphinx'

todo_include_todos = False

numpydoc_show_class_members = False

html_theme = 'sphinx_rtd_theme'

html_last_updated_fmt = ""

html_domain_indices = False

html_use_index = False
