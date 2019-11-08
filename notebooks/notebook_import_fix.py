# Borrowed from here: https://stackoverflow.com/a/35273613

"""
When imported, this module makes sure CANOSP-2019 is in sys.path.
All notebooks that rely on modules defined in the project should have this as their first import!
"""

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
