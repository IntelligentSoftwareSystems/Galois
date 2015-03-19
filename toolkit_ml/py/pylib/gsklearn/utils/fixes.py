"""Compatibility fixes for older version of python, numpy and scipy
If you add content to this file, please give the version of the package
at which the fixe is no longer needed.
"""
# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
# Gael Varoquaux <gael.varoquaux@normalesup.org>
# Fabian Pedregosa <fpedregosa@acm.org>
# Lars Buitinck
#
# License: BSD 3 clause

import inspect
import warnings

import numpy as np
import scipy.sparse as sp


np_version = []
for x in np.__version__.split('.'):
    try:
        np_version.append(int(x))
    except ValueError:
        # x may be of the form dev-1ea1592
        np_version.append(x)
np_version = tuple(np_version)

# vim: set ts=4 sw=4:
