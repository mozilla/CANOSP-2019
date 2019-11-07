# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pytest

def test_get_clone():
    # TODO: get_clone(trained=False) should return an object with the same
    # initialization params but missing coef_/intercept_/t_.
    # get_clone(trained=True) should return an object for which
    # classifier.__getstate__() returns the same as for the original one.
    pass


def test_minibatch_update():
    pass
