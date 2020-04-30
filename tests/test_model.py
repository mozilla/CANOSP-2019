# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pytest


@pytest.mark.skip("TODO")
def test_get_clone():
    # TODO: get_clone(trained=False) should return an object with the same
    # initialization params but missing coef_/intercept_/t_.
    # get_clone(trained=True) should return an object for which
    # classifier.__getstate__() returns the same as for the original one.
    pass


@pytest.mark.skip("TODO")
def test_minibatch_update():
    # TODO: check weights and iteration counters t_, n_iter_ for correctness.
    pass
