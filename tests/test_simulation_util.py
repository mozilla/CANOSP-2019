# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pytest

def test_simulation_util():
    pass


@pytest.mark.skip("some reason goes here")
def test_some_known_bug_but_we_want_to_skip_the_test():
    raise StandardError("some message here")
