# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# we should package up the canosp project as a pypi project so we can
# just import it like any other module.

from sklearn.linear_model import SGDClassifier


class Server:
    """
    This class acts as a facade around the functions in
    `simulation_util` to provide a consistent interface to load data
    into a classifier.
    """

    def __init__(self):
        self._classifier = SGDClassifier(loss="hinge", penalty="l2")

    def init(self, coef, intercept):
        """
        Initialize the features and labels to use when the
        SGDClassifier starts on a cold start.
        """
        self._coef = coef
        self._intercept = intercept

    def fit(self, features, labels):
        self._classifier.fit(
            features, labels, coef_init=self._coef, intercept_init=self._intercept
        )
        coef = self._classifier.coef_
        intercept = self._classifier.intercept_
        return {"coef": coef, "intercept": intercept}

    def update_classifier(self):
        # TODO: this is where the magic happens
        #
        # I think roughly, this should be applying a sample of client
        # data through a training loop.
        #
        # I believe you want to take the code from:
        # https://github.com/mozilla/CANOSP-2019/blob/master/simulation_util.py#L98
        # to
        # https://github.com/mozilla/CANOSP-2019/blob/master/simulation_util.py#L149
        # and update the parameters of the SGDClassifier.
        #
        # The samples `S` in simulation_util.py would be equivalent to self._buffer
        # in this class
        pass

    def classify(self, some_input_json):
        """
        Apply the classifer to some sample input
        """
        reshaped_X_test = None
        reshaped_Y_test = None
        score = self._classifier.score(reshaped_X_test, reshaped_Y_test)
        return score
