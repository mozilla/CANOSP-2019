# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from sklearn.linear_model import SGDClassifier
from sklearn.base import clone
from sklearn.utils.multiclass import unique_labels

import copy


class SGDModel:
    """Wrapper around `scikit-learn`'s SGD classifier allowing for external
    updating of model weights and a modified training interface.

    all_training_labels: the labels in the training data (doesn't need to be
        unique). This is needed to enable `partial_fit`.

    Other kwargs supplied to the constructor are passed to the underlying
    classifier.
    """

    def __init__(self, all_training_labels, **kwargs):
        self.classifier = SGDClassifier(**kwargs)
        self.classifier.classes_ = unique_labels(all_training_labels)

    def get_clone(self, trained=False):
        """Create a clone of this classifier.

        trained: if `True`, maintains current state including trained model
            weights, interation counter, etc. Otherwise, returns an unfitted
            model with the same initialization params.
        """
        if trained:
            new_classifier = copy.deepcopy(self.classifier)
        else:
            new_classifier = clone(self.classifier)

        new_model = self.__class__()
        new_model.classifier = new_classifier
        return new_model

    def __repr__(self):
        return "SGDModel(\n{}\n)".format(self.classifier.__repr__())

    def set_weights(self, coef, intercept):
        """Update the current model weights.

        Note that this leaves the iteration counters as-is. These are used
        internally in computing an adaptive learning rate.
        """
        self.classifier.coef_ = coef
        self.classifier.intercept_ = intercept

    def get_weights(self):
        """Return the current model weights as (coef, intercept)."""
        return (self.classifier.coef_, self.classifier.intercept_)

    def minibatch_update(self, X, y):
        """Run a single weight update on the given minibatch.

        X and y should be arrays of the appropriate dimensions as required by
        `SGDClassifier.fit()`.
        """
        # TODO: implement. Need to consider how to set `t_` and `n_iter_`
        pass
