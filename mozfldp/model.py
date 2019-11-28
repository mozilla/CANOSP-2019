# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from sklearn.linear_model import SGDClassifier
from sklearn.base import clone
from sklearn.utils.multiclass import unique_labels
import numpy as np

import copy


class SGDModel:
    """Wrapper around `scikit-learn`'s SGD classifier allowing for external
    updating of model weights and a modified training interface.

    Other kwargs supplied to the constructor are passed to the underlying
    classifier.
    """

    def __init__(self, **kwargs):
        self.classifier = SGDClassifier(**kwargs)

    def set_training_classes(self, all_training_labels):
        """Specify the set of all known classes in the training set.

        This is necessary in order to use `classifier.partial_fit()`. It should
        be called at some point prior to starting model training.

        all_training_labels: the labels in the training data (doesn't need to be
            unique). This is needed to enable `partial_fit`.
        """
        self.classifier.classes_ = unique_labels(all_training_labels)

    def get_clone(self, trained=False):
        """Create a clone of this classifier.

        trained: if `True`, maintains current state including trained model
            weights, interation counter, etc. Otherwise, returns an unfitted
            model with the same initialization params.

        Returns a new `SGDModel` instance.
        """
        if trained:
            new_classifier = copy.deepcopy(self.classifier)
        else:
            new_classifier = clone(self.classifier)
            # If class labels have been specified, maintain these.
            try:
                new_classifier.classes_ = self.classifier.classes_
            except AttributeError:
                pass

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
        """Return the current model weights as (coef, intercept).

        If either weight component is not present (ie. the model has not been
        initialized or trained), it will be None.
        """
        return (
            getattr(self.classifier, "coef_", None),
            getattr(self.classifier, "intercept_", None),
        )

    def minibatch_update(self, X, y, labels):
        """Run a single weight update on the given minibatch.

        X and y should be arrays of the appropriate dimensions as required by
        `SGDClassifier.fit()`.

        Note that the internal `SGDClassifier` step counter `t_` increments
        once for each training example in the minibatch. This is used in the
        adaptive learning rate.
        """
        init_coef, init_intercept = self.get_weights()
        # array used to store sum of all the weights for current min-batch
        mini_batch_coefs = np.zeros(init_coef.shape, dtype=np.float64, order="C")
        mini_batch_intercept = np.zeros(
            init_intercept.shape, dtype=np.float64, order="C"
        )

        for i in range(len(X)):
            self.classifier.partial_fit(X[i : (i + 1)], y[i : (i + 1)])

            coef, intercept = self.get_weights()
            # save the weight per sample
            mini_batch_coefs = np.add(mini_batch_coefs, coef)
            mini_batch_intercept = np.add(mini_batch_intercept, intercept)

            # refresh the classifier
            self.set_weights(init_coef, init_intercept)

        # average against number of sample
        mini_batch_coefs = np.true_divide(mini_batch_coefs, len(X))
        mini_batch_intercept = np.true_divide(mini_batch_intercept, len(X))

        self.set_weights(mini_batch_coefs, mini_batch_intercept)
