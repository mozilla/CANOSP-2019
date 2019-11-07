# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

class Client:
    """A client which trains model updates on its personal dataset for FL.

    features: an array of shape (`n_examples`, `n_features`) containing the
        client's personal dataset features
    labels: an array of shape (`n_features`) containing the client's personal
        dataset labels
    model: a `SGDModel` instance to be used for training
    """

    def __init__(self, features, labels, model):
        self._features = features
        self._labels = labels
        self._model = model

    def _get_batch_indices(self, batch_size):
        """Randomly split data into minibatches of target size `batch_size`.

        Returns a list `[(i_1_1, i_1_2, ..., i_1_b), (i_2_1, ..., i_2_b), ...]`
        of length `len(features) // batch_size` containing index sets for each
        batch.
        """
        # TODO
        pass

    def update_and_submit_weights(self, current_weights, num_epochs, batch_size):
        """Update the current model weights for FL using the client's data.

        Resulting weights are submitted to the server.
        """
        # TODO
        pass

    def update_and_submit_weights_dp(self, current_weights, num_epochs, batch_size):
        """Update the current model weights for FL with DP using the client's data.

        Resulting weights are submitted to the server.
        """
        # TODO
        pass
