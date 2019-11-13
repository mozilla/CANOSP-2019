# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


class Client:
    """A client which trains model updates on its personal dataset for FL.

    client_id: a unique ID to assign each client
    features: an array of shape (`n_examples`, `n_features`) containing the
        client's personal dataset features
    labels: an array of shape (`n_features`) containing the client's personal
        dataset labels
    model: a `SGDModel` instance to be used for training
    """

    def __init__(self, client_id, features, labels, model):
        self._id = client_id
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

    def _run_model_update_step(self, X, y):
        """Run a single GD update step on the given data minibatch."""
        self._model.minibatch_update(X, y)

    def update_and_submit_weights(
        self, current_coef, current_intercept, num_epochs, batch_size
    ):
        """Update the current model weights for FL using the client's data.

        Resulting weights are submitted to the server.
        """
        self._model.set_weights(current_coef, current_intercept)

        # TODO: split data into batches.

        # TODO: update weights across batches and epochs.
        for epoch in range(num_epochs):
            for i in range(batch_size):
                self._run_model_update_step(self._features[i], self._labels[i])

        # TODO: submit new weights to the server via API request.
        # self._model.get_weights() -> server

    def update_and_submit_weights_dp(
        self, current_coef, current_intercept, num_epochs, batch_size
    ):
        """Update the current model weights for FL with DP using the client's data.

        Resulting weights are submitted to the server.
        """
        # TODO
        pass
