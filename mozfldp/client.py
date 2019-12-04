# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np

from mozfldp.server import submit_client_data_request


def _flat_clip(sensitivity, *vecs):
    """Clip vectors by scaling to a given maximum norm.

    The norm is computed across all vectors, ie. applied to their
    concatenation. Returns scaled versions of the original vectors.

    sensitivity: the bound on the vector norm
    """
    norm = np.linalg.norm(np.concatenate(vecs, axis=None))
    if norm > sensitivity:
        scaling = sensitivity / norm
        vecs = tuple([v * scaling for v in vecs])

    return vecs[0] if len(vecs) == 1 else vecs

class Client:
    """A client which trains model updates on its personal dataset for FL.

    client_id: a unique ID to assign each client
    features: an array of shape (`n_examples`, `n_features`) containing the
        client's personal dataset features
    labels: an array of shape (`n_examples`) containing the client's personal
        dataset labels
    model: a `SGDModel` instance to be used for training
    """

    def __init__(self, client_id, features, labels, model):
        if not features.shape[0] == len(labels):
            raise ValueError(
                "Features and labels have incompatible shapes for client {}".format(
                    client_id
                )
            )

        self._id = client_id
        self._features = features
        self._labels = labels
        self._model = model
        self._n = len(labels)
        self._contrib_weight = self._n

    def get_current_weights(self, copy=True):
        """Return the current weights set in the Client's model as (coef, intercept)."""
        return self._model.get_weights(copy=copy)

    def _get_batch_indices(self, batch_size):
        """Randomly split data into minibatches of target size `batch_size`.

        Returns a list of length `ceiling(self._n / batch_size)` containing
        index lists for each batch.
        """
        shuffled_ind = np.random.permutation(self._n)
        return [shuffled_ind[i : i + batch_size] for i in range(0, self._n, batch_size)]

    def _run_model_update_step(self, X, y):
        """Run a single GD update step on the given data minibatch."""
        self._model.minibatch_update(X, y)

    def _submit_weight_updates(
        self, current_coef, current_intercept, num_epochs, batch_size, sensitivity=None
    ):
        """Update the current model weights for FL using the client's data.

        current_coef: current model coefficients to start from
        current_intercept: current model intercept to start from
        num_epochs: number of passes through the client data
        batch_size: size of data minibatch used in each weight update step
        sensitivity: the sensitivity (norm) bound on the weights update from
        each batch

        Resulting weight updates (differences from current weights) are
        submitted to the server.
        """
        self._model.set_weights(current_coef, current_intercept)

        batch_ind_list = self._get_batch_indices(batch_size)
        for epoch in range(num_epochs):
            for batch_ind in batch_ind_list:
                self._run_model_update_step(
                    self._features[batch_ind], self._labels[batch_ind]
                )

                if sensitivity is not None:
                    # Flat clip
                    batch_coef, batch_inter = self._model.get_weights()
                    clipped_coef_update, clipped_inter_update = _flat_clip(
                        sensitivity,
                        batch_coef - current_coef,
                        batch_inter - current_intercept,
                    )
                    self._model.set_weights(
                        current_coef + clipped_coef_update,
                        current_intercept + clipped_inter_update,
                    )

        # load the client weight into json payload
        new_coef, new_intercept = self._model.get_weights()

        coef_update = new_coef - current_coef
        intercept_update = new_intercept - current_intercept

        submit_client_data_request(
            self._id, coef_update, intercept_update, self._contrib_weight
        )

    def submit_weight_updates(
        self, current_coef, current_intercept, num_epochs, batch_size
    ):
        """Update the current model weights for FL using the client's data.

        current_coef: current model coefficients to start from
        current_intercept: current model intercept to start from
        num_epochs: number of passes through the client data
        batch_size: size of data minibatch used in each weight update step

        Resulting weight updates (differences from current weights) are
        submitted to the server.
        """
        self._submit_weight_updates(
            current_coef=current_coef,
            current_intercept=current_intercept,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

    def update_contrib_weight(self, contrib_weight_cap):
        """Set and return the contribution weight in terms of the given cap.

        By default, each user's contribution is weighted by the number of
        observations they train on. This allows for capping the weights, such
        that users with data sizes above the threshold are all weighted the
        same.

        contrib_weight_cap: the capping threshold applied to the personal
        dataset size
        """
        self._contrib_weight = min(self._n / contrib_weight_cap, 1)
        return self._contrib_weight

    def submit_dp_weight_updates(
        self, current_coef, current_intercept, num_epochs, batch_size, sensitivity
    ):
        """Update the current model weights for FL with DP using the client's data.

        current_coef: current model coefficients to start from
        current_intercept: current model intercept to start from
        num_epochs: number of passes through the client data
        batch_size: size of data minibatch used in each weight update step
        sensitivity: the sensitivity bound on user update norms

        Resulting weight updates (differences from current weights) are
        submitted to the server.
        """
        self._submit_weight_updates(
            current_coef=current_coef,
            current_intercept=current_intercept,
            num_epochs=num_epochs,
            batch_size=batch_size,
            sensitivity=sensitivity,
        )

