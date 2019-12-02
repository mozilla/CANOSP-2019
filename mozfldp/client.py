# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
import requests
import json

from decouple import config

HOSTNAME = config("FLDP_HOST", default="127.0.0.1")
PORT = config("FLDP_PORT", default=8000)
API_ENDPOINT_BASE = "http://{hostname:s}:{port:d}/api/v1/ingest_client_data/{{id:s}}".format(
    hostname=HOSTNAME, port=PORT
)

from decouple import config

HOSTNAME = config("FLDP_HOST", "127.0.0.1")
PORT = config("FLDP_PORT", 8000)

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
        coef, intercept = self._model.get_weights()
        if copy:
            coef = np.copy(coef)
            intercept = np.copy(intercept)
        return coef, intercept

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

    def update_and_submit_weights(
        self, current_coef, current_intercept, num_epochs, batch_size
    ):
        """Update the current model weights for FL using the client's data.

        current_coef: current model coefficients to start from
        current_intercept: current model intercept to start from
        num_epochs: number of passes through the client data
        batch_size: size of data minibatch used in each weight update step

        Resulting weights are submitted to the server. Returns the server
        response.
        """
        self._model.set_weights(np.copy(current_coef), np.copy(current_intercept))

        batch_ind_list = self._get_batch_indices(batch_size)
        for epoch in range(num_epochs):
            for batch_ind in batch_ind_list:
                self._run_model_update_step(
                    self._features[batch_ind], self._labels[batch_ind]
                )

        # load the client weight into json payload
        new_coef, new_intercept = self._model.get_weights()
        client_data = {
            "coefs": new_coef.tolist(),
            "intercept": new_intercept.tolist(),
            "num_samples": self._n,
        }
        payload = json.dumps(client_data)

        # send the post request to update the weights
        api_endpoint = API_ENDPOINT_BASE.format(id=str(self._id))
        response = requests.post(url=api_endpoint, json=payload)

        return response

    def update_contrib_weight(self, contrib_weight_cap):
        """Set and return the contribution weight in terms of the given cap."""
        # TODO apply the cap to self._n.

        self._contrib_weight = min(self._n / contrib_weight_cap, 1)
        return self._contrib_weight

    def update_and_submit_weights_dp(
        self, current_coef, current_intercept, num_epochs, batch_size, sensitivity
    ):
        """Update the current model weights for FL with DP using the client's data.

        Resulting weights are submitted to the server.
        """

        # theta assign
        self._model.set_weights(np.copy(current_coef), np.copy(current_intercept))

        batch_ind_list = self._get_batch_indices(batch_size)
        for epoch in range(num_epochs):
            for batch_ind in batch_ind_list:
                self._run_model_update_step(
                    self._features[batch_ind], self._labels[batch_ind]
                )

                # Flat clip
                coef, inter = self._model.get_weights()
                coef = current_coef + Client._flat_clip(sensitivity, coef)
                inter = current_intercept + Client._flat_clip(sensitivity, inter)
                self._model.set_weights(coef, inter)


        # theta - theta_0
        coef, inter = self._model.get_weights()
        coef, inter = coef - current_coef, inter - current_intercept

        # TODO: Currently identical to FL server data submission. Consider making this into a shared function.
        # TODO transmit self._contrib_weight in the place of self._n to the
        # server
        client_data = {
            "coefs": current_coef.tolist(),
            "intercept": current_intercept.tolist(),
            "num_samples": self._n,
        }

        payload = json.dumps(client_data)

        # send the post request to update the weights
        api_endpoint = "http://{hostname:s}:{port:d}/api/v1/ingest_client_data/{id:d}".format(
            hostname=HOSTNAME, port=PORT, id=self._id
        )
        response = requests.post(url=api_endpoint, json=payload)

        return response

    # I don't know if this should be a method/function of Client.
    # It may make more sense for this to live somewhere else.
    def _flat_clip(sensitivity, vec):
        """
        "Clips" a vector, in order to limit how "long" a vector can be.

        sensitivity: Affects how long a vector can be. Higher values --> longer vector.
        """
        norm = np.linalg.norm(vec)
        if norm > sensitivity:
            vec *= sensitivity / norm

        return vec
