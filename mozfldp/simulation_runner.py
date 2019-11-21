# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from mozfldp.client import Client
from mozfldp.server import ServerFacade

import json

import numpy as np


def _format_data_for_model(dataset, label_col, user_id_col):
    """Split a DataFrame into feature & label arrays."""
    if dataset is None or len(dataset) == 0:
        return (None, None)
    if user_id_col in dataset.columns:
        dataset = dataset.drop(columns=user_id_col)
    return (np.asarray(dataset.drop(columns=label_col)), np.asarray(dataset[label_col]))


class BaseSimulationRunner:
    """Manager for the server/client simulation infrastructure.

    On initialization, the data is allocated across clients.
    Individual simulation rounds can then be performed using
    `run_simulation_round()`.

    model: a `SGDModel` instance to be used for the simulation
    training_data: a DataFrame containing the full training data. It should have
        columns indicating a user ID and training labels. All other columns are
        considered to be features
    coef_init: the initial choice for the model coefficients
    intercept_init: the initial choice for the model intercept
    test_data: a DataFrame in the same format as `training_data`
    label_col: name of the column containing labels
    user_id_col: name of the column containing user IDs
    """

    def __init__(
        self,
        model,
        training_data,
        coef_init,
        intercept_init,
        test_data=None,
        label_col="label",
        user_id_col="user_id",
    ):
        self._model = model
        # Make the model aware of the full set of labels (necessary for partial
        # fit updating).
        self._model.set_training_classes(training_data[label_col])
        # Maintain the history of model weights for each round.
        self._coefs = [coef_init]
        self._intercepts = [intercept_init]
        self._num_rounds_completed = 0

        # Initialize the clients with their respective datasets.
        self._clients = []
        by_user_data = training_data.groupby("user_id")
        for user_id, user_data in by_user_data:
            feats, labs = _format_data_for_model(user_data, label_col, user_id_col)
            self._clients.append(
                Client(user_id, feats, labs, self._model.get_clone(trained=True))
            )

        # Format the test data into features/labels arrays, if provided.
        self._test_data_features, self._test_data_labels = _format_data_for_model(
            test_data, label_col, user_id_col
        )

        self._server = ServerFacade(coef_init, intercept_init)

    def run_simulation_round(self):
        """Perform a single round of model training.

        At the base level, just increment the round counter.
        """
        self._num_rounds_completed += 1


class FLSimulationRunner(BaseSimulationRunner):
    """Simulation runner for standard federated learning.

    In addition to model, data and initial weights as required by
    `BaseSimulationRunner`, supply:

    num_epochs: number of passes through each client's dataset on each round.
    client_fraction: target fraction of clients on which to run updates on each
        round.
    batch_size: target size of client minibatches. Weights are updated once per
        minibatch on each client in a given round.
    """

    def __init__(
        self,
        num_epochs,
        client_fraction,
        batch_size,
        model,
        training_data,
        coef_init,
        intercept_init,
        test_data=None,
        label_col="label",
        user_id_col="user_id",
    ):
        self._num_epochs = num_epochs
        self._client_fraction = client_fraction
        self._batch_size = batch_size

        super().__init__(
            model,
            training_data,
            coef_init,
            intercept_init,
            test_data,
            label_col,
            user_id_col,
        )

    def run_simulation_round(self):
        """Perform a single round of federated learning."""
        # TODO finish implementing this. Should it return the current weights?
        for client in self._clients:
            if np.random.random_sample() < self._client_fraction:
                client.update_and_submit_weights(
                    self._coefs[-1],
                    self._intercepts[-1],
                    self._num_epochs,
                    self._batch_size,
                )
                self._submit_client_weights_temp_hack(client)

        new_coef, new_intercept = self._server.compute_new_weights()
        self._coefs.append(new_coef)
        self._intercepts.append(new_intercept)

        # Increment the round counter.
        super().run_simulation_round()

        return new_coef, new_intercept

    def _submit_client_weights_temp_hack(self, client):
        """Temporary shim to submit client weights to the server."""
        coef, intercept = client._model.get_weights()
        request_dict = {
            "coefs": coef.tolist(),
            "intercept": intercept.tolist(),
            "num_samples": client._n,
        }
        request_json = json.dumps(request_dict)
        self._server.ingest_client_data(request_json)


class FLDPSimulationRunner(BaseSimulationRunner):
    """Simulation runner for federated learning with DP.

    In addition to model, data and initial weights as required by
    `BaseSimulationRunner`, supply:

    num_epochs: number of passes through each client's dataset on each round.
    client_fraction: target fraction of clients on which to run updates on each
        round.
    batch_size: target size of client minibatches. Weights are updated once per
        minibatch on each client in a given round.
    sensitivity: limit on the change in weights from a client update
    noise_scale: desired noise scale. Controls the amount of privacy budget
        spent on each round.
    user_weight_cap: limit on the influence of a single user in the federated
        averaging
    """

    def __init__(
        self,
        num_epochs,
        client_fraction,
        batch_size,
        sensitivity,
        noise_scale,
        user_weight_cap,
        model,
        training_data,
        coef_init,
        intercept_init,
        test_data=None,
        label_col="label",
        user_id_col="user_id",
    ):
        self._num_epochs = num_epochs
        self._client_fraction = client_fraction
        self._batch_size = batch_size
        self._sensitivity = sensitivity
        self._noise_scale = noise_scale
        self._user_weight_cap = user_weight_cap

        super().__init__(
            model,
            training_data,
            coef_init,
            intercept_init,
            test_data,
            label_col,
            user_id_col,
        )

        # TODO: maintain user contribution weights in the Clients.
        # Maybe call a method to set and return the weights on each client, and
        # accumulate them here in the weight sum.
        # user_contrib_weight_sum = 0
        # for client in self._clients:
        #     user_contrib_weight_sum += client.update_contrib_weight(self._user_weight_cap)
        # TODO initialize standard deviation

    def run_simulation_round(self):
        """Perform a single round of federated learning with DP."""
        for client in self._clients:
            if np.random.random_sample() < self._client_fraction:
                client.update_and_submit_weights_dp(
                    self._coefs[-1],
                    self._intercepts[-1],
                    self._num_epochs,
                    self._batch_size,
                    self._sensitivity,
                )

        new_coef, new_intercept = self._server.compute_new_weights_dp(
            self._standard_dev, self._client_contrib_weight_sum
        )

        self._coefs.append(new_coef)
        self._intercepts.append(new_intercept)

        # Increment the round counter.
        super().run_simulation_round()

        return new_coef, new_intercept
