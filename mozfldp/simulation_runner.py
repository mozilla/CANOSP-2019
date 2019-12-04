# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from mozfldp.client import Client
from mozfldp.server import compute_weights_request, reset_server_params_request

import json
import sys

# Flag to drop dependencies on the rest of TensorFlow.
sys.skip_tf_privacy_import = True

import numpy as np
from tensorflow_privacy.privacy.analysis import rdp_accountant


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
        self._model = model.get_clone()
        # Make the model aware of the full set of labels (necessary for partial
        # fit updating).
        self._model.set_training_classes(training_data[label_col])
        # Maintain the history of model weights for each round.
        self._coefs = [coef_init]
        self._intercepts = [intercept_init]
        self._num_rounds_completed = 0

        # Initialize the clients with their respective datasets.
        self._clients = []
        if user_id_col is not None and user_id_col in training_data:
            by_user_data = training_data.groupby(user_id_col)
        else:
            by_user_data = [(None, training_data)]
        for user_id, user_data in by_user_data:
            feats, labs = _format_data_for_model(user_data, label_col, user_id_col)
            self._clients.append(
                Client(user_id, feats, labs, self._model.get_clone(trained=True))
            )

        # Format the test data into features/labels arrays, if provided.
        self._test_data_features, self._test_data_labels = _format_data_for_model(
            test_data, label_col, user_id_col
        )

    def reset_server(self, **kwargs):
        """Reset server state prior to running this simulation."""
        reset_server_params_request(coef=self._coefs[-1],
                intercept=self._intercepts[-1], **kwargs)


    def run_simulation_round(self):
        """Perform a single round of model training.

        At the base level, just increment the round counter.
        """
        self._num_rounds_completed += 1


class SGDSimulationRunner(BaseSimulationRunner):
    """Simulation runner for standard (non-federated) minibatch SGD.

    In addition to model, data and initial weights as required by
    `BaseSimulationRunner`, supply:

    num_epochs: number of passes through the dataset on each round.
    batch_size: target size of minibatches. Weights are updated once per minibatch in a given round.
    """

    def __init__(
        self,
        num_epochs,
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
        self._batch_size = batch_size

        # Implement as a single client with all the data.
        super().__init__(
            model,
            training_data.drop(columns=user_id_col),
            coef_init,
            intercept_init,
            test_data,
            label_col,
            None,
        )

        assert len(self._clients) == 1
        self._dummy_client = self._clients[0]

    def run_simulation_round(self):
        """Perform a single round of federated learning."""
        # TODO finish implementing this
        self._dummy_client.submit_weight_updates(
            self._coefs[-1], self._intercepts[-1], self._num_epochs, self._batch_size
        )

        new_coef, new_intercept = self._dummy_client.get_current_weights()
        self._coefs.append(new_coef)
        self._intercepts.append(new_intercept)

        # Increment the round counter.
        self._num_rounds_completed += 1

        return new_coef, new_intercept


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
                client.submit_weight_updates(
                    self._coefs[-1],
                    self._intercepts[-1],
                    self._num_epochs,
                    self._batch_size,
                )

        new_coef, new_intercept = compute_weights_request()
        self._coefs.append(new_coef)
        self._intercepts.append(new_intercept)

        # Increment the round counter.
        self._num_rounds_completed += 1

        return new_coef, new_intercept

    def _submit_client_weights_temp_hack(self, client):
        """Temporary shim to submit client weights to the server."""
        coef, intercept = client._model.get_weights()
        request_dict = {
            "coef_update": coef.tolist(),
            "intercept_update": intercept.tolist(),
            "user_contrib_weight": client._n,
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
    delta: the target value for the privacy parameter delta
    """

    # Order at which to calculate RDP
    # (drawn from examples in TensorFlow Privacy).
    RDP_ORDERS = (
        [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 65)) + [128, 256, 512]
    )

    def __init__(
        self,
        num_epochs,
        client_fraction,
        batch_size,
        sensitivity,
        noise_scale,
        user_weight_cap,
        delta,
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
        self._delta = delta

        # Privacy cost (RDP) can be precomputed.
        self._rdp = rdp_accountant.compute_rdp(
            q=self._client_fraction,
            noise_multiplier=self._noise_scale,
            steps=1,
            orders=self.RDP_ORDERS,
        )

        # Store the progressive epsilon values (privacy budget used).
        # Start from an initial value of 0 to align with the coefficient arrays.
        self._eps = [0]

        super().__init__(
            model,
            training_data,
            coef_init,
            intercept_init,
            test_data,
            label_col,
            user_id_col,
        )

        user_contrib_weight_sum = 0.0
        for client in self._clients:
            user_contrib_weight_sum += client.update_contrib_weight(
                self._user_weight_cap
            )
        self._user_contrib_weight_sum = user_contrib_weight_sum

        self._standard_dev = (self._noise_scale * self._sensitivity) / (
            self._client_fraction * self._user_contrib_weight_sum
        )
        self._avg_denom = self._client_fraction * self._user_contrib_weight_sum

    def _compute_privacy_budget_spent(self):
        """Compute the epsilon value representing the privacy budget spent up to now."""
        current_rdp = self._rdp * self._num_rounds_completed
        eps, _, _ = rdp_accountant.get_privacy_spent(
            orders=self.RDP_ORDERS, rdp=current_rdp, target_delta=self._delta
        )
        return eps

    def reset_server(self, **kwargs):
        """Reset server state prior to running this simulation."""
        super().reset_server(avg_denom=self._avg_denom, standard_dev=self._standard_dev)


    def run_simulation_round(self):
        """Perform a single round of federated learning with DP.

        Returns latest coefficient matrix, intercept vector, and spent privacy budget
        epsilon.
        """

        for client in self._clients:
            if np.random.random_sample() < self._client_fraction:
                client.submit_dp_weight_updates(
                    self._coefs[-1],
                    self._intercepts[-1],
                    self._num_epochs,
                    self._batch_size,
                    self._sensitivity,
                )

        new_coef, new_intercept = compute_weights_request()

        self._coefs.append(new_coef)
        self._intercepts.append(new_intercept)

        # Increment the round counter.
        self._num_rounds_completed += 1

        # Compute the privacy budget consumed up until this point.
        new_eps = self._compute_privacy_budget_spent()
        self._eps.append(new_eps)

        return new_coef, new_intercept, new_eps
