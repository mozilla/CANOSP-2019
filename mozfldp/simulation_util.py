from sklearn.linear_model import SGDClassifier
import numpy as np
import random
import mozfldp.server as server
import json

def client_update(
    init_weights, epochs, batch_size, features, labels, all_classes, rand_seed
):
    """
    Given the previous weights from the server, it does updates on the model
    and returns the new set of weights

    init_weights: weights to initialize the training with
        ex: [weights of size num_classes*num_features, intercepts of size num_classes]
    epochs: number of epochs to run the training for
    batch_size: the size of each batch of data while training
    features: a 2D array containing features for each sample
        ex: [[feature1, feature2], [feature1, feature2], ...]
    labels: an array containing the labels for the corresponding sample in "features"
        ex: [label1, label2, ...]
    all_classes: an array containing the unique labels across the entire dataset (`labels` may not contain all of these)
    rand_seed: a seed to use with any random number generation in order to get consistant results between runs
    """

    # split the data into batches by given batch_size
    # TODO: need to ensure that a batch doesn't just contain 1 label
    batches_features = []
    batches_labels = []

    for i in range(0, len(features), batch_size):
        batches_features.append(features[i: i + batch_size])
        batches_labels.append(labels[i: i + batch_size])

    coef = list(init_weights[0])
    intercept = list(init_weights[1])

    classifier = SGDClassifier(loss="log", random_state=rand_seed)
    classifier.coef_ = np.array(coef)
    classifier.intercept_ = np.array(intercept)

    for epoch in range(epochs):
        for i in range(len(batches_features)):
            classifier.partial_fit(
                batches_features[i],
                batches_labels[i],
                # list of all possible classes - need to get all unique values instead of hardcoding
                classes=all_classes,
            )

            # update the weights so for the next batch the new ones are used
            coef = classifier.coef_
            intercept = classifier.intercept_

    weights = [coef, intercept]

    return weights


def server_update(
    init_weight,
    client_fraction,
    num_rounds,
    features,
    labels,
    epoch,
    batch_size,
    display_weight_per_round,
    rand_seed,
):
    """
    Calls client_update to get the updated weights from clients, and applies Federated
    Averaging Algorithm to update the weight on server side

    init_weights: weights to initialize the training with
        ex: [weights of size num_classes*num_features, intercepts of size num_classes]
    client_fraction: fraction of clients to use per round
    num_rounds: number of rounds used to update the weight
    features: a 3D array containing features for each sample
        ex: [[[feature1, feature2], [feature1, feature2], ...]]
    labels: an array containing the labels for the corresponding sample in "features"
        ex: [label1, label2, ...]
    epoch: number of epochs to run the training for
    batch_size: the size of each batch of data while training
    display_weight_per_round: a boolean value used to toggle the display of weight value per round
    rand_seed: a seed to use with any random number generation in order to get consistant results between runs

    """
    # initialize the weights
    coef = init_weight[0]
    intercept = init_weight[1]

    # unique classes in the dataset
    all_classes = np.unique(labels)

    # number of clients
    client_num = len(features)
    # fraction of clients
    C = client_fraction

    # reseed the rng each run
    random.seed(rand_seed)

    serv = server.ServerFacade(coef, intercept, client_num, client_fraction)
    
    # use to generate n_k so that the sum of n_k equals to n
    for i in range(num_rounds):
        # calculate the number of clients used in this round
        m = max(int(client_num * C), 1)
        # random set of m client's index
        user_ids = np.array(random.sample(range(client_num), m))

        for user_id in user_ids:
            client_features = features[user_id]
            num_samples = len(client_features)
            client_labels = labels[user_id]
            coefs, intercept = client_update(
                [coef, intercept],
                epoch,
                batch_size,
                client_features,
                client_labels,
                all_classes,
                rand_seed,
            )

            # this will get moved to the end of Client.update_and_submit_weights
            payload = {
                "coefs": coefs.tolist(),
                "intercept": intercept.tolist(),
                "num_samples": num_samples
            };
            serv.ingest_client_data(json.dumps(payload))

    coef, intercept = serv.compute_new_weights()

    # TODO: extract down to end of function so that we can construct
    # a new SGD using new coef+intercept data.

    # Reconstruct a new classifier so that we can test the accuracy
    # using new coef and intercept

    # load coefficients and intercept into the classifier
    clf = SGDClassifier(loss="log", random_state=rand_seed)

    clf.coef_ = coef
    clf.intercept_ = intercept
    clf.classes_ = np.unique(
        list(labels)
    )  # the unique labels are the classes for the classifier

    return clf
