from sklearn.linear_model import SGDClassifier
import numpy as np
import random
import mozfldp.server as server


def client_update(init_weights, epochs, batch_size, features, labels):
    """
    Given the previous weights from the server, it does updates on the model
    and returns the new set of weights

    init_weights: weights to initialize the training with
        ex: [weights of size num_classes*num_features, intercepts of size num_classes]
    epochs: number of epochs to run the training for
    batch_size: the size of each batch of data while training
    features: a 2D array containing features for each sample
        ex: [[feature1, feature2], [feature1, feature2], ...]
    label: an array containing the labels for the corresponding sample in "features"
        ex: [label1, label2, ...]
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

    classifier = SGDClassifier(loss="log")
    classifier.coef_ = np.array(coef)
    classifier.intercept_ = np.array(intercept)

    for epoch in range(epochs):
        for i in range(len(batches_features)):
            classifier.partial_fit(
                batches_features[i],
                batches_labels[i],
                # list of all possible classes - need to get all unique values instead of hardcoding
                classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
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
):
    """
    Calls clientUpdate to get the updated weights from clients, and applies Federated
    Averaging Algorithm to update the weight on server side

    init_weights: weights to initialize the training with
        ex: [weights of size num_classes*num_features, intercepts of size num_classes]
    client_fraction: fraction of clients to use per round
    num_rounds: number of rounds used to update the weight
    features: a 3D array containing features for each sample
        ex: [[[feature1, feature2], [feature1, feature2], ...]]
    label: an array containing the labels for the corresponding sample in "features"
        ex: [label1, label2, ...]
    epoch: number of epochs to run the training for
    batch_size: the size of each batch of data while training
    display_weight_per_round: a boolean value used to toggle the display of weight value per round

    """
    # initialize the weights
    coef = list(init_weight[0])
    intercept = list(init_weight[1])

    # number of clients
    client_num = len(features)
    # fraction of clients
    C = client_fraction

    serv = server.Server(coef, intercept, len(features), client_fraction)

    # use to generate n_k so that the sum of n_k equals to n
    for i in range(num_rounds):
        # calculate the number of clients used in this round
        m = max(int(client_num * C), 1)
        # random set of m client's index
        user_ids = np.array(random.sample(range(client_num), m))

        for user_id in user_ids:
            client_feature = features[user_id]
            num_features = len(client_feature)
            client_label = labels[user_id]
            coefs, intercept = client_update(
                [coef, intercept], epoch, batch_size, client_feature, client_label
            )

            serv.send_weights(coefs, intercept, num_features)

    coef, intercept = serv.compute_new_weights()

    # TODO: extract down to end of function so that we can construct
    # a new SGD using new coef+intercept data.

    # Reconstruct a new classifier so that we can test the accuracy
    # using new coef and intercept

    # load coefficients and intercept into the classifier
    clf = SGDClassifier(loss="log")

    clf.coef_ = coef
    clf.intercept_ = intercept
    clf.classes_ = np.unique(
        list(labels)
    )  # the unique labels are the classes for the classifier

    return clf
