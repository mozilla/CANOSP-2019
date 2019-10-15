from sklearn.linear_model import SGDClassifier
import numpy as np
import random


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
        batches_features.append(features[i : i + batch_size])
        batches_labels.append(labels[i : i + batch_size])

    coef = list(init_weights[0])
    intercept = list(init_weights[1])

    # set max_iter to 1 so that each .fit() call only does one training step
    classifier = SGDClassifier(loss="log", max_iter=1)
    classifier.coef_ = np.array(coef)
    classifier.intercept_ = np.array(intercept)

    for epoch in range(epochs):
        for i in range(len(batches_features)):
            # print(coef)
            
            classifier.partial_fit(
                batches_features[i],
                batches_labels[i],
                classes=[0,1,2,3,4,5,6,7,8,9]
            )


            # update the weights so for the next batch the new ones are used
            coef = classifier.coef_
            intercept = classifier.intercept_
        print("epoch", epoch)
    weights = [coef, intercept]

    return weights


def append(list, element):
    """
    helper function to append array into array in numpy
    """
    return np.concatenate((list, [element])) if list is not None else [element]


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
    print(init_weight[0].shape)
    # initialize the weights
    coef = list(init_weight[0])
    intercept = list(init_weight[1])

    # number of clients
    client_num = len(features)
    # fraction of clients
    C = client_fraction

    # use to generate n_k so that the sum of n_k equals to n
    for i in range(num_rounds):
        # calculate the number of clients used in this round
        m = max(int(client_num * C), 1)
        # random set of m client's index
        user_ids = np.array(random.sample(range(client_num), m))

        # print(user_ids)
        num_samples = []

        # grab all the weights from clients
        client_coefs = None
        client_intercepts = None

        for user_id in user_ids:
            client_feature = features[user_id]
            client_label = labels[user_id]

            coefs, intercept = client_update([coef, intercept], epoch, batch_size, client_feature, client_label)

            client_coefs = append(
                client_coefs,
                coefs,
            )

            client_intercepts = append(
                client_intercepts,
                intercept
            )

            num_samples.append(len(client_feature))

        # calculate the new server weights based on new weights coming from client
        new_coefs = np.zeros(init_weight[0].shape, dtype=np.float64, order="C")
        new_intercept = np.zeros(init_weight[1].shape, dtype=np.float64, order="C")
        
        for index, user_id in enumerate(user_ids):
            client_coef = client_coefs[index]
            client_intercept = client_intercepts[index]

            n_k = num_samples[index]              # FIXME: need to user the correct user id based on the radnomization
            added_coef = [value * (n_k) / sum(num_samples) for value in client_coef]
            added_intercept = [value * (n_k) / sum(num_samples) for value in client_intercept]

            new_coefs = np.add(new_coefs, added_coef)
            new_intercept = np.add(new_intercept, added_intercept)

        # update the server weights to newly calculated weights
        coef = new_coefs
        intercept = new_intercept

        if display_weight_per_round:
            print("Updated Weights: ", coef, intercept)

    # load coefficients and intercept into the classifier
    clf = SGDClassifier(loss="log")

    clf.coef_ = coef
    clf.intercept_ = intercept
    clf.classes_ = np.unique(
        list(labels)
    )  # the unique labels are the classes for the classifier

    return clf
