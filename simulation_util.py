from sklearn.linear_model import SGDClassifier
import numpy as np

def client_update(init_weights, epochs, batch_size, features, labels):
    """
    Given the previous weights from the server, it does updates on the model
    and returns the new set of weights

    init_weights: weights to initialize the training with 
        ex: [coef1, coef2, ..., coefn, intercept]
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
        # TODO: merge the features and labels into one tuple?

    weights = list(init_weights)

    # set max_iter to 1 so that each .fit() call only does one training step
    classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=1)

    for epoch in range(epochs):
        for i in range(len(batches_features)):
            classifier.fit(
                batches_features[i],
                batches_labels[i],
                coef_init=weights[:-1],
                intercept_init=weights[-1],
            )

            # update the weights so for the next batch the new ones are used
            weights = np.append(classifier.coef_[0], classifier.intercept_)

    return weights
