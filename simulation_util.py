from sklearn.linear_model import SGDClassifier
import numpy as np
import random

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

def append(list, element):
    """
    helper function to append array into array in numpy
    """
    return np.concatenate((list, [element])) if list is not None else [element]

def server_update(init_weight,client_fraction,num_rounds,features,labels,epoch,batch_size,display_weight_per_round):
    
    """
    Calls clientUpdate to get the updated weights from clients, and applies Federated
    Averaging Algorithm to update the weight on server side
    
    init_weights: weights to initialize the training with 
        ex: [coef1, coef2, ..., coefn, intercept]
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

    # initialize the weight
    w = init_weight
    # number of clients
    client_num = len(features)
    # fraction of clients
    C = client_fraction

    # use to generate n_k so that the sum of n_k equals to n
    for i in range(num_rounds):
        # calculate the number of clients used in this round
        m = max(int(client_num*C),1)
        # random set of m client's index
        S = np.array(random.sample(range(client_num), client_num))
        
        num_samples = []
            
        # grab all the weights from clients
        client_weights = None   
        for i in S:
            client_feature = features[i]
            client_label = labels[i]
            client_weights = append(client_weights,client_update(w,epoch,batch_size,client_feature,client_label))         
            num_samples.append(len(client_feature))

        # calculate the new server weight based on new weights coming from client
        new_w = np.zeros(len(w))
        for i in range(len(client_weights)):
            current_weight = client_weights[i]
            n_k = len(features[i])
            added_w = [value*(n_k)/sum(num_samples) for value in current_weight]
            
            new_w = np.add(new_w,added_w)
        
        # update the server weight to newly calculated weight
        w = new_w
        
        if display_weight_per_round:
            print(w)
            
    # load properties and intercept into the classifier
    clf = SGDClassifier()
    clf.coef_ = w[:-1]
    clf.intercept_ = w[-1]
    clf.classes_ = labels
    
    return clf
