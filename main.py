import numpy as np
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
import time
from sklearn.model_selection import ParameterGrid, train_test_split
from simulation_util import server_update
import numpy as np
import random_data_gen as rdata_gen
import pandas as pd


NUM_SAMPLES = 20000
NUM_LABELS = 3
NUM_FEATURES = 4
NUM_CLIENTS = 100
g_prms = rdata_gen.InputGenParams(NUM_SAMPLES, NUM_LABELS, NUM_FEATURES, NUM_CLIENTS)
df = pd.read_csv("datasets/blob_S20000_L3_F4_U100.csv")

sim_labels, sim_features = rdata_gen.transform_data_for_simulator_format(df, g_prms)
features = np.array(sim_features)
labels = np.array(sim_labels)

# features.flatten()

print(features[0])
# (100, 200, 4)
# (100, 200)
print(features.shape)
print(labels.shape)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=0)
X_train = np.reshape(X_train, (X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0] * X_test.shape[1], X_test.shape[2]))
y_test = np.reshape(y_test, y_test.size)
y_train = np.reshape(y_train, y_train.size)


#use all digits
# mnist = fetch_mldata("MNIST original")

# X_train, y_train = mnist.data[:70000] / 255., mnist.target[:70000]
print("Train: ",X_train.shape,y_train.shape)
# X_train, y_train = shuffle(X_train, y_train)
# X_test, y_test = X_train[60000:70000], y_train[60000:70000]  
print("TEST:", X_test.shape, y_test.shape)
step = 100
batches= np.arange(0,12000,step)
all_classes = np.array([0,1,2])
classifier = SGDClassifier()
for curr in batches:
    X_curr, y_curr = X_train[curr:curr+step], y_train[curr:curr+step]
    classifier.partial_fit(X_curr, y_curr, classes=all_classes)
    score= classifier.score(X_test, y_test)
    print(score)

print("all done")

clf = SGDClassifier()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Score:", score)



# Load the data


# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=0)
