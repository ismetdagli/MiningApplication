import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from sklearn.datasets import make_classification

import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import pickle
import joblib


X, y = make_classification(
    n_samples=26000, n_features=5, n_classes=4, n_informative=3, random_state=0
)


def build_fn(optimizer):
    model = Sequential()
    model.add(
        Dense(2200, input_dim=5, kernel_initializer="he_normal", activation="relu")
    )
    model.add(Dense(2200, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(4, kernel_initializer="he_normal", activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=[
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model

# with tf.device('/CPU:0'):
clf = KerasClassifier(build_fn, optimizer="rmsprop", epochs=5, batch_size=300)
clf.fit(X, y)   
# print("after fit")

# # checkpoint_path = "mlp_tf.checkpoint"
filename = 'MLPClass2_torch.sav'
# clf.save(clf, filename)
# pickle.dump(clf, open(filename, 'wb'))
# with open('model.pkl', 'wb') as f:
#     pickle.dump(clf, f)
# print("after save")


# clf = KerasClassifier(build_fn, optimizer="rmsprop", epochs=5, batch_size=300)

# Load the previously saved weights

# clf = pickle.load(open(filename, 'rb'))
print("after load")



that_time = time.time()
clf.predict(X)
this_time = time.time()
print("predicted 1 iteration in {0} sec".format(this_time - that_time), flush=True)



that_time = time.time()
for i in range(10):
    clf.predict(X)
this_time = time.time()
print("predicted 10 iteration in {0} sec".format(this_time - that_time), flush=True)

# that_time = time.time()
# for i in range(100):
#     clf.predict(X)
# this_time = time.time()
# print("predicted 100 iteration in {0} sec".format(this_time - that_time), flush=True)