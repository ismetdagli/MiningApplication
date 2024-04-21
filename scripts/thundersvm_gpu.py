# import necessary libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# from thundersvm import SVC
import pickle
import joblib
import time

from importlib.machinery import SourceFileLoader
thundersvm = SourceFileLoader("thundersvm", "thundersvm/python/thundersvm/thundersvm.py").load_module()


# Create synthetic dataset of 100000 samples
X, y = make_classification(n_samples=100000, n_features=20, n_informative=17, n_redundant=3, random_state=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=5) #put 80% data in training set

# Initialize model
model = thundersvm.SVC()

# Fit the model to training data
# model.fit(X_train, y_train)

# Check test set accuracy
# accuracy = model.score(X_test, y_test)

# print('Accuracy: {}'.format(accuracy))

filename = 'thundersvm_gpu.sav'
# pickle.dump(model, open(filename, 'wb'))

model = pickle.load(open(filename, 'rb'))


that_time = time.time()
y_test_pred  = model.predict(X_test)
this_time = time.time()
print("predicted 1 iteration in {0} sec".format(this_time - that_time), flush=True)



that_time = time.time()
for i in range(10):
    y_test_pred  = model.predict(X_test)

this_time = time.time()
print("predicted 10 iteration in {0} sec".format(this_time - that_time), flush=True)