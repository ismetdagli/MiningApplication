# import necessary libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from thundersvm import SVC

# Create synthetic dataset of 100000 samples
X, y = make_classification(n_samples=100000, n_features=20, n_informative=17, n_redundant=3, random_state=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=5) #put 80% data in training set

# Initialize model
model = SVC(C=100, kernel='rbf')

# Fit the model to training data
model.fit(X_train, y_train)

# Check test set accuracy
accuracy = model.score(X_test, y_test)

print('Accuracy: {}'.format(accuracy))