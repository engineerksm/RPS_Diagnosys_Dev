import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def my_kernel(X, Y):
    return np.dot(X, Y.T)

linear_svc = SVC(kernel='linear')
linear_svc.kernel

rbf_svc = SVC(kernel='rbf')
rbf_svc.kernel

clf = SVC(kernel=my_kernel)
rbf_svc.kernel

X, y = make_classification(n_samples=10, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# clf = SVC(kernel='precomputed')
# gram_train = np.dot(X_train, X_train.T)
clf.fit(X_train, y_train)

# gram_test = np.dot(X_test, X_train.T)
clf.predict(X_test)
