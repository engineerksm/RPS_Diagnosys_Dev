"""
==========================================
One-class SVM with non-linear kernel (RBF)
==========================================

An example using a one-class SVM for novelty detection.

:ref:`One-class SVM <svm_outlier_detection>` is an unsupervised
algorithm that learns a decision function for novelty detection:
classifying new data as similar or different to the training set.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import *

def fit_model(X_train):
    # fit the model
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)

    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])  # , yy.ravel()])
    Z = Z.reshape(yy.shape)
    return Z

def fit_model2(X_train):
    # fit the model
    clf = svm.OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)
    clf.fit(X_train)

    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])  # , yy.ravel()])
    Z = Z.reshape(yy.shape)
    return Z

def fit_model3(X_train):
    # fit the model
    clf = svm.OneClassSVM(nu=0.1, kernel="sigmoid", gamma=0.1)
    clf.fit(X_train)

    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])  # , yy.ravel()])
    Z = Z.reshape(yy.shape)
    return Z

def fit_model4(X_train):
    # fit the model
    clf = svm.OneClassSVM(nu=0.1, kernel="poly", gamma=0.1)
    clf.fit(X_train)

    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])  # , yy.ravel()])
    Z = Z.reshape(yy.shape)
    return Z

# define Criterion Parameter
Parameter_List = ['Master_C1(%)', 'Master_C2(%)']
Scaler_List = ['Min-Max Scaler', 'Robust Scaler', 'Standard Scaler', 'Normalizer']
Algorithm_List = ['rbf SVM', 'Linear SVM', 'sigmoid', 'polynomial']

# Generation the Min-Max Scaler
sc_MinMax = MinMaxScaler()
sc_Robust = RobustScaler()
sc_Standard = StandardScaler()
sc_Normalize = Normalizer()

# Generate x-/y- axis data
xAxisMin = -5
xAxisMax = 5
yAxisMin = -5
yAxisMax = 5
xx, yy = np.meshgrid(np.linspace(xAxisMin, xAxisMax, 500), np.linspace(yAxisMin, yAxisMax, 500))

# Load csv file data
csv_data = pd.read_csv('./Ref_Data/Reference_RFM_Data.csv')
print(csv_data.shape)
print(csv_data.columns)

# Generate train data
# X = 0.3 * np.random.randn(100, 2)
# X_train = np.r_[X + 2, X - 2]
X = csv_data[Parameter_List]
# Data Normalization
X_nor = sc_MinMax.fit_transform(X)
X_nor1 = sc_Robust.fit_transform(X)
X_nor2 = sc_Standard.fit_transform(X)
X_nor3 = sc_Normalize.fit_transform(X)
# assign normalized data to train data label
X_train = X_nor
X_train1 = X_nor1
X_train2 = X_nor2
X_train3 = X_nor3

# Load test csv file data
csv_test_data = pd.read_csv('./Tst_Data/PF_PL0004543_Oxide 28.5kW Matcher Log.csv')
print(csv_test_data.shape)
print(csv_test_data.columns)
# Generate test data
#X = 0.3 * np.random.randn(20, 2)
#X_test = X   #np.r_[X + 2, X - 2]
X_test = csv_test_data[Parameter_List]
# Data Normalization
X_test_nor = sc_MinMax.fit_transform(X_test)
X_test_nor1 = sc_Robust.fit_transform(X_test)
X_test_nor2 = sc_Standard.fit_transform(X_test)
X_test_nor3 = sc_Normalize.fit_transform(X_test)
# assign normalized data to train data label
X_tst = X_test_nor
X_tst1 = X_test_nor1
X_tst2 = X_test_nor2
X_tst3 = X_test_nor3

plt.figure(1)
plt.subplot(2, 2, 1)
plt.title("Anomaly Detection by " + Algorithm_List[0] + " using " + Scaler_List[0])
plt.contourf(xx, yy, fit_model(X_train), levels=np.linspace(fit_model(X_train).min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, fit_model(X_train), levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, fit_model(X_train), levels=[0, fit_model(X_train).max()], colors="palevioletred")
s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_tst[:, 0], X_tst[:, 1], c="blueviolet", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((xAxisMin, xAxisMax))
plt.ylim((yAxisMin, yAxisMax))
plt.legend(
    [a.collections[0], b1, b2],
    [
        "learned frontier",
        "training observations of " + Parameter_List[0] + " " + Parameter_List[1],
        "test observations of " + Parameter_List[0] + " " +Parameter_List[1]
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.subplot(2, 2, 2)
plt.title("Anomaly Detection by " + Algorithm_List[0] + " using" + Scaler_List[1])
plt.contourf(xx, yy, fit_model(X_train1), levels=np.linspace(fit_model(X_train1).min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, fit_model(X_train1), levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, fit_model(X_train1), levels=[0, fit_model(X_train1).max()], colors="palevioletred")
s = 40
b1 = plt.scatter(X_train1[:, 0], X_train1[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_tst1[:, 0], X_tst1[:, 1], c="blueviolet", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((xAxisMin, xAxisMax))
plt.ylim((yAxisMin, yAxisMax))
plt.legend(
    [a.collections[0], b1, b2],
    [
        "learned frontier",
        "training observations of " + Parameter_List[0] + " " + Parameter_List[1],
        "test observations of " + Parameter_List[0] + " " + Parameter_List[1]
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.subplot(2, 2, 3)
plt.title("Anomaly Detection by " + Algorithm_List[0] + " using" + Scaler_List[2])
plt.contourf(xx, yy, fit_model(X_train2), levels=np.linspace(fit_model(X_train2).min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, fit_model(X_train2), levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, fit_model(X_train2), levels=[0, fit_model(X_train2).max()], colors="palevioletred")
s = 40
b1 = plt.scatter(X_train2[:, 0], X_train2[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_tst2[:, 0], X_tst2[:, 1], c="blueviolet", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((xAxisMin, xAxisMax))
plt.ylim((yAxisMin, yAxisMax))
plt.legend(
    [a.collections[0], b1, b2],
    [
        "learned frontier",
        "training observations of " + Parameter_List[0] + " " + Parameter_List[1],
        "test observations of " + Parameter_List[0] + " " + Parameter_List[1]
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.subplot(2, 2, 4)
plt.title("Anomaly Detection by " + Algorithm_List[0] + " using" + Scaler_List[3])
plt.contourf(xx, yy, fit_model(X_train3), levels=np.linspace(fit_model(X_train3).min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, fit_model(X_train3), levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, fit_model(X_train3), levels=[0, fit_model(X_train3).max()], colors="palevioletred")
s = 40
b1 = plt.scatter(X_train3[:, 0], X_train3[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_tst3[:, 0], X_tst3[:, 1], c="blueviolet", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((xAxisMin, xAxisMax))
plt.ylim((yAxisMin, yAxisMax))
plt.legend(
    [a.collections[0], b1, b2],
    [
        "learned frontier",
        "training observations of " + Parameter_List[0] + " " + Parameter_List[1],
        "test observations of " + Parameter_List[0] + " " + Parameter_List[1]
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)

plt.figure(2)
plt.subplot(2, 2, 1)
plt.title("Anomaly Detection by " + Algorithm_List[1] + " using " + Scaler_List[0])
plt.contourf(xx, yy, fit_model2(X_train), levels=np.linspace(fit_model2(X_train).min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, fit_model2(X_train), levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, fit_model2(X_train), levels=[0, fit_model2(X_train).max()], colors="palevioletred")
s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_tst[:, 0], X_tst[:, 1], c="blueviolet", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((xAxisMin, xAxisMax))
plt.ylim((yAxisMin, yAxisMax))
plt.legend(
    [a.collections[0], b1, b2],
    [
        "learned frontier",
        "training observations of " + Parameter_List[0] + " " + Parameter_List[1],
        "test observations of " + Parameter_List[0] + " " + Parameter_List[1]
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.subplot(2, 2, 2)
plt.title("Anomaly Detection by " + Algorithm_List[1] + " using" + Scaler_List[1])
plt.contourf(xx, yy, fit_model2(X_train1), levels=np.linspace(fit_model2(X_train1).min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, fit_model2(X_train1), levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, fit_model2(X_train1), levels=[0, fit_model2(X_train1).max()], colors="palevioletred")
s = 40
b1 = plt.scatter(X_train1[:, 0], X_train1[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_tst1[:, 0], X_tst1[:, 1], c="blueviolet", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((xAxisMin, xAxisMax))
plt.ylim((yAxisMin, yAxisMax))
plt.legend(
    [a.collections[0], b1, b2],
    [
        "learned frontier",
        "training observations of " + Parameter_List[0] + " " + Parameter_List[1],
        "test observations of " + Parameter_List[0] + " " + Parameter_List[1]
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.subplot(2, 2, 3)
plt.title("Anomaly Detection by " + Algorithm_List[1] + " using" + Scaler_List[2])
plt.contourf(xx, yy, fit_model2(X_train2), levels=np.linspace(fit_model2(X_train2).min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, fit_model2(X_train2), levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, fit_model2(X_train2), levels=[0, fit_model2(X_train2).max()], colors="palevioletred")
s = 40
b1 = plt.scatter(X_train2[:, 0], X_train2[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_tst2[:, 0], X_tst2[:, 1], c="blueviolet", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((xAxisMin, xAxisMax))
plt.ylim((yAxisMin, yAxisMax))
plt.legend(
    [a.collections[0], b1, b2],
    [
        "learned frontier",
        "training observations of " + Parameter_List[0] + " " + Parameter_List[1],
        "test observations of " + Parameter_List[0] + " " + Parameter_List[1]
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.subplot(2, 2, 4)
plt.title("Anomaly Detection by " + Algorithm_List[1] + " using" + Scaler_List[3])
plt.contourf(xx, yy, fit_model2(X_train3), levels=np.linspace(fit_model2(X_train3).min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, fit_model2(X_train3), levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, fit_model2(X_train3), levels=[0, fit_model2(X_train3).max()], colors="palevioletred")
s = 40
b1 = plt.scatter(X_train3[:, 0], X_train3[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_tst3[:, 0], X_tst3[:, 1], c="blueviolet", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((xAxisMin, xAxisMax))
plt.ylim((yAxisMin, yAxisMax))
plt.legend(
    [a.collections[0], b1, b2],
    [
        "learned frontier",
        "training observations of " + Parameter_List[0] + " " + Parameter_List[1],
        "test observations of " + Parameter_List[0] + " " + Parameter_List[1]
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)

plt.figure(3)
plt.subplot(2, 2, 1)
plt.title("Anomaly Detection by " + Algorithm_List[2] + " using " + Scaler_List[0])
plt.contourf(xx, yy, fit_model3(X_train), levels=np.linspace(fit_model3(X_train).min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, fit_model3(X_train), levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, fit_model3(X_train), levels=[0, fit_model3(X_train).max()], colors="palevioletred")
s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_tst[:, 0], X_tst[:, 1], c="blueviolet", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((xAxisMin, xAxisMax))
plt.ylim((yAxisMin, yAxisMax))
plt.legend(
    [a.collections[0], b1, b2],
    [
        "learned frontier",
        "training observations of " + Parameter_List[0] + " " + Parameter_List[1],
        "test observations of " + Parameter_List[0] + " " + Parameter_List[1]
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.subplot(2, 2, 2)
plt.title("Anomaly Detection by " + Algorithm_List[2] + " using" + Scaler_List[1])
plt.contourf(xx, yy, fit_model3(X_train1), levels=np.linspace(fit_model3(X_train1).min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, fit_model3(X_train1), levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, fit_model3(X_train1), levels=[0, fit_model3(X_train1).max()], colors="palevioletred")
s = 40
b1 = plt.scatter(X_train1[:, 0], X_train1[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_tst1[:, 0], X_tst1[:, 1], c="blueviolet", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((xAxisMin, xAxisMax))
plt.ylim((yAxisMin, yAxisMax))
plt.legend(
    [a.collections[0], b1, b2],
    [
        "learned frontier",
        "training observations of " + Parameter_List[0] + " " + Parameter_List[1],
        "test observations of " + Parameter_List[0] + " " + Parameter_List[1]
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.subplot(2, 2, 3)
plt.title("Anomaly Detection by " + Algorithm_List[2] + " using" + Scaler_List[2])
plt.contourf(xx, yy, fit_model3(X_train2), levels=np.linspace(fit_model3(X_train2).min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, fit_model3(X_train2), levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, fit_model3(X_train2), levels=[0, fit_model3(X_train2).max()], colors="palevioletred")
s = 40
b1 = plt.scatter(X_train2[:, 0], X_train2[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_tst2[:, 0], X_tst2[:, 1], c="blueviolet", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((xAxisMin, xAxisMax))
plt.ylim((yAxisMin, yAxisMax))
plt.legend(
    [a.collections[0], b1, b2],
    [
        "learned frontier",
        "training observations of " + Parameter_List[0] + " " + Parameter_List[1],
        "test observations of " + Parameter_List[0] + " " + Parameter_List[1]
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.subplot(2, 2, 4)
plt.title("Anomaly Detection by " + Algorithm_List[2] + " using" + Scaler_List[3])
plt.contourf(xx, yy, fit_model3(X_train3), levels=np.linspace(fit_model3(X_train3).min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, fit_model3(X_train3), levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, fit_model3(X_train3), levels=[0, fit_model3(X_train3).max()], colors="palevioletred")
s = 40
b1 = plt.scatter(X_train3[:, 0], X_train3[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_tst3[:, 0], X_tst3[:, 1], c="blueviolet", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((xAxisMin, xAxisMax))
plt.ylim((yAxisMin, yAxisMax))
plt.legend(
    [a.collections[0], b1, b2],
    [
        "learned frontier",
        "training observations of " + Parameter_List[0] + " " + Parameter_List[1],
        "test observations of " + Parameter_List[0] + " " + Parameter_List[1]
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)

plt.figure(4)
plt.subplot(2, 2, 1)
plt.title("Anomaly Detection by " + Algorithm_List[3] + " using " + Scaler_List[0])
plt.contourf(xx, yy, fit_model4(X_train), levels=np.linspace(fit_model4(X_train).min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, fit_model4(X_train), levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, fit_model4(X_train), levels=[0, fit_model4(X_train).max()], colors="palevioletred")
s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_tst[:, 0], X_tst[:, 1], c="blueviolet", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((xAxisMin, xAxisMax))
plt.ylim((yAxisMin, yAxisMax))
plt.legend(
    [a.collections[0], b1, b2],
    [
        "learned frontier",
        "training observations of " + Parameter_List[0] + " " + Parameter_List[1],
        "test observations of " + Parameter_List[0] + " " + Parameter_List[1]
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.subplot(2, 2, 2)
plt.title("Anomaly Detection by " + Algorithm_List[3] + " using" + Scaler_List[1])
plt.contourf(xx, yy, fit_model4(X_train1), levels=np.linspace(fit_model4(X_train1).min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, fit_model4(X_train1), levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, fit_model4(X_train1), levels=[0, fit_model4(X_train1).max()], colors="palevioletred")
s = 40
b1 = plt.scatter(X_train1[:, 0], X_train1[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_tst1[:, 0], X_tst1[:, 1], c="blueviolet", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((xAxisMin, xAxisMax))
plt.ylim((yAxisMin, yAxisMax))
plt.legend(
    [a.collections[0], b1, b2],
    [
        "learned frontier",
        "training observations of " + Parameter_List[0] + " " + Parameter_List[1],
        "test observations of " + Parameter_List[0] + " " + Parameter_List[1]
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.subplot(2, 2, 3)
plt.title("Anomaly Detection by " + Algorithm_List[3] + " using" + Scaler_List[2])
plt.contourf(xx, yy, fit_model4(X_train2), levels=np.linspace(fit_model4(X_train2).min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, fit_model4(X_train2), levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, fit_model4(X_train2), levels=[0, fit_model4(X_train2).max()], colors="palevioletred")
s = 40
b1 = plt.scatter(X_train2[:, 0], X_train2[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_tst2[:, 0], X_tst2[:, 1], c="blueviolet", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((xAxisMin, xAxisMax))
plt.ylim((yAxisMin, yAxisMax))
plt.legend(
    [a.collections[0], b1, b2],
    [
        "learned frontier",
        "training observations of " + Parameter_List[0] + " " + Parameter_List[1],
        "test observations of " + Parameter_List[0] + " " + Parameter_List[1]
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.subplot(2, 2, 4)
plt.title("Anomaly Detection by " + Algorithm_List[3] + " using" + Scaler_List[3])
plt.contourf(xx, yy, fit_model4(X_train3), levels=np.linspace(fit_model4(X_train3).min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, fit_model4(X_train3), levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, fit_model4(X_train3), levels=[0, fit_model4(X_train3).max()], colors="palevioletred")
s = 40
b1 = plt.scatter(X_train3[:, 0], X_train3[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_tst3[:, 0], X_tst3[:, 1], c="blueviolet", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((xAxisMin, xAxisMax))
plt.ylim((yAxisMin, yAxisMax))
plt.legend(
    [a.collections[0], b1, b2],
    [
        "learned frontier",
        "training observations of " + Parameter_List[0] + " " + Parameter_List[1],
        "test observations of " + Parameter_List[0] + " " + Parameter_List[1]
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)

# plt.xlabel(
#     "error train: %d/200 ; errors novel regular: %d/40 ; errors novel abnormal: %d/40"
#     % (n_error_train, n_error_test, n_error_outliers)
# )

# Generate some abnormal novel observations
# X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# y_pred_train = clf.predict(X_train)
# y_pred_test = clf.predict(X_test)
# y_pred_outliers = clf.predict(X_outliers)
# n_error_train = y_pred_train[y_pred_train == -1].size
# n_error_test = y_pred_test[y_pred_test == -1].size
# n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
# c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c="gold", s=s, edgecolors="k")
'''plt.legend(
    [a.collections[0], b1, b2, c],
    [
        "learned frontier",
        "training observations",
        "test observations",
        "Outlier abnormal Data"
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)'''

plt.show()

'''
def z_score_normalize(lst):
    normalized = []

    for value in lst:
        normalized_num = (value - np.mean(lst)) / np.std(lst)
        normalized.append(normalized_num)

    return normalized
'''