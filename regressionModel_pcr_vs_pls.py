"""
==================================================================
Principal Component Regression vs Partial Least Squares Regression
==================================================================

This example compares `Principal Component Regression
<https://en.wikipedia.org/wiki/Principal_component_regression>`_ (PCR) and
`Partial Least Squares Regression
<https://en.wikipedia.org/wiki/Partial_least_squares_regression>`_ (PLS) on a
toy dataset. Our goal is to illustrate how PLS can outperform PCR when the
target is strongly correlated with some directions in the data that have a
low variance.

PCR is a regressor composed of two steps: first,
:class:`~sklearn.decomposition.PCA` is applied to the training data, possibly
performing dimensionality reduction; then, a regressor (e.g. a linear
regressor) is trained on the transformed samples. In
:class:`~sklearn.decomposition.PCA`, the transformation is purely
unsupervised, meaning that no information about the targets is used. As a
result, PCR may perform poorly in some datasets where the target is strongly
correlated with *directions* that have low variance. Indeed, the
dimensionality reduction of PCA projects the data into a lower dimensional
space where the variance of the projected data is greedily maximized along
each axis. Despite them having the most predictive power on the target, the
directions with a lower variance will be dropped, and the final regressor
will not be able to leverage them.

PLS is both a transformer and a regressor, and it is quite similar to PCR: it
also applies a dimensionality reduction to the samples before applying a
linear regressor to the transformed data. The main difference with PCR is
that the PLS transformation is supervised. Therefore, as we will see in this
example, it does not suffer from the issue we just mentioned.

"""

# %%
# The data
# --------
#
# We start by creating a simple dataset with two features. Before we even dive
# into PCR and PLS, we fit a PCA estimator to display the two principal
# components of this dataset, i.e. the two directions that explain the most
# variance in the data.
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA

path_List = [
    r".\CST_SELF\E122NA-01\0. Ignition window"
  ]

# Generation the Min-Max Scaler
sc = MinMaxScaler()

file_list_csv = []
for idx in range(0, len(path_List)):
  file_list = os.listdir(path_List[idx])
  file_list = [file for file in file_list if file.endswith(".csv")]
  file_list_csv.append(file_list)
  print("{0} file_list : {1}".format(path_List[idx], file_list))

idx_col = []
ROW = 2
COL = 3

idx = 0  # Ignition Window
# for file_idx in range(len(file_list)):
file_idx = 0

# Load the dataset
file_path = path_List[idx] + '/' + file_list[file_idx]
df = pd.read_csv(file_path)
print(file_idx)
print(df.shape)
print(df.columns)

idx_col = df.columns
num_of_feature = len(idx_col)
ROW = int(np.floor(num_of_feature/COL))+1

# Change DataFrame's Index to TimeStamp
dateTime = pd.to_datetime(df[idx_col[0]])
timeStamp = dateTime.values.astype(np.int64) // 10 ** 9
df.index = timeStamp

# Slicing dataset
ChkStartValue = 0.0
Extracted_Params = df.columns[1:]
All_Params = df.columns[:]
Extracted_Data = df.loc[(df['SetPower'] >= ChkStartValue), Extracted_Params]
print("Extracted Params : {0}".format(Extracted_Params))

# Data Scaling
sc_df = sc.fit_transform(Extracted_Data)
Params_Result = list()
sc_df_time = sc.fit_transform(timeStamp.reshape(-1, 1))

rng = np.random.RandomState(0)
# n_samples = 2188
# cov = [[3, 3], [3, 4]]

# Set Dataset to the X
X = sc_df
pca = PCA(n_components=2).fit(X)

plt.figure(1, figsize=(18, 20))

for i in range(len(Extracted_Params)):
    plt.scatter(sc_df_time, X[:, i], alpha=0.3, label=Extracted_Params[i])
plt.legend()
plt.show()

for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp = comp * var  # scale component by its variance explanation power
    plt.plot(
        [0, comp[0]],
        [0, comp[1]],
        label=f"Component {i}",
        linewidth=5,
        color=f"C{i + 12}",
    )
plt.gca().set(
    aspect="equal",
    title="Multi-dimensional dataset with principal components",
    xlabel="time",
    ylabel="second feature",
)
plt.legend()
plt.show()

plt.figure(1, figsize=(18, 20))
for i in range(len(Extracted_Params)):
    plt.scatter(sc_df_time, X[:, i], alpha=0.3, label=Extracted_Params[i])

for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp = comp * var  # scale component by its variance explanation power
    plt.plot(
        [0, comp[0]],
        [0, comp[1]],
        label=f"Component {i}",
        linewidth=5,
        color=f"C{i + 12}",
    )
plt.gca().set(
    aspect="equal",
    title="Multi-dimensional dataset with principal components",
    xlabel="time",
    ylabel="second feature",
)
plt.legend()
plt.show()

# For the purpose of this example, we now define the target `y` such that it is
# strongly correlated with a direction that has a small variance. To this end,
# we will project `X` onto the second component, and add some noise to it.

y = X.dot(pca.components_[0]) # + rng.normal(size=n_samples) / 2

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

axes[0].scatter(X.dot(pca.components_[0]), y, alpha=0.3)
axes[0].set(xlabel="Projected data onto first PCA component", ylabel="y")
axes[1].scatter(X.dot(pca.components_[1]), y, alpha=0.3)
axes[1].set(xlabel="Projected data onto second PCA component", ylabel="y")
plt.tight_layout()
plt.show()

# %%
# Projection on one component and predictive power
# ------------------------------------------------
#
# We now create two regressors: PCR and PLS, and for our illustration purposes
# we set the number of components to 1. Before feeding the data to the PCA step
# of PCR, we first standardize it, as recommended by good practice. The PLS
# estimator has built-in scaling capabilities.
#
# For both models, we plot the projected data onto the first component against
# the target. In both cases, this projected data is what the regressors will
# use as training data.
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import plotly.express as px

# set to the X, y data set
X = sc_df
y = X.dot(pca.components_[0])

# Train/Teat data split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

# Generation PLS, PCR model, and fit data
pls = PLSRegression(n_components=1)
pls.fit(X_train, y_train)

pcr = make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression())
print(pcr)
pcr.fit(X_train, y_train)
pca = pcr.named_steps["pca"]  # retrieve the PCA step of the pipeline

fig, axes = plt.subplots(1, 3, figsize=(20, 10))

axes[0].scatter(pls.transform(X_test), y_test, label="real-value")
axes[0].scatter(pls.transform(X_test), pls.predict(X_test), label="predictions")
axes[0].set(xlabel="Projected data onto first PLS component", ylabel="y",
            title=f"PLS prediction results, r-squared value : {pls.score(X_test, y_test):.3f}")
axes[0].legend()

axes[1].scatter(pca.transform(X_test), y_test, label="real value")
axes[1].scatter(pca.transform(X_test), pcr.predict(X_test), label="predictions")
axes[1].set(xlabel="Projected data onto first PCA component", ylabel="y",
            title=f"PCR prediction results, r-squared value : {pcr.score(X_test, y_test):.3f}")
axes[1].legend()

# Calculating and plotting the SPE(DModX) with prediction and real-value
differences = []
idx_list = []
for idx in range(len(y_test)):
    idx_list.append(idx)
    actual = np.array(y_test[idx])
    predicted = pls.predict(X_test)[idx]
    differences.append(np.sqrt(np.square(np.subtract(actual, predicted))))

axes[2].scatter(idx_list, differences)
axes[2].set(xlabel='Test Data Count', ylabel='SPE (DModX)',
            title='SPE(DmodX, distance to the model of X) : {0:0.3f}'.format(np.mean(differences)))

plt.tight_layout()
plt.show()

# As expected, the unsupervised PCA transformation of PCR has dropped the
# second component, i.e. the direction with the lowest variance, despite
# it being the most predictive direction. This is because PCA is a completely
# unsupervised transformation, and results in the projected data having a low
# predictive power on the target.
#
# On the other hand, the PLS regressor manages to capture the effect of the
# direction with the lowest variance, thanks to its use of target information
# during the transformation: it can recognize that this direction is actually
# the most predictive. We note that the first PLS component is negatively
# correlated with the target, which comes from the fact that the signs of
# eigenvectors are arbitrary.
#
# We also print the R-squared scores of both estimators, which further confirms
# that PLS is a better alternative than PCR in this case. A negative R-squared
# indicates that PCR performs worse than a regressor that would simply predict
# the mean of the target.

print(f"PCR r-squared {pcr.score(X_test, y_test):.3f}")
print(f"PLS r-squared {pls.score(X_test, y_test):.3f}")

# %%
# As a final remark, we note that PCR with 2 components performs as well as
# PLS: this is because in this case, PCR was able to leverage the second
# component which has the most preditive power on the target.

pca_2 = make_pipeline(PCA(n_components=2), LinearRegression())
pca_2.fit(X_train, y_train)
print(f"PCR r-squared with 2 components {pca_2.score(X_test, y_test):.3f}")
