# -*- coding: utf-8 -*-
"""
=====================
Classifier comparison
=====================

A comparison of a several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.

"""

# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.utils import _safe_indexing, check_matplotlib_support
from sklearn.utils.validation import _num_features, check_is_fitted
from sklearn.preprocessing import LabelEncoder
from sklearn.base import is_regressor
from functools import reduce
import pandas as pd
# from sklearn.inspection import DecisionBoundaryDisplay


def _check_boundary_response_method(estimator, response_method):
    has_classes = hasattr(estimator, "classes_")
    # if has_classes and _is_arraylike_not_scalar(estimator.classes_[0]):
    #     msg = "Multi-label and multi-output multi-class classifiers are not supported"
    #     raise ValueError(msg)

    if has_classes and len(estimator.classes_) > 2:
        if response_method not in {"auto", "predict"}:
            msg = (
                "Multiclass classifiers are only supported when response_method is"
                " 'predict' or 'auto'"
            )
            raise ValueError(msg)
        methods_list = ["predict"]
    elif response_method == "auto":
        methods_list = ["decision_function", "predict_proba", "predict"]
    else:
        methods_list = [response_method]

    prediction_method = [getattr(estimator, method, None) for method in methods_list]
    prediction_method = reduce(lambda x, y: x or y, prediction_method)
    if prediction_method is None:
        raise ValueError(
            f"{estimator.__class__.__name__} has none of the following attributes: "
            f"{', '.join(methods_list)}."
        )

    return prediction_method


class DecisionBoundaryDisplay:
    """Decisions boundary visualization.
    It is recommended to use
    :func:`~sklearn.inspection.DecisionBoundaryDisplay.from_estimator`
    to create a :class:`DecisionBoundaryDisplay`. All parameters are stored as
    attributes.
    Read more in the :ref:`User Guide <visualizations>`.
    .. versionadded:: 1.1
    Parameters
    ----------
    xx0 : ndarray of shape (grid_resolution, grid_resolution)
        First output of :func:`meshgrid <numpy.meshgrid>`.
    xx1 : ndarray of shape (grid_resolution, grid_resolution)
        Second output of :func:`meshgrid <numpy.meshgrid>`.
    response : ndarray of shape (grid_resolution, grid_resolution)
        Values of the response function.
    xlabel : str, default=None
        Default label to place on x axis.
    ylabel : str, default=None
        Default label to place on y axis.
    Attributes
    ----------
    surface_ : matplotlib `QuadContourSet` or `QuadMesh`
        If `plot_method` is 'contour' or 'contourf', `surface_` is a
        :class:`QuadContourSet <matplotlib.contour.QuadContourSet>`. If
        `plot_method is `pcolormesh`, `surface_` is a
        :class:`QuadMesh <matplotlib.collections.QuadMesh>`.
    ax_ : matplotlib Axes
        Axes with confusion matrix.
    figure_ : matplotlib Figure
        Figure containing the confusion matrix.
    """

    def __init__(self, *, xx0, xx1, response, xlabel=None, ylabel=None):
        self.xx0 = xx0
        self.xx1 = xx1
        self.response = response
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(self, plot_method="contourf", ax=None, xlabel=None, ylabel=None, **kwargs):
        """Plot visualization.
        Parameters
        ----------
        plot_method : {'contourf', 'contour', 'pcolormesh'}, default='contourf'
            Plotting method to call when plotting the response. Please refer
            to the following matplotlib documentation for details:
            :func:`contourf <matplotlib.pyplot.contourf>`,
            :func:`contour <matplotlib.pyplot.contour>`,
            :func:`pcolomesh <matplotlib.pyplot.pcolomesh>`.
        ax : Matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        xlabel : str, default=None
            Overwrite the x-axis label.
        ylabel : str, default=None
            Overwrite the y-axis label.
        **kwargs : dict
            Additional keyword arguments to be passed to the `plot_method`.
        Returns
        -------
        display: :class:`~sklearn.inspection.DecisionBoundaryDisplay`
        """
        check_matplotlib_support("DecisionBoundaryDisplay.plot")
        import matplotlib.pyplot as plt  # noqa

        if plot_method not in ("contourf", "contour", "pcolormesh"):
            raise ValueError(
                "plot_method must be 'contourf', 'contour', or 'pcolormesh'"
            )

        if ax is None:
            _, ax = plt.subplots()

        plot_func = getattr(ax, plot_method)
        self.surface_ = plot_func(self.xx0, self.xx1, self.response, **kwargs)

        if xlabel is not None or not ax.get_xlabel():
            xlabel = self.xlabel if xlabel is None else xlabel
            ax.set_xlabel(xlabel)
        if ylabel is not None or not ax.get_ylabel():
            ylabel = self.ylabel if ylabel is None else ylabel
            ax.set_ylabel(ylabel)

        self.ax_ = ax
        self.figure_ = ax.figure
        return self

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        *,
        grid_resolution=100,
        eps=1.0,
        plot_method="contourf",
        response_method="auto",
        xlabel=None,
        ylabel=None,
        ax=None,
        **kwargs,
    ):
        check_matplotlib_support(f"{cls.__name__}.from_estimator")
        check_is_fitted(estimator)

        if not grid_resolution > 1:
            raise ValueError(
                "grid_resolution must be greater than 1. Got"
                f" {grid_resolution} instead."
            )

        if not eps >= 0:
            raise ValueError(
                f"eps must be greater than or equal to 0. Got {eps} instead."
            )

        possible_plot_methods = ("contourf", "contour", "pcolormesh")
        if plot_method not in possible_plot_methods:
            available_methods = ", ".join(possible_plot_methods)
            raise ValueError(
                f"plot_method must be one of {available_methods}. "
                f"Got {plot_method} instead."
            )

        num_features = _num_features(X)
        if num_features != 2:
            raise ValueError(
                f"n_features must be equal to 2. Got {num_features} instead."
            )

        x0, x1 = _safe_indexing(X, 0, axis=1), _safe_indexing(X, 1, axis=1)

        x0_min, x0_max = x0.min() - eps, x0.max() + eps
        x1_min, x1_max = x1.min() - eps, x1.max() + eps

        xx0, xx1 = np.meshgrid(
            np.linspace(x0_min, x0_max, grid_resolution),
            np.linspace(x1_min, x1_max, grid_resolution),
        )
        if hasattr(X, "iloc"):
            # we need to preserve the feature names and therefore get an empty dataframe
            X_grid = X.iloc[[], :].copy()
            X_grid.iloc[:, 0] = xx0.ravel()
            X_grid.iloc[:, 1] = xx1.ravel()
        else:
            X_grid = np.c_[xx0.ravel(), xx1.ravel()]

        pred_func = _check_boundary_response_method(estimator, response_method)
        response = pred_func(X_grid)

        # convert classes predictions into integers
        if pred_func.__name__ == "predict" and hasattr(estimator, "classes_"):
            encoder = LabelEncoder()
            encoder.classes_ = estimator.classes_
            response = encoder.transform(response)

        if response.ndim != 1:
            if is_regressor(estimator):
                raise ValueError("Multi-output regressors are not supported")

            # TODO: Support pos_label
            response = response[:, 1]

        if xlabel is None:
            xlabel = X.columns[0] if hasattr(X, "columns") else ""

        if ylabel is None:
            ylabel = X.columns[1] if hasattr(X, "columns") else ""

        display = DecisionBoundaryDisplay(
            xx0=xx0,
            xx1=xx1,
            response=response.reshape(xx0.shape),
            xlabel=xlabel,
            ylabel=ylabel,
        )
        return display.plot(ax=ax, plot_method=plot_method, **kwargs)

# sys.path.append('C:\\Jonathan.Kim_GEN_Dr_Dev\\Jonathan.Kim_GEN_Dr_Dev\\scikit-learn-main\\sklearn')
# import sklearn.inspection._plot.decision_boundary
# from inspection import decision_boundary

names = [
    # "Nearest Neighbors",
    # "Linear SVM",
    # "RBF SVM",
    # "Gaussian Process",
    # "Decision Tree",
    # "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    # make_moons(noise=0.3, random_state=0),
    # make_circles(noise=0.2, factor=0.5, random_state=1),
    linearly_separable,
]

filePath = "./CST_SELF/E122NA-01/1. SRF/DataSet/0. Normal_Case/SRF_Dataset_normal.csv"
raw = pd.read_csv(filePath)
X = raw.values.tolist()
y = np.ones(np.shape(X)[0]).tolist()
linearly_separable = (X, y)

filePath = "./CST_SELF/E122NA-01/1. SRF/DataSet/1. Abnormal_Case/SRF_Dataset_fault.csv"
raw2 = pd.read_csv(filePath)
X2 = raw2.values.tolist()
y2 = np.zeros(np.shape(X2)[0]).tolist()
X += X2
y += y2
linearly_separable = (X, y)

datasets = [
    linearly_separable,
]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        DecisionBoundaryDisplay.from_estimator(
             clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )

        # Plot the training points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )

        # Plot the testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1

plt.tight_layout()
plt.show()
