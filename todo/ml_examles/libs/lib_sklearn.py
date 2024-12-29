# -*- coding: utf-8 -*-


# *********************************************
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2021.01.01
# * Version     : 1.0.0
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# **********************************************


# python libraries
from math import remainder
import os
import sys
import numpy as np
from numpy import positive
from numpy.core.fromnumeric import mean
from numpy.lib.npyio import load
from scipy.sparse import data, dia
from scipy.sparse.construct import random

import sklearn
from sklearn import linear_model



# global variable
rng = np.random.RandomState(0)


def test_make_regression():
    """
    测试 make_regression 生成的数据格式
        X.shape=(1000, 100)
        y.shape=(1000,)
    """
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples = 1000, random_state = rng)
    print(X.shape)
    print(y.shape)


def test_train_test_split():
    """
    测试 train_test_split 的分割比例
        X_train.shape / X.shape = 0.75
    """
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples = 1000, random_state = rng)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rng)
    print(f"{X_train.shape[0] / X.shape[0] * 100} %")


def linear_regression_sample():
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import datasets
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y = True)
    diabetes_X = diabetes_X[:, np.newaxis, 2]
    diabetes_X_train = diabetes_X[:-20]
    diabetes_y_train = diabetes_y[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_test = diabetes_y[-20:]
    
    regr = linear_model.LinearRegression()
    regr.fit(diabetes_X_train, diabetes_y_train)
    diabetes_y_pred = regr.predict(diabetes_X_test)
    print("Coefficients:", regr.coef_)
    print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))
    
    plt.scatter(diabetes_X_test, diabetes_y_test, color = "black")  # scatter
    plt.plot(diabetes_X_test, diabetes_y_pred, color = "blue", linewidth = 2)  # line
    plt.xticks(())
    plt.yticks(())
    plt.show()


def non_negative_least_squares_sample():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    # data
    np.random.seed(42)    
    n_samples, n_features = 200, 50
    X = np.random.randn(n_samples, n_features)
    true_coef = 3 * np.random.randn(n_features)
    true_coef[true_coef < 0] = 0
    y = np.dot(X, true_coef)
    y += 5 * np.random.normal(size = (n_samples,))

    # data prepare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

    # model
    # Non-Negative least squares
    reg_nnls = LinearRegression(positive = True)
    reg_nnls.fit(X_train, y_train)
    y_pred_nnls = reg_nnls.predict(X_test)
    r2_score_nnls = r2_score(y_test, y_pred_nnls)
    print("Coefficient:", reg_nnls.coef_)
    print("NNLS R2 score:", r2_score_nnls)

    # OLS
    reg_ols = LinearRegression()
    reg_ols.fit(X_train, y_train)
    y_pred_ols = reg_ols.predict(X_test)
    r2_score_ols = r2_score(y_test, y_pred_ols)
    print("Coefficient:", reg_ols.coef_)
    print("OLS R2 score:", r2_score_ols)

    fig, ax = plt.subplots()
    ax.plot(reg_ols.coef_, reg_nnls.coef_, linewidth = 0, marker = ".")
    low_x, high_x = ax.get_xlim()
    low_y, high_y = ax.get_ylim()
    low = max(low_x, low_y)
    high = min(high_x, high_y)
    ax.plot([low, high], [low, high], ls = "--", c = "red", alpha = 0.5)
    ax.set_xlabel("OLS regression coefficients", fontweight = "bold")
    ax.set_ylabel("NNLS regression coefficients", fontweight = "bold")
    plt.show()


def ridge_regression():
    from sklearn import linear_model

    X = [
        [0, 0],
        [0, 0],
        [1, 1]
    ]
    y = [0, 0.1, 1]
    reg = linear_model.Ridge(alpha = 0.5)
    reg.fit(X, y)
    print(reg.coef_)
    print(reg.intercept_)


def ridge_classification():
    from sklearn import linear_model


def ridge_regression_param_tune():
    import numpy as np
    from sklearn import linear_model

    X = [
        [0, 0],
        [0, 0],
        [1, 1]
    ]
    y = [0, 0.1, 1]
    reg = linear_model.RidgeCV(alphas = np.logspace(-6, 6, 13))
    reg.fit(X, y)
    print(reg.alpha_)


def ridge_regression_coefficients_regularizaton():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import linear_model

    X = 1.0 / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
    y = np.ones(10)

    # #########################################
    # compute paths
    n_alphas = 200
    alphas = np.logspace(-10, -2, n_alphas)  # 200x1
    coefs = []  # 200x10
    for a in alphas:
        ridge = linear_model.Ridge(alpha = a, fit_intercept = False)
        ridge.fit(X, y)
        coefs.append(ridge.coef_)
    print(alphas.shape)
    print(np.array(coefs).shape)
    # #########################################
    # display results
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale("log")
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.xlabel("alpha")
    plt.ylabel("weights")
    plt.title("Ridge coefficients as a function of the regularization")
    plt.axis("tight")
    plt.show()


def test():
    import numpy as np
    import scipy as sp
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.compose import make_column_transformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Ridge
    from sklearn.compose import TransformedTargetRegressor

    # data
    survey = fetch_openml(data_id = 534, as_frame = True)
    X = survey.data[survey.feature_names]
    y = survey.target.values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rng)
    train_dataset = X_train.copy()
    train_dataset.insert(0, "WAGE", y_train)
    # _ = sns.pairplot(train_dataset, kind = "reg", diag_kind = "kde")
    
    # machine learning pipeline
    categorical_columns = ["RACE", "OCCUPATION", "SECTOR", "MARR", "UNION", "SEX", "SOUTH"]
    numerical_columns = ["EDUCATION", "EXPERIENCE", "AGE"]

    preprocessor = make_column_transformer(
        (OneHotEncoder(drop = "if_binary"), categorical_columns),
        remainder = "passthrough",
        verbose_feature_names_out = False
    )

    model = make_pipeline(
        preprocessor,
        TransformedTargetRegressor(
            regressor = Ridge(alpha = 1e-10), 
            func = np.log10, 
            inverse_func = sp.special.exp10
        ),
    )

    # processing the data
    _ = model.fit(X_train, y_train)


def lasso_regression():
    from sklearn import linear_model

    reg = linear_model.Lasso(alpha = 0.1)
    reg.fit([
        [0, 0],
        [1, 1],
        [0, 1],
    ])
    reg.predict([
        [1, 1]
    ])


def lasso_regression_param_tune():
    from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV


def multi_task_lasso():
    from sklearn.linear_model import MultiTaskLasso, MultiTaskLassoCV


def lasso_lars_regression():
    pass


def elastic_net_regression():
    from sklearn.linear_model import ElasticNet, ElasticNetCV


def multi_task_elastic_net():
    from sklearn.linear_model import MultiTaskElasticNet, MultiTaskElasticNetCV


def quantile_regression():
    # data
    import numpy as np
    rng = np.random.RandomState(42)
    x = np.linspace(start = 0, stop = 10, num = 100)
    X = x[:, np.newaxis]
    y_true_mean = 10 + 0.5 * x
    y_normal = y_true_mean + rng.normal(loc = 0, scale = 0.5 + 0.5 * x, size = x.shape[0])


def polynomial_regression():
    pass


def data_leakage_avoid():
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import SelectKBest
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score

    # data
    n_samples, n_features, n_classes = 200, 10000, 2
    rng = np.random.RandomState(42)
    X = rng.standard_normal((n_samples, n_features))
    y = rng.choice(n_classes, n_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state = rng
    )
    pipeline = make_pipeline(
        SelectKBest(k = 25), 
        GradientBoostingClassifier(random_state = rng),
    )
    pipeline.fit(X_train, y_train)

    # score
    score = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"Accuracy: {score:.2f}")
    # score
    scores = cross_val_score(pipeline, X, y)
    print(f"Mean accuracy: {scores.mean():.2f}+/-{scores.std():.2f}")


def sklearn_visualizations():
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import RocCurveDisplay
    from sklearn import datasets

    # data
    X, y = datasets.load_wine(return_X_y = True)
    y = y == 2
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rng)
    
    # svm
    svc = SVC(random_state = rng)
    svc.fit(X_train, y_train)
    svc_disp = RocCurveDisplay.from_estimator(svc, X_test, y_test)
    plt.show()

    # random forest
    rfc = RandomForestClassifier(random_state = rng)
    rfc.fit(X_train, y_train)
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax = ax, alpha = 0.8)
    svc_disp.plot(ax = ax, alpha = 0.8)
    plt.show()


def sklearn_visualizations_2():
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay

    X, y = datasets.fetch_openml(data_id = 1464, return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rng)

    clf = make_pipeline(
        StandardScaler(), 
        LogisticRegression(random_state = rng),
    )
    clf.fit(X_train, y_train)





# 测试代码 main 函数
def main():
    sklearn_visualizations()

if __name__ == "__main__":
    main()
