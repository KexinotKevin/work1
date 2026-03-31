import time
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin, clone


class DeconfoundWrapper(BaseEstimator, RegressorMixin):
    """
    双向去交杂包装器：自动分离特征与协变量，在内部对 X 和 y 进行残差化。
    """
    def __init__(self, base_estimator, n_confounds=2):
        self.base_estimator = base_estimator
        self.n_confounds = n_confounds

    def fit(self, X, y):
        # 1. 拆分出主特征和协变量 (假设协变量拼接在 X 的最后面)
        X_feat = X[:, :-self.n_confounds]
        C = X[:, -self.n_confounds:]

        # 2. 拟合协变量到特征和标签的线性关系
        self.model_X_ = LinearRegression().fit(C, X_feat)
        self.model_y_ = LinearRegression().fit(C, y)

        # 3. 计算残差
        X_res = X_feat - self.model_X_.predict(C)
        y_res = y - self.model_y_.predict(C)

        # 4. 用残差数据训练你传入的基础模型（例如带有特征选择的 Pipeline）
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X_res, y_res)
        return self

    def predict(self, X):
        X_feat = X[:, :-self.n_confounds]
        C = X[:, -self.n_confounds:]

        # 1. 对测试特征去交杂 (应用训练集拟合的系数)
        X_res = X_feat - self.model_X_.predict(C)

        # 2. 预测残差目标的得分
        y_res_pred = self.estimator_.predict(X_res)

        # 3. 将协变量的基线影响加回去，使得预测值与原始 y 处于同一尺度
        return y_res_pred + self.model_y_.predict(C)


def get_regression_model(model_group, model_name, **model_params):
    """
    Return a sklearn regressor instance by group and algorithm name.

    Supported groups:
    - linear: lasso, ridge, linear, huber
    - nonlinear: kernel_ridge_rbf, decision_tree, gradient_boosting
    - mlp: mlp_regressor (single hidden layer)
    """
    group = str(model_group).strip().lower().replace("-", "_").replace(" ", "_")
    name = str(model_name).strip().lower().replace("-", "_").replace(" ", "_")

    linear_group = {"linear", "linear_regression", "linearity"}
    nonlinear_group = {"nonlinear", "non_linear", "nonlinear_regression"}
    mlp_group = {"mlp", "single_layer_mlp"}

    if group in linear_group:
        if name in {"lasso", "lasso_regression", "l1"}:
            return Lasso(**model_params)
        if name in {"ridge", "ridge_regression", "l2"}:
            return Ridge(**model_params)
        if name in {"linear", "linear_regression", "simple_linear"}:
            return LinearRegression(**model_params)
        if name in {"huber", "huber_regression", "robust"}:
            return HuberRegressor(**model_params)

    elif group in nonlinear_group:
        if name in {"kernel_ridge", "kernel_ridge_rbf", "rbf_kernel_ridge", "krr"}:
            params = dict(model_params)
            params.setdefault("kernel", "rbf")
            return KernelRidge(**params)
        if name in {"decision_tree", "tree", "decision_tree_regression"}:
            return DecisionTreeRegressor(**model_params)
        if name in {"gradient_boosting", "gbr", "gradient_boosting_regression"}:
            return GradientBoostingRegressor(**model_params)

    elif group in mlp_group:
        if name in {"mlp", "mlp_regressor", "single_layer_mlp"}:
            params = dict(model_params)
            params.setdefault("hidden_layer_sizes", (100,))
            return MLPRegressor(**params)

    raise ValueError(
        f"Unsupported model_group='{model_group}' or model_name='{model_name}'. "
        "Expected groups: linear/nonlinear/mlp."
    )


def train_penaled_regression():
    # model setting: ridge regression
    rg_grid = GridSearchCV(model, cv=10, param_grid={'alpha':alphas})
    reg = Pipeline([
    ('feature_selection', SelectPercentile(f_regression, percentile=pct)),
    ('regression', rg_grid)
    ])

    cv10 = KFold(n_splits=10, random_state=42, shuffle=True)
    # rpcv10 = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42) # repeated Kfolds

    start = time.time() # time the function
    train_pred = cross_val_predict(reg, train_picked_edges.T, lb_col, cv=cv10, n_jobs=20)
    # train_pred = cross_val_score(reg, train_picked_edges.T, lb_col, cv=rpcv10, n_jobs=4)
    end = time.time() 

    # print(" ****** {} ****** \n".format(labellist[k]),
    #     "time: {:.4f}".format(end-start), 
    #         " | r_pearson betwwen train labels and predictions: {:.4f}".format(np.corrcoef(train_pred.T, lb_col.T)[0, 1]),
    #         " | MAE : {:.4f}".format())

    # 训练模型
    reg.fit(train_picked_edges.T, lb_col)

    # 获取最佳参数
    best_alpha = reg.named_steps['regression'].best_params_['alpha']

    # 使用最佳参数创建新的 Ridge 回归模型
    if method=="ridge":
        best_ridge = Ridge(alpha=best_alpha)
    elif method=="kridge":
        best_ridge = KernelRidge(kernel="rbf",alpha=best_alpha)

    # 将特征选择和最佳模型组合成新的 Pipeline
    best_pipeline = Pipeline([
        ('feature_selection', SelectPercentile(f_regression, percentile=pct)),
        ('regression', best_ridge)
    ])

    # 使用最佳参数训练模型
    best_pipeline.fit(train_picked_edges.T, lb_col)

    # for test set
    # 使用最佳模型对测试集进行预测
    test_e = rearrange_edges(test_edges, num_nodes, test_edges.shape[2])
    test_pred = best_pipeline.predict(test_e.T)
    print("r_pearson betwwen test labels and predictions: {:.4f}".format(np.corrcoef(test_pred.T, test_lbs[:, k].T)[0, 1]))


