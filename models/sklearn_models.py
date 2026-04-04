"""
Sklearn-compatible baseline models.
"""

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def ridge(alpha: float = 1.0) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))])


def lasso(alpha: float = 0.01) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=alpha, max_iter=5000))])


def elastic_net(alpha: float = 0.01, l1_ratio: float = 0.5) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000))
    ])


def svr_rbf(C: float = 1.0, epsilon: float = 0.1) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=C, epsilon=epsilon))])


def random_forest(n_estimators: int = 300, n_jobs: int = -1) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, random_state=42)


def extra_trees(n_estimators: int = 300, n_jobs: int = -1) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=n_jobs, random_state=42)


def xgboost_model(n_estimators: int = 500, max_depth: int = 6, lr: float = 0.05):
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        device="cuda", n_jobs=-1, random_state=42, verbosity=0,
    )


def lightgbm_model(n_estimators: int = 500, max_depth: int = -1, lr: float = 0.05):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
        subsample=0.8, colsample_bytree=0.8, device="gpu",
        n_jobs=-1, random_state=42, verbose=-1,
    )
