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


# ── Gaussian Process models (GPyTorch, GPU-accelerated) ──────────────────

import numpy as np
import torch
import gpytorch
from gpytorch.kernels import Kernel, RBFKernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean


class TanimotoKernelGPyTorch(Kernel):
    """Tanimoto kernel for GPyTorch — GPU-accelerated.

    k(x, y) = dot(x, y) / (sum(x) + sum(y) - dot(x, y))
    """
    has_lengthscale = False

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[0], device=x1.device)
        x1_bin = (x1 > 0).float()
        x2_bin = (x2 > 0).float()
        dot = x1_bin @ x2_bin.T
        sum1 = x1_bin.sum(dim=1, keepdim=True)
        sum2 = x2_bin.sum(dim=1, keepdim=True)
        denom = sum1 + sum2.T - dot
        denom = denom.clamp(min=1e-12)
        return dot / denom


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, base_kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPyTorchWrapper:
    """Sklearn-compatible wrapper around GPyTorch ExactGP.

    .fit(X, y) → trains on GPU, .predict(X) → returns posterior mean.
    """
    def __init__(self, base_kernel, training_iter=50, lr=0.1):
        self.base_kernel = base_kernel
        self.training_iter = training_iter
        self.lr = lr
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = None
        self.likelihood_ = None

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self._device)

        self.likelihood_ = GaussianLikelihood().to(self._device)
        self.model_ = ExactGPModel(X_t, y_t, self.likelihood_, self.base_kernel).to(self._device)
        self.model_.train()
        self.likelihood_.train()

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_, self.model_)

        for i in range(self.training_iter):
            optimizer.zero_grad()
            output = self.model_(X_t)
            loss = -mll(output, y_t)
            loss.backward()
            optimizer.step()
            if (i + 1) % 20 == 0:
                print(f"  [gpytorch] iter {i+1}/{self.training_iter}  loss={loss.item():.4f}")

        return self

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
        self.model_.eval()
        self.likelihood_.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood_(self.model_(X_t))
        return preds.mean.cpu().numpy()


# ── GP builders (sklearn-compatible interface) ────────────────────────────

def gp_tanimoto():
    """GPyTorch GPR with Tanimoto kernel."""
    return GPyTorchWrapper(base_kernel=TanimotoKernelGPyTorch())


def gp_rbf():
    """GPyTorch GPR with RBF kernel."""
    return GPyTorchWrapper(base_kernel=RBFKernel())


def gp_matern():
    """GPyTorch GPR with Matern 5/2 kernel."""
    return GPyTorchWrapper(base_kernel=MaternKernel(nu=2.5))


def gp_rq():
    """GPyTorch GPR with Rational Quadratic kernel."""
    from gpytorch.kernels import RQKernel
    return GPyTorchWrapper(base_kernel=RQKernel())
