# losses.py
# Pérdidas convexas con regularización L2 para TRON.
#   - backend con métodos:
#       * margin(w):      X @ w
#       * X_dot(v):       X @ v
#       * Xt_dot(r):      X^T @ r
#   - y en {+1.0, -1.0} (dtype float64)

from __future__ import annotations
import numpy as np
from typing import Protocol


class Backend(Protocol):
    def margin(self, w: np.ndarray) -> np.ndarray: ...
    def X_dot(self, v: np.ndarray) -> np.ndarray: ...
    def Xt_dot(self, r: np.ndarray) -> np.ndarray: ...


def _softplus(x: np.ndarray) -> np.ndarray:
    """
    softplus(x) = log(1 + exp(x)) estable numéricamente.
    """
    # Implementación estable:
    #   if x > 0: log1p(exp(-x)) + x
    #   else:     log1p(exp(x))
    return np.where(x > 0, np.log1p(np.exp(-x)) + x, np.log1p(np.exp(x)))


def _expit(x: np.ndarray) -> np.ndarray:
    """
    expit(x) = 1 / (1 + exp(-x)) estable en extremos.
    """
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


class BaseLoss:
    """
    Clase base para pérdidas L2-regularizadas.
    f(w) = 0.5 ||w||^2 + C * sum_i loss_i(w)
    """

    def __init__(self, C: float, y: np.ndarray, backend: Backend):
        self.C = float(C)
        self.y = y.astype(np.float64, copy=False)
        self.backend = backend

    def f(self, w: np.ndarray) -> float:
        raise NotImplementedError

    def grad(self, w: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def hess_vec(self, w: np.ndarray, s: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LogisticLoss(BaseLoss):
    """
    Pérdida logística binaria (y ∈ {+1, -1}) con regularización L2.
    Definiciones (con m = Xw y t = -y * m):
      f(w) = 0.5 ||w||^2 + C * sum log(1 + exp(t))
      g(w) = w + C * X^T( -y * σ(t) )
      H(w) @ s = s + C * X^T( D * (X s) ), D_i = σ(t_i)*(1-σ(t_i))
    """

    def f(self, w: np.ndarray) -> float:
        m = self.backend.margin(w)                 # m = Xw
        t = -self.y * m                            # t = -y * m
        loss_sum = _softplus(t).sum()
        reg = 0.5 * float(w @ w)
        return reg + self.C * float(loss_sum)

    def grad(self, w: np.ndarray) -> np.ndarray:
        m = self.backend.margin(w)                 # m = Xw
        t = -self.y * m
        p = _expit(t)                              # σ(t)
        r = -self.y * p                            # tamaño l
        return w + self.C * self.backend.Xt_dot(r) # tamaño n

    def hess_vec(self, w: np.ndarray, s: np.ndarray) -> np.ndarray:
        m = self.backend.margin(w)                 # m = Xw
        t = -self.y * m
        p = _expit(t)                              # σ(t)
        D = p * (1.0 - p)                          # diag elementos (tamaño l)
        Xs = self.backend.X_dot(s)                 # tamaño l
        return s + self.C * self.backend.Xt_dot(D * Xs)


class L2SVMLoss(BaseLoss):
    """
    SVM L2 (hinge cuadrática) con regularización L2.
    Con m = Xw:
      f(w)   = 0.5 ||w||^2 + C * sum max(0, 1 - y*m)^2
      grad   = w + 2C * X^T( mask * (m - y) ), mask_i = [1 - y*m > 0]
      Hv(s)  = s + 2C * X^T( mask * (X s) )
    """

    def f(self, w: np.ndarray) -> float:
        m = self.backend.margin(w)                 # m = Xw
        z = 1.0 - self.y * m
        h = np.maximum(0.0, z)                     # hinge
        loss_sum = (h * h).sum()
        reg = 0.5 * float(w @ w)
        return reg + self.C * float(loss_sum)

    def grad(self, w: np.ndarray) -> np.ndarray:
        m = self.backend.margin(w)
        z = 1.0 - self.y * m
        mask = (z > 0.0).astype(np.float64)        # 0/1
        r = mask * (m - self.y)                    # tamaño l
        return w + (2.0 * self.C) * self.backend.Xt_dot(r)

    def hess_vec(self, w: np.ndarray, s: np.ndarray) -> np.ndarray:
        m = self.backend.margin(w)
        z = 1.0 - self.y * m
        mask = (z > 0.0).astype(np.float64)        # tamaño l
        Xs = self.backend.X_dot(s)                 # tamaño l
        return s + (2.0 * self.C) * self.backend.Xt_dot(mask * Xs)

