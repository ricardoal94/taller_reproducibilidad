#src/distribuited_tron.py

#se hará primero el gradiente y luego el producto
#∇f(w) = w + C * Σ ( -y_i x_i / (1 + exp(y_i x_i^T w)) )
#H·v  = v + C * Σ ( D_i * x_i (x_i^T v) )

# src/tron.py
# -*- coding: utf-8 -*-
"""
TRON (Trust Region Newton) distribuido para Logistic Regression en PySpark.
- Optimiza: f(w) = 0.5 * ||w||^2 + C * sum_i log(1 + exp(-y_i x_i^T w))
- Gradiente y producto Hessiano-vector se calculan de forma distribuida con mapPartitions.
- Usa broadcast(w) y broadcast(v) para reducir comunicación.
- Coalesce previo al reduce para disminuir mensajes al driver.
- CG de Steihaug para respetar la región de confianza (trust region).
"""

import os
import math
import numpy as np

from pyspark import StorageLevel
from pyspark.mllib.linalg import SparseVector, DenseVector

from data_loader import  get_spark_context, load_svm_data



def _is_sparse(x): #verifica si x es un SparceVector, se usa para ahorrar memoria y tiempo
    return isinstance(x, SparseVector)

def _is_dense(x): #detecta si x es un vector DenseVector, para operaciones sobre vectores completos
    return isinstance(x, DenseVector) or isinstance(x, np.ndarray) or isinstance(x, list) or isinstance(x, tuple)

def _sparse_dot(x: SparseVector, w: np.ndarray) -> float: #producto punto de un vector denso y uno disperso
    # w es denso
    idx = x.indices
    val = x.values
    return float(np.dot(w[idx], val)) #multiplica solo los elementos de w correspondientes a los índices no nulos de x.

def _sparse_axpy(alpha: float, x: SparseVector, y: np.ndarray): #operacion tiempo AXPY
    # y += alpha * x
    if alpha == 0.0:
        return
    idx = x.indices
    val = x.values
    y[idx] += alpha * val #esto se realiza dentro de cada partición del RDD usando mappartitions

def _dense_dot(x: DenseVector, w: np.ndarray) -> float: #calcula el producto punto entre dos vectores densos.
    return float(np.dot(np.asarray(x), w))

def _dense_axpy(alpha: float, x: DenseVector, y: np.ndarray): # ealiza la operación y←y+αx para vectores densos.
    if alpha == 0.0:
        return
    y += alpha * np.asarray(x)



def _partition_obj_grad_hv(iterator, w_b, v_b, C, dim):
    """
    Calcula la contribución de cada partición al valor objetivo, gradiente y Hv.
    Incluye manejo robusto de errores y validación de dimensiones.
    """
    import numpy as _np
    import math, sys, traceback

    w = w_b.value
    v = v_b.value if v_b is not None else None

    grad_loc = _np.zeros(dim, dtype=_np.float64)
    hv_loc = _np.zeros(dim, dtype=_np.float64) if v is not None else None
    loss_sum = 0.0
    bad_count = 0

    for i, item in enumerate(iterator):
        try:
            if item is None:
                bad_count += 1
                continue
            y, x = item

            # --- validaciones básicas ---
            if y not in (1.0, -1.0):
                y = 1.0 if y > 0 else -1.0  # normaliza etiquetas
            if hasattr(x, "size"):
                xdim = x.size
            elif hasattr(x, "indices"):
                xdim = int(max(x.indices) + 1) if len(x.indices) > 0 else 0
            else:
                bad_count += 1
                continue

            if xdim != dim:
                # vector con dimensión inconsistente
                # lo recortamos o extendemos a tamaño dim
                if _is_sparse(x):
                    indices = [idx for idx in x.indices if idx < dim]
                    values = x.values[:len(indices)]
                    x = SparseVector(dim, indices, values)
                else:
                    xv = _np.asarray(x)
                    if len(xv) < dim:
                        xv = _np.pad(xv, (0, dim - len(xv)))
                    else:
                        xv = xv[:dim]
                    x = DenseVector(xv)

            # --- producto punto x·w ---
            if _is_sparse(x):
                dot_w = _sparse_dot(x, w)
            else:
                dot_w = _dense_dot(x, w)

            yz = y * dot_w

            # Evita overflow en exp()
            if yz > 0:
                loss_i = math.log1p(math.exp(-yz))
            else:
                loss_i = (-yz) + math.log1p(math.exp(yz))
            loss_sum += loss_i

            exp_m = math.exp(-yz)
            inv = 1.0 / (1.0 + exp_m)
            one_minus_sigma = 1.0 - inv
            coeff_g = -y * one_minus_sigma

            if _is_sparse(x):
                _sparse_axpy(coeff_g, x, grad_loc)
            else:
                _dense_axpy(coeff_g, x, grad_loc)

            if v is not None:
                D_i = exp_m / ((1.0 + exp_m) ** 2)
                if _is_sparse(x):
                    dot_v = _sparse_dot(x, v) # type: ignore
                    _sparse_axpy(D_i * dot_v, x, hv_loc) # type: ignore
                else:
                    dot_v = _dense_dot(x, v) # type: ignore
                    _dense_axpy(D_i * dot_v, x, hv_loc) # type: ignore

        except Exception as e:
            bad_count += 1
            print(f"[WorkerError] {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()

    if bad_count > 0:
        print(f"[WARN] {bad_count} instancias ignoradas en esta partición.", file=sys.stderr)

    if v is not None:
        yield (loss_sum, grad_loc, hv_loc)
    else:
        yield (loss_sum, grad_loc)


def _aggregate_sums(rdd, num_parts=None):
    """Reduce de arrays/escalares sumando."""
    mapped = rdd
    if num_parts is not None:
        # Reducimos #particiones antes del reduce para bajar costo de comunicación
        mapped = mapped.coalesce(num_parts)
    return mapped.reduce(lambda a, b: tuple(x + y for x, y in zip(a, b)))


def compute_obj_grad(data_rdd, w_b, C, dim, coalesce_parts=None):
    """
    Devuelve:
      f(w) = 0.5 ||w||^2 + C * sum_i log(1 + exp(-y_i x_i^T w))
      grad(w) = w + C * sum_i -(y_i(1 - sigma)) x_i
    """
    parts = data_rdd.mapPartitions(lambda it: _partition_obj_grad_hv(it, w_b, None, C, dim))
    loss_sum, grad_sum = _aggregate_sums(parts, num_parts=coalesce_parts)
    w = w_b.value
    f = 0.5 * float(np.dot(w, w)) + C * float(loss_sum)
    g = w + C * grad_sum
    return f, g


def hessian_vector_product(data_rdd, w_b, v, C, dim, coalesce_parts=None):
    """Devuelve H(w)·v + término identidad (incluido en el acumulado)."""
    v_b = sc.broadcast(v)
    parts = data_rdd.mapPartitions(lambda it: _partition_obj_grad_hv(it, w_b, v_b, C, dim))
    # ignoramos loss y grad de esta pasada (ya se calcularon cuando toque), nos interesa hv
    # sumamos las tuplas: (loss, grad, hv)
    loss_sum, grad_sum, hv_sum = _aggregate_sums(parts, num_parts=coalesce_parts)
    v_b.unpersist(blocking=False)
    # H = I + C * X^T D X  => H v = v + C * hv_sum
    return v + C * hv_sum


# Conjugate Gradient (Steihaug) con trust-region

def cg_steihaug(hvp_func, g, delta, tol=1e-3, max_iter=250):
    """
    Resuelve aprox: min  q(d) = g^T d + 0.5 d^T H d  s.a. ||d|| <= delta
    mediante CG de Steihaug. Devuelve d y (aprox) d^T H d para el modelo.
    """
    n = g.shape[0]
    d = np.zeros(n, dtype=np.float64)
    r = -g.copy()
    p = r.copy()
    rTr = float(np.dot(r, r))

    if math.sqrt(rTr) < tol:
        # d=0
        Hd_d = 0.0
        return d, Hd_d

    for _ in range(max_iter):
        Hp = hvp_func(p)  # H p
        pHp = float(np.dot(p, Hp))

        # Si detecta indefinida (curvatura no positiva), proyecta a la frontera
        if pHp <= 1e-16:
            # Encuentra tau tal que ||d + tau p|| = delta
            tau = _tau_to_boundary(d, p, delta)
            d = d + tau * p
            Hd_d = float(np.dot(d, hvp_func(d)))
            return d, Hd_d

        alpha = rTr / pHp
        d_next = d + alpha * p

        if np.linalg.norm(d_next) >= delta:
            tau = _tau_to_boundary(d, p, delta)
            d = d + tau * p
            Hd_d = float(np.dot(d, hvp_func(d)))
            return d, Hd_d

        d = d_next
        r = r - alpha * Hp
        rTr_new = float(np.dot(r, r))

        if math.sqrt(rTr_new) < tol:
            Hd_d = float(np.dot(d, hvp_func(d)))
            return d, Hd_d

        beta = rTr_new / rTr
        p = r + beta * p
        rTr = rTr_new

    Hd_d = float(np.dot(d, hvp_func(d)))
    return d, Hd_d


def _tau_to_boundary(d, p, delta):
    """Coeficiente tau >= 0 tal que ||d + tau p|| = delta."""
    # Resolver ||d + tau p||^2 = delta^2 => (p·p) tau^2 + 2(d·p) tau + (d·d - delta^2) = 0
    pTp = float(np.dot(p, p))
    dTp = float(np.dot(d, p))
    dTd = float(np.dot(d, d))
    A = pTp
    B = 2.0 * dTp
    C = dTd - delta * delta
    # Elegir la raíz positiva
    disc = max(0.0, B * B - 4.0 * A * C)
    tau = (-B + math.sqrt(disc)) / (2.0 * A)
    return max(0.0, tau)


# TRON principal

def tron(
    data_rdd,
    dim,
    C=1.0,
    max_outer_iter=20,
    tol_grad=1e-4,
    cg_tol=1e-3,
    cg_max_iter=250,
    delta0=None,
    eta=0.1,
    verbose=True,
    coalesce_parts=None
):
    """
    Implementación simplificada de TRON para logistic regression.

    Parámetros clave:
    - delta0: radio inicial de la trust region (por defecto, ||g||)
    - eta: umbral de aceptación del paso (0.1 típico)
    - coalesce_parts: int o None => reduce particiones antes de los reduce (recomendado: #nodos)
    """
    # Cachear datos (útil para múltiples pasadas por iteración)
    data_rdd.persist(StorageLevel.MEMORY_ONLY)

    w = np.zeros(dim, dtype=np.float64)
    w_b = sc.broadcast(w)

    f, g = compute_obj_grad(data_rdd, w_b, C, dim, coalesce_parts)
    gnorm = float(np.linalg.norm(g))
    if delta0 is None:
        delta = max(1.0, gnorm)
    else:
        delta = float(delta0)

    if verbose:
        print(f"[TRON] iter=0 f={f:.6f} ||g||={gnorm:.6e} delta={delta:.3f}")

    for t in range(1, max_outer_iter + 1):
        # Define función Hvp(w, .) sobre el w actual
        def hvp(v):
            return hessian_vector_product(data_rdd, w_b, v, C, dim, coalesce_parts)

        # Resuelve subproblema de TR mediante CG de Steihaug
        d, Hd_d = cg_steihaug(hvp, g, delta, tol=cg_tol, max_iter=cg_max_iter)

        # Modelo cuadrático: q(d) = g^T d + 0.5 d^T H d
        gTd = float(np.dot(g, d))
        qd = gTd + 0.5 * Hd_d  # predicción de reducción: -q(d)

        # Evalúa f(w + d)
        w_new = w + d
        w_new_b = sc.broadcast(w_new)
        f_new, g_new = compute_obj_grad(data_rdd, w_new_b, C, dim, coalesce_parts)

        # Ratio de reducción
        ared = f - f_new
        pred = -qd
        if pred <= 0:
            rho = -np.inf  # Modelo no es de confianza
        else:
            rho = ared / pred

        # Actualiza radio de la trust region
        if rho < 0.25:
            delta *= 0.25
        elif rho > 0.75 and abs(np.linalg.norm(d) - delta) < 1e-10:
            delta = min(2.0 * delta, 1e8)

        # Acepta o rechaza
        if rho > eta:
            # aceptar
            w = w_new
            f = f_new
            g = g_new
            # Cambiar broadcast
            w_b.unpersist(blocking=False)
            w_b = w_new_b
        else:
            # rechazar
            w_new_b.unpersist(blocking=False)

        gnorm = float(np.linalg.norm(g))

        if verbose:
            print(f"[TRON] iter={t} f={f:.6f} ||g||={gnorm:.6e} "
                  f"rho={rho:.4f} ared={ared:.6e} pred={pred:.6e} delta={delta:.3f}")

        if gnorm < tol_grad:
            if verbose:
                print("[TRON] Convergencia por norma de gradiente.")
            break

    data_rdd.unpersist()
    w_b.unpersist(blocking=False)
    return w

def predict_labels(data_rdd, w):
    """
    Devuelve (accuracy, total, correctos). Predicción por signo(x·w).
    """
    def part(iterator):
        w_loc = w
        correct = 0
        total = 0
        for y, x in iterator:
            if _is_sparse(x):
                s = _sparse_dot(x, w_loc)
            else:
                s = _dense_dot(x, w_loc)
            yhat = 1.0 if s >= 0.0 else -1.0
            correct += 1 if yhat == y else 0
            total += 1
        yield (correct, total)

    counts = data_rdd.mapPartitions(part).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    correct, total = counts
    acc = (correct / total) if total > 0 else 0.0
    return acc, total, correct


if __name__ == "__main__":
    import argparse
    sc = get_spark_context()

    parser = argparse.ArgumentParser(description="Distributed TRON (LogReg) with PySpark")
    parser.add_argument("--data", type=str, default="../dataset/webspam_wc_normalized_unigram.svm",
                        help="Ruta al dataset en formato SVMlight/libSVM")
    parser.add_argument("--C", type=float, default=1.0, help="Regularización C")
    parser.add_argument("--max_outer_iter", type=int, default=20, help="Máx iteraciones externas de TRON")
    parser.add_argument("--tol_grad", type=float, default=1e-4, help="Tolerancia de norma de gradiente")
    parser.add_argument("--cg_tol", type=float, default=1e-3, help="Tolerancia de CG")
    parser.add_argument("--cg_max_iter", type=int, default=250, help="Máx iteraciones de CG")
    parser.add_argument("--delta0", type=float, default=None, help="Radio inicial trust region")
    parser.add_argument("--eta", type=float, default=0.1, help="Umbral de aceptación TRON")
    parser.add_argument("--partitions", type=int, default=32, help="# particiones para cargar el RDD")
    parser.add_argument("--coalesce", type=int, default=None,
                        help="Coalesce a este # de particiones antes de reduce (recomendado: #nodos)")
    parser.add_argument("--save", type=str, default="../models/model.npy", help="Ruta para guardar pesos .npy")
    parser.add_argument("--eval", action="store_true", help="Evalúa accuracy al final (en el mismo dataset)")
    args = parser.parse_args()

    data_rdd = load_svm_data(args.data, partitions=args.partitions)
    data_rdd.persist(StorageLevel.MEMORY_ONLY)
    first = data_rdd.first()
    dim = data_rdd.map(lambda x: x[1].size).max()  # type: ignore # tamaño del vector
    print(f"[INFO] Dimensión máxima detectada: {dim}")
    print(f"[INFO] Dataset cargado: {data_rdd.count()} instancias, dim={dim}, partitions={data_rdd.getNumPartitions()}")

    # Entrenar
    w = tron(
        data_rdd=data_rdd,
        dim=dim,
        C=args.C,
        max_outer_iter=args.max_outer_iter,
        tol_grad=args.tol_grad,
        cg_tol=args.cg_tol,
        cg_max_iter=args.cg_max_iter,
        delta0=args.delta0,
        eta=args.eta,
        verbose=True,
        coalesce_parts=args.coalesce
    )

    # Guardar
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    np.save(args.save, w)
    print(f"[INFO] Modelo guardado en: {args.save}")

    # Evaluar 
    if args.eval:
        acc, total, correct = predict_labels(data_rdd, w)
        print(f"[EVAL] Accuracy={acc*100:.2f}%  ({correct}/{total})")

    data_rdd.unpersist()
