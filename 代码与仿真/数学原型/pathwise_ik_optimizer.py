"""Path-wise optimizer for a fixed Cartesian end-effector path."""
import numpy as np
from scipy.optimize import minimize


def optimize_joint_path(
    arm, targets, initial_path, *, limits=None, smoothness_weight=1.0,
    motion_weight=1e-3, max_iterations=300, endpoint_tolerance=2e-6,
    fixed_start=True, fixed_end=False, fixed_indices=None,
):
    """Optimize joint samples together without changing any Cartesian target."""
    targets = np.asarray(targets, dtype=float)
    q0 = np.asarray(initial_path, dtype=float)
    n = arm.joint_count
    if targets.ndim != 2 or targets.shape[1] != 2 or len(targets) < 3:
        raise ValueError("targets must have shape (N, 2), N >= 3")
    if q0.shape != (len(targets), n):
        raise ValueError("initial_path must have shape (N, joint_count)")
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(q0)):
        raise ValueError("targets and initial_path must be finite")
    if smoothness_weight < 0 or motion_weight < 0:
        raise ValueError("objective weights must be nonnegative")
    if limits is None:
        lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
    else:
        lower, upper = np.asarray(limits.lower), np.asarray(limits.upper)
        if lower.shape != (n,) or upper.shape != (n,) or np.any(lower >= upper):
            raise ValueError("invalid joint limits")
    if np.any(q0 < lower - 1e-10) or np.any(q0 > upper + 1e-10):
        raise ValueError("initial_path violates joint limits")
    count = len(targets)
    fixed_indices = set(int(i) for i in (fixed_indices or ()))
    if any(i < 0 or i >= count for i in fixed_indices):
        raise ValueError("fixed_indices contains an invalid path index")
    if fixed_start:
        fixed_indices.add(0)
    if fixed_end:
        fixed_indices.add(count - 1)
    for i in fixed_indices:
        error = np.linalg.norm(arm.forward_kinematics(q0[i])[-1] - targets[i])
        if error > endpoint_tolerance:
            raise ValueError("fixed endpoint joint configuration does not match its Cartesian target")

    def unpack(x):
        return np.asarray(x).reshape(count, n)

    def objective(x):
        q = unpack(x)
        d1, d2 = np.diff(q, axis=0), np.diff(q, n=2, axis=0)
        return smoothness_weight * float(np.sum(d2 * d2)) + motion_weight * float(np.sum(d1 * d1))

    def objective_jac(x):
        q = unpack(x)
        grad = np.zeros_like(q)
        d1 = np.diff(q, axis=0)
        grad[:-1] -= 2.0 * motion_weight * d1
        grad[1:] += 2.0 * motion_weight * d1
        d2 = np.diff(q, n=2, axis=0)
        grad[:-2] += 2.0 * smoothness_weight * d2
        grad[1:-1] -= 4.0 * smoothness_weight * d2
        grad[2:] += 2.0 * smoothness_weight * d2
        return grad.ravel()

    constrained_targets = [i for i in range(count) if i not in fixed_indices]

    def constraints(x):
        q = unpack(x)
        parts = [np.concatenate([
            arm.forward_kinematics(q[i])[-1] - targets[i]
            for i in constrained_targets
        ])]
        for i in sorted(fixed_indices):
            parts.append(q[i] - q0[i])
        return np.concatenate(parts)

    def constraints_jac(x):
        q = unpack(x)
        jac = np.zeros((2 * len(constrained_targets) + len(fixed_indices) * n, count * n))
        row = 0
        for i in constrained_targets:
            jac[row:row + 2, i * n:(i + 1) * n] = arm.jacobian(q[i])
            row += 2
        for i in sorted(fixed_indices):
            jac[row:row + n, i * n:(i + 1) * n] = np.eye(n)
            row += n
        return jac

    result = minimize(
        objective, q0.ravel(), method="SLSQP", jac=objective_jac,
        bounds=list(zip(np.tile(lower, count), np.tile(upper, count))),
        constraints=[{"type": "eq", "fun": constraints, "jac": constraints_jac}],
        options={"maxiter": int(max_iterations), "ftol": 1e-10, "disp": False},
    )
    q = unpack(result.x)
    errors = np.array([
        np.linalg.norm(arm.forward_kinematics(qi)[-1] - p)
        for qi, p in zip(q, targets)
    ])
    limits_ok = bool(np.all(q >= lower - 1e-8) and np.all(q <= upper + 1e-8))
    start_ok = not fixed_start or np.allclose(q[0], q0[0], atol=1e-7, rtol=0)
    end_ok = not fixed_end or np.allclose(q[-1], q0[-1], atol=1e-7, rtol=0)
    success = bool(result.success and np.max(errors) <= endpoint_tolerance
                   and limits_ok and start_ok and end_ok)
    return {
        "success": success, "optimizer_success": bool(result.success),
        "message": str(result.message), "joint_angles": q,
        "endpoint_errors": errors, "objective": float(result.fun),
        "iterations": int(result.nit),
    }
