"""将钢笔式几何路径转换为按弧长近似均匀分布的时间轨迹。"""

import numpy as np


def sample_path_by_arclength(anchors, sample_count=200, samples_per_segment=40, *, preserve_anchors=False):
    """采样锚点路径，返回按弧长近似均匀分布的二维点阵。

    anchors 中每项为 (point, incoming_handle, outgoing_handle)，控制柄为相对向量。
    """
    if sample_count < 2:
        raise ValueError("sample_count 必须至少为 2")
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment 必须至少为 2")
    if not anchors:
        raise ValueError("路径至少需要一个锚点")

    points = []
    for point, incoming, outgoing in anchors:
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("锚点必须是有限的二维坐标")
        points.append((point, None if incoming is None else np.asarray(incoming, dtype=float),
                       None if outgoing is None else np.asarray(outgoing, dtype=float)))

    if len(points) == 1:
        return np.repeat(points[0][0][None, :], sample_count, axis=0)

    dense = []
    for index in range(len(points) - 1):
        p0, _in0, out0 = points[index]
        p3, in3, _out3 = points[index + 1]
        if out0 is None and in3 is None:
            segment = np.linspace(p0, p3, samples_per_segment + 1)
        else:
            delta = p3 - p0
            p1 = p0 + out0 if out0 is not None else p0 + delta / 3.0
            p2 = p3 + in3 if in3 is not None else p3 - delta / 3.0
            t = np.linspace(0.0, 1.0, samples_per_segment + 1)[:, None]
            u = 1.0 - t
            segment = u**3 * p0 + 3 * u**2 * t * p1 + 3 * u * t**2 * p2 + t**3 * p3
        if index:
            segment = segment[1:]
        dense.extend(segment)

    dense = np.asarray(dense, dtype=float)
    distances = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    total_length = float(cumulative[-1])
    if total_length <= 1e-12:
        return np.repeat(dense[:1], sample_count, axis=0)

    target_distances = np.linspace(0.0, total_length, sample_count)
    if preserve_anchors:
        anchor_indices = np.arange(1, len(points) - 1) * samples_per_segment
        anchor_distances = cumulative[anchor_indices]
        target_distances = np.unique(np.concatenate((target_distances, anchor_distances)))
    sampled = np.column_stack([
        np.interp(target_distances, cumulative, dense[:, axis])
        for axis in range(2)
    ])
    sampled[0] = dense[0]
    sampled[-1] = dense[-1]
    return sampled


def find_sharp_corner_indices(anchors, sampled_points, angle_tolerance_degrees=1.0):
    """返回采样路径中方向不连续的内部锚点索引，不改变任何路径点。"""
    sampled = np.asarray(sampled_points, dtype=float)
    if sampled.ndim != 2 or sampled.shape[1] != 2 or len(sampled) < 2:
        raise ValueError("sampled_points 必须是至少包含两个二维点的数组")
    if not np.isfinite(angle_tolerance_degrees) or not 0 <= angle_tolerance_degrees < 180:
        raise ValueError("角度容差必须在 [0, 180) 度范围内")
    corners = []
    tolerance = np.cos(np.radians(angle_tolerance_degrees))
    for index in range(1, len(anchors) - 1):
        point, incoming, outgoing = anchors[index]
        previous = np.asarray(anchors[index - 1][0], dtype=float)
        current = np.asarray(point, dtype=float)
        following = np.asarray(anchors[index + 1][0], dtype=float)
        incoming_tangent = -np.asarray(incoming, dtype=float) if incoming is not None else current - previous
        outgoing_tangent = np.asarray(outgoing, dtype=float) if outgoing is not None else following - current
        incoming_norm = float(np.linalg.norm(incoming_tangent))
        outgoing_norm = float(np.linalg.norm(outgoing_tangent))
        if incoming_norm <= 1e-12 or outgoing_norm <= 1e-12:
            raise ValueError(f"路径锚点 {index + 1} 的切向量长度为零，无法可靠判断拐点")
        cosine = float(np.dot(incoming_tangent, outgoing_tangent) / (incoming_norm * outgoing_norm))
        if cosine < tolerance:
            sample_index = int(np.argmin(np.linalg.norm(sampled - current, axis=1)))
            if sample_index in (0, len(sampled) - 1):
                continue
            if np.linalg.norm(sampled[sample_index] - current) > 1e-8:
                raise ValueError(f"路径锚点 {index + 1} 未被精确保留在采样路径中")
            corners.append(sample_index)
    return tuple(sorted(set(corners)))


def parameterize_path(points, duration):
    """为路径采样点生成等间隔时间戳。"""
    points = np.asarray(points, dtype=float)
    duration = float(duration)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("points 必须是至少包含两个二维点的数组")
    if not np.all(np.isfinite(points)):
        raise ValueError("路径点必须为有限数值")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration 必须为正数")
    times = np.linspace(0.0, duration, len(points))
    return times, points.copy()
