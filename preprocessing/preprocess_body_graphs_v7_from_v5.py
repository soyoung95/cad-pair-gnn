
from __future__ import annotations


import argparse
import json
import math
from pathlib import Path
from typing import Any


import torch
from torch_geometric.data import Data
from tqdm import tqdm


# v7 keeps compatibility with train_gat_cross_attention_v2.py:
# - surface type one-hot: face dim 0-6
# - log_area: dim 7
# - mean_normal_xyz: dim 15-17
# This is important because GeometricCompatibility uses dim 7 and dims 15-17.
FACE_FEATURE_DIM = 34
EDGE_FEATURE_DIM = 37
EPS = 1e-8


FACE_SCHEMA = [
    "surface_type_oh7_dim0_6",
    "log_area_dim7",
    "reversed_dim8",
    "log_radius_dim9",
    "valid_ratio_from_trimming_mask_dim10",
    "normal_concentration_dim11",
    "mean_point_xyz_as_centroid_proxy_dim12_14",
    "mean_normal_xyz_dim15_17",
    "point_planarity_dim18",
    "log_spread_mean_dim19",
    "log_spread_max_dim20",
    "bbox_center_xyz_dim21_23",
    "log_bbox_extent_xyz_dim24_26",
    "principal_axis_xyz_from_points_dim27_29",
    "curvature_proxy_from_radius_dim30",
    "log_valid_point_count_dim31",
    "bbox_volume_proxy_dim32",
    "mask_missing_flag_dim33",
]


EDGE_SCHEMA = [
    "curve_type_oh7_dim0_6",
    "reversed_dim7",
    "log_length_dim8",
    "convexity_oh4_dim9_12",
    "log_radius_dim13",
    "log_edge_spread_mean_dim14",
    "mean_tangent_xyz_dim15_17",
    "tangent_concentration_dim18",
    "mean_edge_point_xyz_dim19_21",
    "edge_start_point_xyz_dim22_24",
    "edge_end_point_xyz_dim25_27",
    "edge_midpoint_xyz_dim28_30",
    "edge_direction_xyz_dim31_33",
    "log_chord_length_dim34",
    "curvature_proxy_from_radius_dim35",
    "log_edge_point_count_dim36",
]




def torch_load(path: Path) -> Data:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")




def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)




def nan_to_num(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)




def normalize_vec(v: torch.Tensor) -> torch.Tensor:
    if v.numel() == 0:
        return torch.zeros(3, dtype=torch.float32)
    v = v.float().flatten()[:3]
    n = float(v.norm().item())
    if n <= EPS:
        return torch.zeros(3, dtype=torch.float32)
    return v / n




def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.reshape(0, 3).float()
    return torch.nn.functional.normalize(x.float(), dim=-1, eps=EPS)




def log1p_nonnegative(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return float(math.log1p(max(float(value), 0.0)))




def curvature_proxy_from_log_radius(log_radius: float) -> float:
    # Existing radius feature is log(1 + radius). If radius is unknown or 0,
    # keep curvature proxy at 0 rather than injecting unstable large values.
    if not math.isfinite(float(log_radius)) or float(log_radius) <= 0.0:
        return 0.0
    radius = math.expm1(float(log_radius))
    if radius <= EPS:
        return 0.0
    return log1p_nonnegative(1.0 / radius)




def to_xyz_tensor(value: Any) -> torch.Tensor:
    if value is None:
        return torch.zeros((0, 3), dtype=torch.float32)


    if isinstance(value, torch.Tensor):
        t = value.detach().cpu().float()
        if t.numel() == 0:
            return torch.zeros((0, 3), dtype=torch.float32)
        if t.dim() == 1:
            if t.numel() >= 3:
                return nan_to_num(t[:3].reshape(1, 3))
            return torch.zeros((0, 3), dtype=torch.float32)
        t = t.reshape(-1, t.shape[-1])
        if t.shape[1] >= 3:
            return nan_to_num(t[:, :3])
        return torch.zeros((0, 3), dtype=torch.float32)


    if isinstance(value, dict):
        if all(k in value for k in ("x", "y", "z")):
            try:
                return torch.tensor([[float(value["x"]), float(value["y"]), float(value["z"])]], dtype=torch.float32)
            except Exception:
                return torch.zeros((0, 3), dtype=torch.float32)
        if all(k in value for k in ("X", "Y", "Z")):
            try:
                return torch.tensor([[float(value["X"]), float(value["Y"]), float(value["Z"])]], dtype=torch.float32)
            except Exception:
                return torch.zeros((0, 3), dtype=torch.float32)
        for key in ("data", "values", "points", "normals", "tangents", "xyz"):
            if key in value:
                return to_xyz_tensor(value[key])
        return torch.zeros((0, 3), dtype=torch.float32)


    if isinstance(value, (list, tuple)):
        rows: list[list[float]] = []


        def visit(obj: Any) -> None:
            if isinstance(obj, torch.Tensor):
                rows.extend(to_xyz_tensor(obj).tolist())
            elif isinstance(obj, dict):
                if all(k in obj for k in ("x", "y", "z")):
                    try:
                        rows.append([float(obj["x"]), float(obj["y"]), float(obj["z"])])
                    except Exception:
                        pass
                elif all(k in obj for k in ("X", "Y", "Z")):
                    try:
                        rows.append([float(obj["X"]), float(obj["Y"]), float(obj["Z"])])
                    except Exception:
                        pass
                else:
                    for v in obj.values():
                        visit(v)
            elif isinstance(obj, (list, tuple)):
                if len(obj) >= 3 and all(isinstance(v, (int, float)) for v in obj[:3]):
                    try:
                        rows.append([float(obj[0]), float(obj[1]), float(obj[2])])
                    except Exception:
                        pass
                else:
                    for v in obj:
                        visit(v)


        visit(value)
        if not rows:
            return torch.zeros((0, 3), dtype=torch.float32)
        return nan_to_num(torch.tensor(rows, dtype=torch.float32)[:, :3])


    return torch.zeros((0, 3), dtype=torch.float32)




def to_mask_tensor(value: Any, n_points: int) -> torch.Tensor:
    if n_points <= 0:
        return torch.zeros((0,), dtype=torch.bool)
    if value is None:
        return torch.ones((n_points,), dtype=torch.bool)
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu().flatten().float()
        if t.numel() == 0:
            return torch.ones((n_points,), dtype=torch.bool)
        m = t > 0.5
    else:
        vals: list[float] = []


        def visit(obj: Any) -> None:
            if isinstance(obj, dict):
                for k in ("data", "values", "mask"):
                    if k in obj:
                        visit(obj[k])
                        return
                for v in obj.values():
                    visit(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    visit(v)
            elif isinstance(obj, (bool, int, float)):
                vals.append(float(obj))


        visit(value)
        if not vals:
            return torch.ones((n_points,), dtype=torch.bool)
        m = torch.tensor(vals, dtype=torch.float32).flatten() > 0.5


    if m.numel() >= n_points:
        return m[:n_points]
    out = torch.ones((n_points,), dtype=torch.bool)
    out[: m.numel()] = m
    return out




def split_items(value: Any, n_items: int) -> list[Any]:
    if n_items <= 0:
        return []
    if value is None:
        return [None for _ in range(n_items)]
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu()
        if t.dim() >= 3 and t.shape[0] == n_items:
            return [t[i] for i in range(n_items)]
        if t.dim() == 2 and t.shape[0] == n_items and t.shape[1] == 3:
            return [t[i].reshape(1, 3) for i in range(n_items)]
        if n_items == 1:
            return [t]
        return [None for _ in range(n_items)]
    if isinstance(value, (list, tuple)):
        if len(value) == n_items:
            return list(value)
        if n_items == 1:
            return [value]
        out = list(value[:n_items])
        out.extend([None] * (n_items - len(out)))
        return out
    return [value if n_items == 1 else None for _ in range(n_items)]




def mean_xyz(points: torch.Tensor) -> list[float]:
    if points.size(0) == 0:
        return [0.0, 0.0, 0.0]
    m = points.float().mean(dim=0)
    return [float(v) for v in m[:3].tolist()]




def bbox_center_and_log_extent(points: torch.Tensor) -> tuple[list[float], list[float], float]:
    if points.size(0) == 0:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0
    p_min = points.min(dim=0).values
    p_max = points.max(dim=0).values
    center = (p_min + p_max) * 0.5
    extent = (p_max - p_min).clamp(min=0.0)
    log_extent = torch.log1p(extent)
    volume_proxy = log1p_nonnegative(float(extent.prod().item()))
    return [float(v) for v in center[:3].tolist()], [float(v) for v in log_extent[:3].tolist()], volume_proxy




def spread_stats(points: torch.Tensor) -> tuple[float, float]:
    if points.size(0) == 0:
        return 0.0, 0.0
    extent = points.max(dim=0).values - points.min(dim=0).values
    return log1p_nonnegative(float(extent.mean().item())), log1p_nonnegative(float(extent.max().item()))




def point_planarity(points: torch.Tensor) -> float:
    if points.size(0) < 3:
        return 0.0
    centered = points - points.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(points.size(0) - 1, 1)
    try:
        eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    except Exception:
        return 0.0
    denom = float(eigvals.sum().item())
    if denom <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(eigvals[0].item() / denom)))




def principal_axis(points: torch.Tensor) -> list[float]:
    # Approximate axis from the longest spread direction of sampled face points.
    # This is not the exact B-Rep cylinder/cone axis, but it gives a useful
    # orientation cue when exact axis is not available in the v5 graph.
    if points.size(0) < 3:
        return [0.0, 0.0, 0.0]
    centered = points - points.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(points.size(0) - 1, 1)
    try:
        eigvals, eigvecs = torch.linalg.eigh(cov)
    except Exception:
        return [0.0, 0.0, 0.0]
    axis = eigvecs[:, int(torch.argmax(eigvals).item())]
    axis = normalize_vec(axis)
    return [float(v) for v in axis.tolist()]




def mean_unit_and_concentration(vectors: torch.Tensor) -> tuple[list[float], float]:
    if vectors.size(0) == 0:
        return [0.0, 0.0, 0.0], 0.0
    unit = normalize_rows(vectors)
    mean_vec = unit.mean(dim=0)
    concentration = float(mean_vec.norm().item())
    if concentration > EPS:
        mean_unit = (mean_vec / mean_vec.norm()).tolist()
    else:
        mean_unit = [0.0, 0.0, 0.0]
    return [float(v) for v in mean_unit[:3]], float(max(0.0, min(1.0, concentration)))




def face_stats(points_value: Any, normals_value: Any, mask_value: Any, log_radius: float) -> list[float]:
    points = to_xyz_tensor(points_value)
    normals = to_xyz_tensor(normals_value)
    mask = to_mask_tensor(mask_value, points.size(0))


    mask_missing_flag = 1.0 if mask_value is None else 0.0


    if points.size(0) > 0 and mask.numel() == points.size(0):
        valid_points = points[mask]
        valid_ratio = float(mask.float().mean().item())
    else:
        valid_points = points
        valid_ratio = 0.0 if points.size(0) == 0 else 1.0


    if normals.size(0) > 0 and mask.numel() == normals.size(0):
        valid_normals = normals[mask]
    else:
        valid_normals = normals


    normal_mean, normal_conc = mean_unit_and_concentration(valid_normals)
    point_mean = mean_xyz(valid_points)
    planarity = point_planarity(valid_points)
    spread_mean, spread_max = spread_stats(valid_points)
    bbox_center, log_bbox_extent, bbox_volume = bbox_center_and_log_extent(valid_points)
    axis = principal_axis(valid_points)
    curvature_proxy = curvature_proxy_from_log_radius(log_radius)
    log_valid_count = log1p_nonnegative(float(valid_points.size(0)))


    return [
        valid_ratio,
        normal_conc,
        *point_mean,
        *normal_mean,
        planarity,
        spread_mean,
        spread_max,
        *bbox_center,
        *log_bbox_extent,
        *axis,
        curvature_proxy,
        log_valid_count,
        bbox_volume,
        mask_missing_flag,
    ]




def ordered_edge_endpoints(points: torch.Tensor, tangent_mean: list[float]) -> tuple[list[float], list[float], list[float], list[float], float]:
    if points.size(0) == 0:
        z = [0.0, 0.0, 0.0]
        return z, z, z, z, 0.0


    if points.size(0) == 1:
        p = [float(v) for v in points[0, :3].tolist()]
        z = [0.0, 0.0, 0.0]
        return p, p, p, z, 0.0


    tangent = torch.tensor(tangent_mean, dtype=torch.float32)
    if float(tangent.norm().item()) <= EPS:
        # Fall back to longest point spread direction.
        axis = torch.tensor(principal_axis(points), dtype=torch.float32)
    else:
        axis = normalize_vec(tangent)


    if float(axis.norm().item()) <= EPS:
        start = points[0]
        end = points[-1]
    else:
        proj = points @ axis
        start = points[int(torch.argmin(proj).item())]
        end = points[int(torch.argmax(proj).item())]


    midpoint = (start + end) * 0.5
    direction = normalize_vec(end - start)
    chord = float((end - start).norm().item())


    return (
        [float(v) for v in start[:3].tolist()],
        [float(v) for v in end[:3].tolist()],
        [float(v) for v in midpoint[:3].tolist()],
        [float(v) for v in direction[:3].tolist()],
        log1p_nonnegative(chord),
    )




def edge_entity_stats(points_value: Any, tangents_value: Any, log_radius: float) -> list[float]:
    points = to_xyz_tensor(points_value)
    tangents = to_xyz_tensor(tangents_value)


    spread_mean, _ = spread_stats(points)
    tangent_mean, tangent_conc = mean_unit_and_concentration(tangents)
    point_mean = mean_xyz(points)
    start, end, midpoint, direction, chord_log = ordered_edge_endpoints(points, tangent_mean)
    curvature_proxy = curvature_proxy_from_log_radius(log_radius)
    point_count_log = log1p_nonnegative(float(points.size(0)))


    return [
        spread_mean,
        *tangent_mean,
        tangent_conc,
        *point_mean,
        *start,
        *end,
        *midpoint,
        *direction,
        chord_log,
        curvature_proxy,
        point_count_log,
    ]




def make_v7_face_base(base_x: torch.Tensor) -> torch.Tensor:
    # v5_slim face layout is expected to be:
    # 0-6 surface type, 7 reversed, 8 log_area, 9 log_radius.
    # v7 reorders to match GeometricCompatibility:
    # 0-6 surface type, 7 log_area, 8 reversed, 9 log_radius.
    out = torch.zeros((base_x.shape[0], 10), dtype=torch.float32)
    out[:, 0:7] = base_x[:, 0:7]
    if base_x.shape[1] > 8:
        out[:, 7] = base_x[:, 8]  # log_area
    if base_x.shape[1] > 7:
        out[:, 8] = base_x[:, 7]  # reversed
    if base_x.shape[1] > 9:
        out[:, 9] = base_x[:, 9]  # log_radius
    return out




def convert_graph(g: Data) -> Data:
    if not hasattr(g, "x") or not hasattr(g, "edge_index") or not hasattr(g, "edge_attr"):
        raise ValueError("Input graph must contain x, edge_index, and edge_attr.")


    base_x = nan_to_num(g.x)
    base_e = nan_to_num(g.edge_attr)
    n_faces = int(base_x.shape[0])
    n_edges = int(base_e.shape[0])


    if base_x.dim() != 2 or base_x.shape[1] < 10:
        raise ValueError(f"Expected face feature dim >= 10, got {tuple(base_x.shape)}")
    if base_e.dim() != 2 or base_e.shape[1] < 14:
        raise ValueError(f"Expected edge feature dim >= 14, got {tuple(base_e.shape)}")


    face_points = split_items(getattr(g, "face_points", None), n_faces)
    face_normals = split_items(getattr(g, "face_normals", None), n_faces)
    face_masks = split_items(getattr(g, "face_trimming_mask", None), n_faces)


    face_base = make_v7_face_base(base_x)
    face_log_radii = face_base[:, 9].tolist()
    face_extra = [
        face_stats(p, n, m, float(log_r))
        for p, n, m, log_r in zip(face_points, face_normals, face_masks, face_log_radii)
    ]
    x = torch.cat([face_base, torch.tensor(face_extra, dtype=torch.float32)], dim=1)


    edge_entity_points = getattr(g, "edge_entity_points", None)
    edge_entity_tangents = getattr(g, "edge_entity_tangents", None)
    edge_to_entity_idx = getattr(g, "edge_to_entity_idx", None)


    if edge_to_entity_idx is None or n_edges == 0:
        edge_extra = torch.zeros((n_edges, EDGE_FEATURE_DIM - 14), dtype=torch.float32)
    else:
        eidx = torch.as_tensor(edge_to_entity_idx, dtype=torch.long).flatten()
        if eidx.numel() != n_edges:
            edge_extra = torch.zeros((n_edges, EDGE_FEATURE_DIM - 14), dtype=torch.float32)
        else:
            n_entities = int(eidx.max().item() + 1) if eidx.numel() > 0 else 0
            pts_items = split_items(edge_entity_points, n_entities)
            tan_items = split_items(edge_entity_tangents, n_entities)


            # edge_attr is per directed edge, while edge entities are often shared by
            # two directed edges. Use the first directed edge radius for each entity.
            entity_log_radius = [0.0] * n_entities
            if n_entities > 0 and base_e.shape[0] == eidx.numel() and base_e.shape[1] > 13:
                for directed_edge_i, entity_i in enumerate(eidx.tolist()):
                    if 0 <= entity_i < n_entities and entity_log_radius[entity_i] == 0.0:
                        entity_log_radius[entity_i] = float(base_e[directed_edge_i, 13].item())


            entity_extra = torch.tensor(
                [edge_entity_stats(p, t, entity_log_radius[i]) for i, (p, t) in enumerate(zip(pts_items, tan_items))],
                dtype=torch.float32,
            ) if n_entities > 0 else torch.zeros((0, EDGE_FEATURE_DIM - 14), dtype=torch.float32)
            if entity_extra.size(0) == 0:
                edge_extra = torch.zeros((n_edges, EDGE_FEATURE_DIM - 14), dtype=torch.float32)
            else:
                idx = eidx.clamp(0, entity_extra.size(0) - 1)
                edge_extra = entity_extra[idx]


    edge_attr = torch.cat([base_e[:, :14], edge_extra], dim=1)


    if x.shape[1] != FACE_FEATURE_DIM:
        raise RuntimeError(f"Face dim mismatch: {x.shape[1]}")
    if edge_attr.shape[1] != EDGE_FEATURE_DIM:
        raise RuntimeError(f"Edge dim mismatch: {edge_attr.shape[1]}")


    out = Data(
        x=nan_to_num(x),
        edge_index=g.edge_index.long(),
        edge_attr=nan_to_num(edge_attr),
        body_id=getattr(g, "body_id", ""),
        feature_schema={
            "source": "v5_slim_pt",
            "version": "v7_from_v5",
            "note": "v7 adds centroid/axis/start/end approximations from sampled points. Exact B-Rep origin/axis/curvature require raw CAD fields if available.",
            "face_dim": FACE_FEATURE_DIM,
            "edge_dim": EDGE_FEATURE_DIM,
            "face": FACE_SCHEMA,
            "edge": EDGE_SCHEMA,
        },
    )
    return out




def main() -> None:
    parser = argparse.ArgumentParser(description="Build v7 enriched fixed B-Rep features from v5_slim .pt graphs.")
    parser.add_argument("--input_dir", type=str, default="processed/body_graphs_v5_slim")
    parser.add_argument("--output_dir", type=str, default="processed/body_graphs_v7_full")
    parser.add_argument("--report_path", type=str, default="outputs/reports/pyg_preprocess_v7_from_v5_summary.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()


    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)


    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")


    files = sorted(input_dir.glob("*.pt"))
    if args.limit is not None:
        files = files[: args.limit]


    output_dir.mkdir(parents=True, exist_ok=True)


    processed = 0
    skipped = 0
    failed = 0
    failed_examples: list[dict[str, str]] = []


    face_nodes = 0
    face_extra_nonzero = 0
    edge_rows = 0
    edge_extra_nonzero = 0


    print("=" * 80)
    print("[v7] Build enriched fixed B-Rep features from v5_slim graphs")
    print(f"  Input    : {input_dir}")
    print(f"  Output   : {output_dir}")
    print(f"  Face x   : {FACE_FEATURE_DIM} dims")
    print(f"  Edge attr: {EDGE_FEATURE_DIM} dims")
    print("  Important: dim7=log_area, dims15-17=mean_normal_xyz")
    print("=" * 80)


    for src in tqdm(files, total=len(files)):
        dst = output_dir / src.name
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            g = torch_load(src)
            out = convert_graph(g)
            torch.save(out, dst)
            processed += 1


            face_nodes += int(out.x.shape[0])
            edge_rows += int(out.edge_attr.shape[0])
            if out.x.shape[0] > 0:
                face_extra_nonzero += int((out.x[:, 10:FACE_FEATURE_DIM].abs().sum(dim=1) > 1e-6).sum().item())
            if out.edge_attr.shape[0] > 0:
                edge_extra_nonzero += int((out.edge_attr[:, 14:EDGE_FEATURE_DIM].abs().sum(dim=1) > 1e-6).sum().item())
        except Exception as exc:
            failed += 1
            if len(failed_examples) < 20:
                failed_examples.append({"file": src.name, "error": repr(exc)})


    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total": len(files),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "face_dim": FACE_FEATURE_DIM,
        "edge_dim": EDGE_FEATURE_DIM,
        "face_nodes": face_nodes,
        "face_extra_nonzero": face_extra_nonzero,
        "face_extra_nonzero_ratio": face_extra_nonzero / max(face_nodes, 1),
        "edge_rows": edge_rows,
        "edge_extra_nonzero": edge_extra_nonzero,
        "edge_extra_nonzero_ratio": edge_extra_nonzero / max(edge_rows, 1),
        "schema": {"face": FACE_SCHEMA, "edge": EDGE_SCHEMA},
        "failed_examples": failed_examples,
    }
    save_json(report_path, report)


    print("=" * 80)
    print("[DONE] v7 preprocessing completed")
    print(f"  Processed : {processed}")
    print(f"  Skipped   : {skipped}")
    print(f"  Failed    : {failed}")
    print(f"  Face extra nonzero ratio: {report['face_extra_nonzero_ratio']:.4f}")
    print(f"  Edge extra nonzero ratio: {report['edge_extra_nonzero_ratio']:.4f}")
    print(f"  Report    : {report_path}")
    print("=" * 80)




if __name__ == "__main__":
    main()




