from __future__ import annotations


import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


import torch
from torch_geometric.data import Data
from tqdm import tqdm




FACE_TYPES = [
    "PlaneSurfaceType",
    "CylinderSurfaceType",
    "ConeSurfaceType",
    "SphereSurfaceType",
    "TorusSurfaceType",
    "NurbsSurfaceType",
    "UnknownSurfaceType",
]


CURVE_TYPES = [
    "Line3DCurveType",
    "Circle3DCurveType",
    "Arc3DCurveType",
    "Ellipse3DCurveType",
    "EllipticalArc3DCurveType",
    "NurbsCurveType",
    "UnknownCurveType",
]


CONVEXITY_TYPES = [
    "Convex",
    "Concave",
    "Smooth",
    "UnknownConvexity",
]


FACE_FEATURE_DIM = 18
EDGE_FEATURE_DIM = 19


FACE_POINT_KEYS = ["face_points", "points", "sample_points", "surface_points", "uv_grid_points", "grid_points"]
FACE_NORMAL_KEYS = ["face_normals", "normals", "sample_normals", "surface_normals", "normal_grid", "grid_normals"]
FACE_MASK_KEYS = ["face_trimming_mask", "trimming_mask", "trim_mask", "valid_mask", "mask"]
EDGE_POINT_KEYS = ["edge_entity_points", "edge_points", "points", "sample_points", "curve_points"]
EDGE_TANGENT_KEYS = ["edge_entity_tangents", "edge_tangents", "tangents", "sample_tangents", "curve_tangents"]




def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)




def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)




def get_float(node: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = node.get(key, default)
    if value is None:
        return default
    try:
        value = float(value)
    except Exception:
        return default
    if not math.isfinite(value):
        return default
    return value




def get_bool_float(node: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = node.get(key, None)
    if value is None:
        return default
    return 1.0 if bool(value) else 0.0




def log1p_nonnegative(value: float) -> float:
    return float(math.log1p(max(float(value), 0.0)))




def one_hot(value: Any, categories: list[str], unknown_category: str) -> list[float]:
    if value not in categories:
        value = unknown_category
    return [1.0 if value == c else 0.0 for c in categories]




def find_first(node: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in node and node[key] is not None:
            return node[key]
    return None




def value_to_xyz_tensor(value: Any) -> torch.Tensor:
    if value is None:
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
        if "data" in value:
            return value_to_xyz_tensor(value["data"])
        if "values" in value:
            return value_to_xyz_tensor(value["values"])
        return torch.zeros((0, 3), dtype=torch.float32)


    if isinstance(value, list):
        rows: list[list[float]] = []


        def visit(obj: Any) -> None:
            if isinstance(obj, dict):
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
            elif isinstance(obj, list):
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
        tensor = torch.tensor(rows, dtype=torch.float32)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor[:, :3]


    return torch.zeros((0, 3), dtype=torch.float32)




def value_to_mask_tensor(value: Any, n_points: int) -> torch.Tensor:
    if value is None or n_points <= 0:
        return torch.ones((n_points,), dtype=torch.bool)


    rows: list[float] = []


    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            if "data" in obj:
                visit(obj["data"])
            elif "values" in obj:
                visit(obj["values"])
            else:
                for v in obj.values():
                    visit(v)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)
        elif isinstance(obj, (bool, int, float)):
            rows.append(float(obj))


    visit(value)
    if not rows:
        return torch.ones((n_points,), dtype=torch.bool)


    mask = torch.tensor(rows, dtype=torch.float32).flatten() > 0.5
    if mask.numel() >= n_points:
        return mask[:n_points]
    padded = torch.ones((n_points,), dtype=torch.bool)
    padded[: mask.numel()] = mask
    return padded




def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    return torch.nn.functional.normalize(x.float(), dim=-1, eps=1e-8)




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
    planarity = 1.0 - float(eigvals[0].item() / denom)
    return max(0.0, min(1.0, planarity))




def spread_stats(points: torch.Tensor) -> tuple[float, float]:
    if points.size(0) == 0:
        return 0.0, 0.0
    extent = points.max(dim=0).values - points.min(dim=0).values
    mean_spread = float(extent.mean().item())
    max_spread = float(extent.max().item())
    return log1p_nonnegative(mean_spread), log1p_nonnegative(max_spread)




def vector_mean_and_concentration(vectors: torch.Tensor) -> tuple[list[float], float]:
    if vectors.size(0) == 0:
        return [0.0, 0.0, 0.0], 0.0
    unit = normalize_rows(vectors)
    mean_vec = unit.mean(dim=0)
    concentration = float(mean_vec.norm().item())
    if concentration > 1e-8:
        mean_unit = (mean_vec / mean_vec.norm()).tolist()
    else:
        mean_unit = [0.0, 0.0, 0.0]
    return [float(v) for v in mean_unit[:3]], float(max(0.0, min(1.0, concentration)))




def face_stat_features(node: dict[str, Any]) -> list[float]:
    points = value_to_xyz_tensor(find_first(node, FACE_POINT_KEYS))
    normals = value_to_xyz_tensor(find_first(node, FACE_NORMAL_KEYS))
    mask = value_to_mask_tensor(find_first(node, FACE_MASK_KEYS), points.size(0))


    if points.size(0) > 0:
        valid_points = points[mask] if mask.numel() == points.size(0) else points
        valid_ratio = float(mask.float().mean().item()) if mask.numel() > 0 else 1.0
    else:
        valid_points = points
        valid_ratio = 0.0


    if normals.size(0) > 0 and mask.numel() == normals.size(0):
        valid_normals = normals[mask]
    else:
        valid_normals = normals


    planarity = point_planarity(valid_points)
    log_spread_mean, log_spread_max = spread_stats(valid_points)
    mean_normal, normal_concentration = vector_mean_and_concentration(valid_normals)


    return [
        valid_ratio,
        normal_concentration,
        planarity,
        log_spread_mean,
        log_spread_max,
        *mean_normal,
    ]




def edge_stat_features(node: dict[str, Any]) -> list[float]:
    points = value_to_xyz_tensor(find_first(node, EDGE_POINT_KEYS))
    tangents = value_to_xyz_tensor(find_first(node, EDGE_TANGENT_KEYS))


    log_spread_mean, _ = spread_stats(points)
    mean_tangent, tangent_concentration = vector_mean_and_concentration(tangents)


    return [
        log_spread_mean,
        *mean_tangent,
        tangent_concentration,
    ]




def face_feature(node: dict[str, Any]) -> list[float]:
    base = [
        *one_hot(node.get("surface_type"), FACE_TYPES, "UnknownSurfaceType"),
        log1p_nonnegative(get_float(node, "area", 0.0)),
        get_bool_float(node, "reversed", 0.0),
        log1p_nonnegative(get_float(node, "radius", 0.0)),
    ]
    stats = face_stat_features(node)
    out = base + stats
    if len(out) != FACE_FEATURE_DIM:
        raise RuntimeError(f"Face feature dim mismatch: {len(out)}")
    return out




def edge_feature(node: dict[str, Any]) -> list[float]:
    base = [
        *one_hot(node.get("curve_type"), CURVE_TYPES, "UnknownCurveType"),
        get_bool_float(node, "reversed", 0.0),
        log1p_nonnegative(get_float(node, "length", 0.0)),
        *one_hot(node.get("convexity"), CONVEXITY_TYPES, "UnknownConvexity"),
        log1p_nonnegative(get_float(node, "radius", 0.0)),
    ]
    stats = edge_stat_features(node)
    out = base + stats
    if len(out) != EDGE_FEATURE_DIM:
        raise RuntimeError(f"Edge feature dim mismatch: {len(out)}")
    return out




def is_body_graph_file(data: Any) -> bool:
    return isinstance(data, dict) and {"nodes", "links"}.issubset(data.keys())




def is_face_node(node: Any) -> bool:
    return isinstance(node, dict) and "surface_type" in node and "area" in node




def is_edge_entity_node(node: Any) -> bool:
    return isinstance(node, dict) and "curve_type" in node and "length" in node




def node_key(node: dict[str, Any], index: int) -> Any:
    for key in ("id", "node_id", "entity_id", "index", "name"):
        if key in node:
            return node[key]
    return index




def link_endpoints(link: Any) -> tuple[Any, Any] | None:
    if not isinstance(link, dict):
        return None
    source_keys = ("source", "src", "from", "u", "start", "source_id")
    target_keys = ("target", "dst", "to", "v", "end", "target_id")


    src = None
    dst = None
    for key in source_keys:
        if key in link:
            src = link[key]
            break
    for key in target_keys:
        if key in link:
            dst = link[key]
            break


    if isinstance(src, dict):
        src = src.get("id", src.get("node", src.get("index", None)))
    if isinstance(dst, dict):
        dst = dst.get("id", dst.get("node", dst.get("index", None)))


    if src is None or dst is None:
        return None
    return src, dst




def resolve_node_ref(ref: Any, id_to_index: dict[Any, int], n_nodes: int) -> int | None:
    if ref in id_to_index:
        return id_to_index[ref]
    if isinstance(ref, str):
        if ref in id_to_index:
            return id_to_index[ref]
        try:
            as_int = int(ref)
            if as_int in id_to_index:
                return id_to_index[as_int]
            if 0 <= as_int < n_nodes:
                return as_int
        except Exception:
            return None
    if isinstance(ref, int):
        if ref in id_to_index:
            return id_to_index[ref]
        if 0 <= ref < n_nodes:
            return ref
    return None




def graph_to_pyg(data: dict[str, Any], body_id: str) -> Data:
    nodes = data.get("nodes", [])
    links = data.get("links", [])
    if not isinstance(nodes, list):
        raise ValueError("nodes must be a list.")
    if not isinstance(links, list):
        links = []


    id_to_index = {node_key(node, i): i for i, node in enumerate(nodes) if isinstance(node, dict)}
    id_to_index.update({str(k): v for k, v in list(id_to_index.items())})


    face_global_to_local: dict[int, int] = {}
    edge_global_to_local: dict[int, int] = {}
    face_features: list[list[float]] = []
    edge_features: list[list[float]] = []


    for i, node in enumerate(nodes):
        if is_face_node(node):
            face_global_to_local[i] = len(face_features)
            face_features.append(face_feature(node))
        elif is_edge_entity_node(node):
            edge_global_to_local[i] = len(edge_features)
            edge_features.append(edge_feature(node))


    if not face_features:
        raise ValueError("No face nodes found.")


    edge_to_faces: dict[int, set[int]] = defaultdict(set)
    direct_edges: list[tuple[int, int]] = []


    for link in links:
        endpoints = link_endpoints(link)
        if endpoints is None:
            continue
        src_ref, dst_ref = endpoints
        src_i = resolve_node_ref(src_ref, id_to_index, len(nodes))
        dst_i = resolve_node_ref(dst_ref, id_to_index, len(nodes))
        if src_i is None or dst_i is None:
            continue


        src_is_face = src_i in face_global_to_local
        dst_is_face = dst_i in face_global_to_local
        src_is_edge = src_i in edge_global_to_local
        dst_is_edge = dst_i in edge_global_to_local


        if src_is_edge and dst_is_face:
            edge_to_faces[edge_global_to_local[src_i]].add(face_global_to_local[dst_i])
        elif dst_is_edge and src_is_face:
            edge_to_faces[edge_global_to_local[dst_i]].add(face_global_to_local[src_i])
        elif src_is_face and dst_is_face:
            direct_edges.append((face_global_to_local[src_i], face_global_to_local[dst_i]))


    edge_index_pairs: list[tuple[int, int]] = []
    edge_attr_rows: list[list[float]] = []


    for edge_local_idx, faces in edge_to_faces.items():
        faces_list = sorted(faces)
        if len(faces_list) < 2:
            continue
        for a, b in combinations(faces_list, 2):
            edge_index_pairs.append((a, b))
            edge_attr_rows.append(edge_features[edge_local_idx])
            edge_index_pairs.append((b, a))
            edge_attr_rows.append(edge_features[edge_local_idx])


    if not edge_index_pairs and direct_edges:
        default_edge_attr = [0.0] * EDGE_FEATURE_DIM
        for a, b in direct_edges:
            edge_index_pairs.append((a, b))
            edge_attr_rows.append(default_edge_attr)
            edge_index_pairs.append((b, a))
            edge_attr_rows.append(default_edge_attr)


    x = torch.tensor(face_features, dtype=torch.float32)


    if edge_index_pairs:
        edge_index = torch.tensor(edge_index_pairs, dtype=torch.long).T.contiguous()
        edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)


    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)


    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        body_id=body_id,
        feature_schema={
            "face_dim": FACE_FEATURE_DIM,
            "edge_dim": EDGE_FEATURE_DIM,
            "face": [
                "surface_type_oh7",
                "log_area_dim7",
                "reversed_dim8",
                "log_radius_dim9",
                "valid_ratio_dim10",
                "normal_concentration_dim11",
                "point_planarity_dim12",
                "log_spread_mean_dim13",
                "log_spread_max_dim14",
                "mean_normal_xyz_dim15_17",
            ],
            "edge": [
                "curve_type_oh7",
                "reversed_dim7",
                "log_length_dim8",
                "convexity_oh4_dim9_12",
                "log_radius_dim13",
                "log_edge_spread_mean_dim14",
                "mean_tangent_xyz_dim15_17",
                "tangent_concentration_dim18",
            ],
        },
    )




def parse_body_graph_index(index_data: Any, raw_graph_dir: Path | None = None) -> list[tuple[str, Path]]:
    records: list[tuple[str, Path]] = []


    def add_record(body_id: Any, path_value: Any) -> None:
        if body_id is None:
            return
        path: Path | None = None
        if isinstance(path_value, str):
            path = Path(path_value)
        elif isinstance(path_value, dict):
            for key in ("path", "json_path", "file", "filepath", "graph_path"):
                if key in path_value:
                    path = Path(str(path_value[key]))
                    break
        elif isinstance(path_value, Path):
            path = path_value
        if path is None and raw_graph_dir is not None:
            path = raw_graph_dir / f"{body_id}.json"
        if path is not None:
            records.append((str(body_id), path))


    if isinstance(index_data, dict):
        for key, value in index_data.items():
            if isinstance(value, str):
                add_record(key, value)
            elif isinstance(value, dict):
                body_id = value.get("body_id", value.get("id", key))
                add_record(body_id, value)
            elif value is None and raw_graph_dir is not None:
                add_record(key, raw_graph_dir / f"{key}.json")
    elif isinstance(index_data, list):
        for item in index_data:
            if isinstance(item, str):
                body_id = Path(item).stem
                add_record(body_id, item)
            elif isinstance(item, dict):
                body_id = item.get("body_id", item.get("id", item.get("name", None)))
                if body_id is None:
                    for key in ("path", "json_path", "file", "filepath", "graph_path"):
                        if key in item:
                            body_id = Path(str(item[key])).stem
                            break
                add_record(body_id, item)


    if raw_graph_dir is not None:
        fixed_records: list[tuple[str, Path]] = []
        for body_id, path in records:
            if path.exists():
                fixed_records.append((body_id, path))
            else:
                fallback = raw_graph_dir / f"{body_id}.json"
                fixed_records.append((body_id, fallback))
        records = fixed_records


    seen: set[str] = set()
    deduped: list[tuple[str, Path]] = []
    for body_id, path in records:
        if body_id in seen:
            continue
        seen.add(body_id)
        deduped.append((body_id, path))
    return deduped




def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw body graph JSON files into v6 fixed geometric PyG graphs.")
    parser.add_argument("--body_graph_index", type=str, default=r"outputs\pair_index\body_graph_files.json")
    parser.add_argument("--raw_graph_dir", type=str, default=r"D:\soyoung\j1.0.0\joint")
    parser.add_argument("--output_dir", type=str, default=r"processed\body_graphs_v6_full")
    parser.add_argument("--report_path", type=str, default=r"outputs\reports\pyg_preprocess_v6_summary.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()


    index_path = Path(args.body_graph_index)
    raw_graph_dir = Path(args.raw_graph_dir) if args.raw_graph_dir else None
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)


    index_data = load_json(index_path)
    records = parse_body_graph_index(index_data, raw_graph_dir=raw_graph_dir)
    if args.limit is not None:
        records = records[: args.limit]


    output_dir.mkdir(parents=True, exist_ok=True)


    print("=" * 70)
    print("[v6] Body Graph Preprocessing - fixed geometric stats")
    print(f"  Face x    : {FACE_FEATURE_DIM} dims")
    print("              type OH7 + log_area + reversed + log_radius")
    print("              + valid_ratio + normal_concentration + planarity")
    print("              + spread stats + mean_normal_xyz at dims 15-17")
    print(f"  Edge attr : {EDGE_FEATURE_DIM} dims")
    print("              curve OH7 + reversed + log_length + convexity OH4")
    print("              + log_radius + spread + mean_tangent_xyz + concentration")
    print("  PointNet  : not required")
    print("=" * 70)
    print(f"Total: {len(records)}  |  Output: {output_dir}")


    processed = 0
    skipped = 0
    failed = 0
    failed_examples: list[dict[str, str]] = []
    total_face_nodes = 0
    total_directed_edges = 0
    graphs_zero_edges = 0


    for body_id, json_path in tqdm(records):
        out_path = output_dir / f"{body_id}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue


        try:
            data = load_json(json_path)
            if not is_body_graph_file(data):
                raise ValueError("Input JSON does not look like a body graph file.")
            graph = graph_to_pyg(data, body_id)
            torch.save(graph, out_path)


            processed += 1
            total_face_nodes += int(graph.x.shape[0])
            total_directed_edges += int(graph.edge_index.shape[1])
            if int(graph.edge_index.shape[1]) == 0:
                graphs_zero_edges += 1
        except Exception as exc:
            failed += 1
            if len(failed_examples) < 50:
                failed_examples.append({"body_id": body_id, "path": str(json_path), "error": str(exc)})


    report = {
        "face_feature_dim": FACE_FEATURE_DIM,
        "edge_feature_dim": EDGE_FEATURE_DIM,
        "total": len(records),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "failed_examples": failed_examples,
        "total_face_nodes": total_face_nodes,
        "total_directed_edges": total_directed_edges,
        "graphs_zero_edges": graphs_zero_edges,
        "output_dir": str(output_dir),
        "schema": {
            "face_dims": {
                "0_6": "surface_type one-hot",
                "7": "log_area",
                "8": "reversed",
                "9": "log_radius",
                "10": "valid_ratio",
                "11": "normal_concentration",
                "12": "point_planarity",
                "13": "log_spread_mean",
                "14": "log_spread_max",
                "15_17": "mean_normal_xyz",
            },
            "edge_dims": {
                "0_6": "curve_type one-hot",
                "7": "reversed",
                "8": "log_length",
                "9_12": "convexity one-hot",
                "13": "log_radius",
                "14": "log_edge_spread_mean",
                "15_17": "mean_tangent_xyz",
                "18": "tangent_concentration",
            },
        },
    }
    save_json(report_path, report)


    print("=" * 70)
    print("[DONE] v6 preprocessing completed")
    print(f"  Processed : {processed}")
    print(f"  Skipped   : {skipped}")
    print(f"  Failed    : {failed}")
    print(f"  Report    : {report_path}")




if __name__ == "__main__":
    main()
