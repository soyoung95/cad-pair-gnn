
from pathlib import Path
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations

import torch
from torch_geometric.data import Data

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x: x


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


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
    "NurbsCurve3DCurveType",
    "UnknownCurveType",
]

CONVEXITY_TYPES = [
    "Convex",
    "Concave",
    "Smooth",
    "UnknownConvexity",
]

PERPENDICULARITY_TYPES = [
    "Perpendicular",
    "Non-perpendicular",
    "UnknownPerpendicularity",
]


def make_feature_names():
    face_names = []
    face_names += [f"surface_type::{name}" for name in FACE_TYPES]
    face_names += [
        "log1p_area",
        "area_ratio",
        "log1p_radius",
        "has_radius",
        "reversed",
    ]

    face_names += ["centroid_x", "centroid_y", "centroid_z"]
    face_names += ["normal_x", "normal_y", "normal_z"]
    face_names += ["point_on_face_x", "point_on_face_y", "point_on_face_z"]
    face_names += ["max_tangent_x", "max_tangent_y", "max_tangent_z"]
    face_names += ["signed_log1p_max_curvature", "signed_log1p_min_curvature"]

    face_names += ["origin_x", "origin_y", "origin_z", "has_origin"]
    face_names += ["axis_x", "axis_y", "axis_z", "has_axis"]

    for prefix in ["points", "normals"]:
        for stat in ["mean", "std", "min", "max", "range"]:
            face_names += [
                f"{prefix}_{stat}_x",
                f"{prefix}_{stat}_y",
                f"{prefix}_{stat}_z",
            ]

    face_names += [
        "trimming_mask_mean",
        "trimming_mask_std",
        "trimming_mask_min",
        "trimming_mask_max",
        "trimming_mask_nonzero_ratio",
        "trimming_mask_log_count",
    ]

    face_names += [
        "points_log_count",
        "normals_log_count",
    ]

    edge_names = []
    edge_names += [f"curve_type::{name}" for name in CURVE_TYPES]
    edge_names += [f"convexity::{name}" for name in CONVEXITY_TYPES]
    edge_names += [f"perpendicularity::{name}" for name in PERPENDICULARITY_TYPES]

    edge_names += [
        "log1p_length",
        "length_ratio",
        "dihedral_angle",
        "log1p_radius",
        "has_radius",
        "reversed",
        "is_degenerate",
        "signed_log1p_curvature",
    ]

    edge_names += [
        "curvature_direction_x",
        "curvature_direction_y",
        "curvature_direction_z",
    ]

    edge_names += ["start_point_x", "start_point_y", "start_point_z", "has_start_point"]
    edge_names += ["end_point_x", "end_point_y", "end_point_z", "has_end_point"]
    edge_names += ["center_x", "center_y", "center_z", "has_center"]
    edge_names += ["normal_x", "normal_y", "normal_z", "has_normal"]

    for prefix in ["points", "tangents"]:
        for stat in ["mean", "std"]:
            edge_names += [
                f"{prefix}_{stat}_x",
                f"{prefix}_{stat}_y",
                f"{prefix}_{stat}_z",
            ]

    return face_names, edge_names


FACE_FEATURE_NAMES, EDGE_FEATURE_NAMES = make_feature_names()
FACE_FEATURE_DIM = len(FACE_FEATURE_NAMES)
EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def one_hot(value, categories, unknown_value):
    if value not in categories:
        value = unknown_value
    return [1.0 if value == category else 0.0 for category in categories]


def get_float(data, key, default=0.0):
    value = data.get(key, default)
    if value is None:
        return default

    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def get_bool_float(data, key, default=0.0):
    value = data.get(key, None)
    if value is None:
        return float(default)
    return 1.0 if bool(value) else 0.0


def signed_log1p(value):
    value = float(value)
    return math.copysign(math.log1p(abs(value)), value)


def normalize_id(value):
    if isinstance(value, dict):
        for key in ["id", "node", "node_id", "target", "source"]:
            if key in value:
                return str(value[key])
    return str(value)


def is_face_node(node):
    return isinstance(node, dict) and ("surface_type" in node or "area" in node)


def is_edge_node(node):
    return isinstance(node, dict) and (
        "curve_type" in node
        or "convexity" in node
        or "length" in node
        or "dihedral_angle" in node
    )


def get_link_endpoints(link):
    if not isinstance(link, dict):
        return None, None

    key_pairs = [
        ("source", "target"),
        ("src", "dst"),
        ("from", "to"),
        ("source_id", "target_id"),
        ("node_a", "node_b"),
        ("a", "b"),
    ]

    for left_key, right_key in key_pairs:
        if left_key in link and right_key in link:
            return normalize_id(link[left_key]), normalize_id(link[right_key])

    for list_key in ["nodes", "endpoints"]:
        if list_key in link and isinstance(link[list_key], list) and len(link[list_key]) >= 2:
            return normalize_id(link[list_key][0]), normalize_id(link[list_key][1])

    return None, None


def extract_points_from_flat_list(values):
    if not isinstance(values, list):
        return []

    usable = len(values) - (len(values) % 3)
    points = []

    for i in range(0, usable, 3):
        try:
            x = float(values[i])
            y = float(values[i + 1])
            z = float(values[i + 2])
            if not any(math.isnan(v) or math.isinf(v) for v in [x, y, z]):
                points.append((x, y, z))
        except Exception:
            continue

    return points


def compute_body_normalization(nodes):
    all_points = []

    for node in nodes:
        if isinstance(node, dict):
            all_points.extend(extract_points_from_flat_list(node.get("points", [])))

    if not all_points:
        return (0.0, 0.0, 0.0), 1.0

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    zs = [p[2] for p in all_points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    center = (
        0.5 * (min_x + max_x),
        0.5 * (min_y + max_y),
        0.5 * (min_z + max_z),
    )

    scale = max(max_x - min_x, max_y - min_y, max_z - min_z, 1e-6)
    return center, scale


def normalize_xyz(x, y, z, center, scale):
    return [
        (float(x) - center[0]) / scale,
        (float(y) - center[1]) / scale,
        (float(z) - center[2]) / scale,
    ]


def get_position_xyz(data, prefix, center, scale, add_presence=False):
    keys = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
    present = all(key in data and data.get(key) is not None for key in keys)

    if present:
        values = normalize_xyz(
            get_float(data, keys[0]),
            get_float(data, keys[1]),
            get_float(data, keys[2]),
            center,
            scale,
        )
    else:
        values = [0.0, 0.0, 0.0]

    if add_presence:
        values.append(1.0 if present else 0.0)

    return values


def get_vector_xyz(data, prefix, add_presence=False):
    keys = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
    present = all(key in data and data.get(key) is not None for key in keys)

    if present:
        values = [
            get_float(data, keys[0]),
            get_float(data, keys[1]),
            get_float(data, keys[2]),
        ]
    else:
        values = [0.0, 0.0, 0.0]

    if add_presence:
        values.append(1.0 if present else 0.0)

    return values


def xyz_stats(values, center=None, scale=None, vector_mode=False, rich=False):
    points = extract_points_from_flat_list(values)

    if not points:
        return [0.0] * (15 if rich else 6)

    processed = []

    for x, y, z in points:
        if vector_mode:
            processed.append((x, y, z))
        else:
            processed.append(tuple(normalize_xyz(x, y, z, center, scale)))

    xs = [p[0] for p in processed]
    ys = [p[1] for p in processed]
    zs = [p[2] for p in processed]

    def mean(arr):
        return sum(arr) / len(arr)

    def std(arr):
        mu = mean(arr)
        return math.sqrt(sum((v - mu) ** 2 for v in arr) / len(arr))

    basic = [
        mean(xs), mean(ys), mean(zs),
        std(xs), std(ys), std(zs),
    ]

    if not rich:
        return basic

    mins = [min(xs), min(ys), min(zs)]
    maxs = [max(xs), max(ys), max(zs)]
    ranges = [maxs[i] - mins[i] for i in range(3)]

    return [
        mean(xs), mean(ys), mean(zs),
        std(xs), std(ys), std(zs),
        mins[0], mins[1], mins[2],
        maxs[0], maxs[1], maxs[2],
        ranges[0], ranges[1], ranges[2],
    ]


def numeric_list_stats(values):
    if not isinstance(values, list) or not values:
        return [0.0] * 6

    clean = []

    for value in values:
        try:
            v = float(value)
            if not math.isnan(v) and not math.isinf(v):
                clean.append(v)
        except Exception:
            continue

    if not clean:
        return [0.0] * 6

    mean_value = sum(clean) / len(clean)
    std_value = math.sqrt(sum((v - mean_value) ** 2 for v in clean) / len(clean))
    min_value = min(clean)
    max_value = max(clean)
    nonzero_ratio = sum(1 for v in clean if abs(v) > 1e-12) / len(clean)
    log_count = math.log1p(len(clean))

    return [
        mean_value,
        std_value,
        min_value,
        max_value,
        nonzero_ratio,
        log_count,
    ]


def make_face_feature(node, total_area, center, scale):
    surface_type = node.get("surface_type", "UnknownSurfaceType")
    if surface_type not in FACE_TYPES:
        surface_type = "UnknownSurfaceType"

    area = max(get_float(node, "area", 0.0), 0.0)
    area_ratio = area / total_area if total_area > 0 else 0.0

    radius_raw = get_float(node, "radius", 0.0)
    radius = max(radius_raw, 0.0)
    has_radius = 1.0 if "radius" in node and node.get("radius") is not None else 0.0

    feature = []

    feature.extend(one_hot(surface_type, FACE_TYPES, "UnknownSurfaceType"))
    feature.extend(
        [
            math.log1p(area),
            area_ratio,
            math.log1p(radius),
            has_radius,
            get_bool_float(node, "reversed", 0.0),
        ]
    )

    feature.extend(get_position_xyz(node, "centroid", center, scale))
    feature.extend(get_vector_xyz(node, "normal"))
    feature.extend(get_position_xyz(node, "point_on_face", center, scale))
    feature.extend(get_vector_xyz(node, "max_tangent"))

    feature.extend(
        [
            signed_log1p(get_float(node, "max_curvature", 0.0)),
            signed_log1p(get_float(node, "min_curvature", 0.0)),
        ]
    )

    feature.extend(get_position_xyz(node, "origin", center, scale, add_presence=True))
    feature.extend(get_vector_xyz(node, "axis", add_presence=True))

    feature.extend(xyz_stats(node.get("points", []), center=center, scale=scale, vector_mode=False, rich=True))
    feature.extend(xyz_stats(node.get("normals", []), vector_mode=True, rich=True))
    feature.extend(numeric_list_stats(node.get("trimming_mask", [])))

    point_count = len(extract_points_from_flat_list(node.get("points", [])))
    normal_count = len(extract_points_from_flat_list(node.get("normals", [])))
    feature.extend([math.log1p(point_count), math.log1p(normal_count)])

    if len(feature) != FACE_FEATURE_DIM:
        raise ValueError(f"Invalid face feature dimension: {len(feature)}")

    return feature


def make_edge_feature(edge_node, total_length, center, scale):
    curve_type = edge_node.get("curve_type", "UnknownCurveType")
    if curve_type not in CURVE_TYPES:
        curve_type = "UnknownCurveType"

    convexity = edge_node.get("convexity", "UnknownConvexity")
    if convexity not in CONVEXITY_TYPES:
        convexity = "UnknownConvexity"

    perpendicularity = edge_node.get("perpendicularity", "UnknownPerpendicularity")
    if perpendicularity not in PERPENDICULARITY_TYPES:
        perpendicularity = "UnknownPerpendicularity"

    length = max(get_float(edge_node, "length", 0.0), 0.0)
    length_ratio = length / total_length if total_length > 0 else 0.0

    radius_raw = get_float(edge_node, "radius", 0.0)
    radius = max(radius_raw, 0.0)
    has_radius = 1.0 if "radius" in edge_node and edge_node.get("radius") is not None else 0.0

    feature = []

    feature.extend(one_hot(curve_type, CURVE_TYPES, "UnknownCurveType"))
    feature.extend(one_hot(convexity, CONVEXITY_TYPES, "UnknownConvexity"))
    feature.extend(one_hot(perpendicularity, PERPENDICULARITY_TYPES, "UnknownPerpendicularity"))

    feature.extend(
        [
            math.log1p(length),
            length_ratio,
            get_float(edge_node, "dihedral_angle", 0.0),
            math.log1p(radius),
            has_radius,
            get_bool_float(edge_node, "reversed", 0.0),
            get_bool_float(edge_node, "is_degenerate", 0.0),
            signed_log1p(get_float(edge_node, "curvature", 0.0)),
        ]
    )

    feature.extend(get_vector_xyz(edge_node, "curvature_direction"))
    feature.extend(get_position_xyz(edge_node, "start_point", center, scale, add_presence=True))
    feature.extend(get_position_xyz(edge_node, "end_point", center, scale, add_presence=True))
    feature.extend(get_position_xyz(edge_node, "center", center, scale, add_presence=True))
    feature.extend(get_vector_xyz(edge_node, "normal", add_presence=True))

    feature.extend(xyz_stats(edge_node.get("points", []), center=center, scale=scale, vector_mode=False, rich=False))
    feature.extend(xyz_stats(edge_node.get("tangents", []), vector_mode=True, rich=False))

    if len(feature) != EDGE_FEATURE_DIM:
        raise ValueError(f"Invalid edge feature dimension: {len(feature)}")

    return feature


def convert_body_graph_to_pyg_v2(data):
    nodes = data.get("nodes", [])
    links = data.get("links", [])

    if not isinstance(nodes, list) or not isinstance(links, list):
        raise ValueError("Input JSON must contain nodes and links lists.")

    id_to_node = {}
    face_ids = []
    edge_ids = []

    for node in nodes:
        if not isinstance(node, dict) or "id" not in node:
            continue

        node_id = normalize_id(node["id"])
        id_to_node[node_id] = node

        if is_face_node(node):
            face_ids.append(node_id)
        elif is_edge_node(node):
            edge_ids.append(node_id)

    face_ids = sorted(face_ids, key=lambda x: int(x) if x.isdigit() else x)
    edge_ids = sorted(edge_ids, key=lambda x: int(x) if x.isdigit() else x)

    if not face_ids:
        raise ValueError("No face nodes found.")

    center, scale = compute_body_normalization(nodes)

    total_area = sum(
        max(get_float(id_to_node[face_id], "area", 0.0), 0.0)
        for face_id in face_ids
    )

    total_length = sum(
        max(get_float(id_to_node[edge_id], "length", 0.0), 0.0)
        for edge_id in edge_ids
    )

    face_id_to_new_index = {
        face_id: index
        for index, face_id in enumerate(face_ids)
    }

    face_features = [
        make_face_feature(id_to_node[face_id], total_area, center, scale)
        for face_id in face_ids
    ]

    edge_to_faces = defaultdict(set)

    for link in links:
        left, right = get_link_endpoints(link)

        if left is None or right is None:
            continue

        if left in edge_ids and right in face_id_to_new_index:
            edge_to_faces[left].add(right)
        elif right in edge_ids and left in face_id_to_new_index:
            edge_to_faces[right].add(left)

    edge_index_list = []
    edge_feature_list = []

    skipped_no_adjacent_face = 0
    skipped_one_adjacent_face = 0
    edge_entity_more_than_two_faces = 0

    for edge_id in edge_ids:
        adjacent_face_ids = sorted(
            edge_to_faces.get(edge_id, set()),
            key=lambda x: int(x) if x.isdigit() else x,
        )

        if len(adjacent_face_ids) == 0:
            skipped_no_adjacent_face += 1
            continue

        if len(adjacent_face_ids) == 1:
            skipped_one_adjacent_face += 1
            continue

        if len(adjacent_face_ids) > 2:
            edge_entity_more_than_two_faces += 1

        edge_feature = make_edge_feature(id_to_node[edge_id], total_length, center, scale)

        for face_a, face_b in combinations(adjacent_face_ids, 2):
            index_a = face_id_to_new_index[face_a]
            index_b = face_id_to_new_index[face_b]

            edge_index_list.append([index_a, index_b])
            edge_feature_list.append(edge_feature)

            edge_index_list.append([index_b, index_a])
            edge_feature_list.append(edge_feature)

    x = torch.tensor(face_features, dtype=torch.float32)

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feature_list, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    pyg_data.feature_version = "geometry_v2"
    pyg_data.face_feature_dim = FACE_FEATURE_DIM
    pyg_data.edge_feature_dim = EDGE_FEATURE_DIM

    stats = {
        "num_faces": int(x.size(0)),
        "num_directed_edges": int(edge_index.size(1)),
        "num_undirected_edges": int(edge_index.size(1) // 2),
        "face_feature_dim": int(x.size(1)),
        "edge_feature_dim": int(edge_attr.size(1)) if edge_attr.dim() == 2 else EDGE_FEATURE_DIM,
        "skipped_no_adjacent_face": skipped_no_adjacent_face,
        "skipped_one_adjacent_face": skipped_one_adjacent_face,
        "edge_entity_more_than_two_faces": edge_entity_more_than_two_faces,
        "body_center": center,
        "body_scale": scale,
    }

    return pyg_data, stats


def replace_path_if_needed(path_str, replace_from, replace_to):
    if not replace_from:
        return path_str
    return path_str.replace(replace_from, replace_to)


def make_markdown_report(summary):
    lines = []
    lines.append("# PyG Geometry v2 Preprocess Summary")
    lines.append("")
    lines.append(f"- Total body graphs: {summary['total_body_graphs']}")
    lines.append(f"- Processed body graphs: {summary['processed_body_graphs']}")
    lines.append(f"- Skipped existing files: {summary['skipped_existing_files']}")
    lines.append(f"- Failed body graphs: {summary['failed_body_graphs']}")
    lines.append(f"- Face feature dimensions: {summary['face_feature_dimensions']}")
    lines.append(f"- Edge feature dimensions: {summary['edge_feature_dimensions']}")
    lines.append("")
    lines.append("## Feature dimensions")
    lines.append("")
    lines.append(f"- Face feature dim: {FACE_FEATURE_DIM}")
    lines.append(f"- Edge feature dim: {EDGE_FEATURE_DIM}")
    lines.append("")
    lines.append("## Failed examples")
    lines.append("")

    if not summary["failed_examples"]:
        lines.append("- None")
    else:
        for item in summary["failed_examples"][:50]:
            lines.append(f"- `{item['body_id']}`: {item['error']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--body_graph_map", type=str, default="outputs/pair_index/body_graph_files.json")
    parser.add_argument("--output_dir", type=str, default="processed/body_graphs_v2_geometry")
    parser.add_argument("--report_json", type=str, default="outputs/reports/pyg_preprocess_geometry_v2_summary.json")
    parser.add_argument("--report_md", type=str, default="outputs/reports/pyg_preprocess_geometry_v2_summary.md")
    parser.add_argument("--schema_json", type=str, default="outputs/reports/pyg_preprocess_geometry_v2_schema.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--path_replace_from", type=str, default="")
    parser.add_argument("--path_replace_to", type=str, default="")
    args = parser.parse_args()

    body_graph_map_path = Path(args.body_graph_map)
    output_dir = Path(args.output_dir)
    report_json_path = Path(args.report_json)
    report_md_path = Path(args.report_md)
    schema_json_path = Path(args.schema_json)

    body_graph_map = load_json(body_graph_map_path)

    if not isinstance(body_graph_map, dict):
        raise TypeError("body_graph_map must be a dict of body_id -> raw json path.")

    items = sorted(body_graph_map.items())

    if args.limit is not None:
        items = items[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)

    processed_body_graphs = 0
    skipped_existing_files = 0
    failed_body_graphs = 0
    failed_examples = []

    face_feature_dim_counter = Counter()
    edge_feature_dim_counter = Counter()

    print("=" * 100)
    print("[STEP] Preprocessing body graphs to PyG Data - geometry v2")
    print("=" * 100)
    print(f"Body graph map : {body_graph_map_path}")
    print(f"Output dir     : {output_dir}")
    print(f"Total items    : {len(items)}")
    print(f"Face dim       : {FACE_FEATURE_DIM}")
    print(f"Edge dim       : {EDGE_FEATURE_DIM}")

    for body_id, json_path_str in tqdm(items):
        mapped_path = replace_path_if_needed(
            str(json_path_str),
            args.path_replace_from,
            args.path_replace_to,
        )

        json_path = Path(mapped_path)
        output_path = output_dir / f"{body_id}.pt"

        if output_path.exists() and not args.overwrite:
            skipped_existing_files += 1
            continue

        try:
            raw_data = load_json(json_path)
            pyg_data, stats = convert_body_graph_to_pyg_v2(raw_data)

            if pyg_data.x.size(1) != FACE_FEATURE_DIM:
                raise ValueError(f"Unexpected x dimension: {pyg_data.x.size()}")

            if pyg_data.edge_attr.size(1) != EDGE_FEATURE_DIM:
                raise ValueError(f"Unexpected edge_attr dimension: {pyg_data.edge_attr.size()}")

            torch.save(pyg_data, output_path)

            processed_body_graphs += 1
            face_feature_dim_counter[stats["face_feature_dim"]] += 1
            edge_feature_dim_counter[stats["edge_feature_dim"]] += 1

        except Exception as exc:
            failed_body_graphs += 1
            if len(failed_examples) < 100:
                failed_examples.append(
                    {
                        "body_id": body_id,
                        "json_path": str(json_path),
                        "error": repr(exc),
                    }
                )

    summary = {
        "feature_version": "geometry_v2",
        "body_graph_map": str(body_graph_map_path),
        "output_dir": str(output_dir),
        "total_body_graphs": len(items),
        "processed_body_graphs": processed_body_graphs,
        "skipped_existing_files": skipped_existing_files,
        "failed_body_graphs": failed_body_graphs,
        "failed_examples": failed_examples,
        "face_feature_dim": FACE_FEATURE_DIM,
        "edge_feature_dim": EDGE_FEATURE_DIM,
        "face_feature_dimensions": dict(face_feature_dim_counter),
        "edge_feature_dimensions": dict(edge_feature_dim_counter),
    }

    schema = {
        "feature_version": "geometry_v2",
        "face_feature_dim": FACE_FEATURE_DIM,
        "edge_feature_dim": EDGE_FEATURE_DIM,
        "face_feature_names": FACE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
    }

    save_json(report_json_path, summary)
    save_json(schema_json_path, schema)

    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text(make_markdown_report(summary), encoding="utf-8")

    print()
    print("=" * 100)
    print("[DONE]")
    print("=" * 100)
    print(f"Processed: {processed_body_graphs}")
    print(f"Skipped  : {skipped_existing_files}")
    print(f"Failed   : {failed_body_graphs}")
    print(f"Report   : {report_json_path}")
    print(f"Schema   : {schema_json_path}")
    print(f"Markdown : {report_md_path}")

    if failed_examples:
        print()
        print("[FAILED EXAMPLES]")
        for item in failed_examples[:10]:
            print(f"{item['body_id']}: {item['error']} | {item['json_path']}")


if __name__ == "__main__":
    main()
