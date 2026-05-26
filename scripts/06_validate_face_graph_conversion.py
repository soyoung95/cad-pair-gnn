
import argparse
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations


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


def safe_load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="cp949") as f:
                return json.load(f), None
        except Exception as e:
            return None, str(e)
    except Exception as e:
        return None, str(e)


def is_body_graph_file(data):
    if not isinstance(data, dict):
        return False

    required_keys = {"nodes", "links", "graph", "properties"}
    return required_keys.issubset(set(data.keys()))


def is_face_node(node):
    return isinstance(node, dict) and "surface_type" in node and "area" in node


def is_edge_entity_node(node):
    return isinstance(node, dict) and "curve_type" in node and "length" in node


def get_float(node, key, default=0.0):
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


def get_bool_float(node, key, default=0.0):
    value = node.get(key, None)

    if value is None:
        return default

    return 1.0 if bool(value) else 0.0


def one_hot(value, categories, unknown_category):
    if value not in categories:
        value = unknown_category

    return [1.0 if value == category else 0.0 for category in categories]


def build_adjacency(links):
    adjacency = defaultdict(set)

    for link in links:
        if not isinstance(link, dict):
            continue

        source = link.get("source")
        target = link.get("target")

        if source is None or target is None:
            continue

        adjacency[source].add(target)
        adjacency[target].add(source)

    return adjacency


def convert_to_face_graph(data):
    nodes = data.get("nodes", [])
    links = data.get("links", [])

    id_to_node = {}
    face_ids = []
    edge_entity_ids = []

    for node in nodes:
        node_id = node.get("id", None)

        if node_id is None:
            continue

        id_to_node[node_id] = node

        if is_face_node(node):
            face_ids.append(node_id)
        elif is_edge_entity_node(node):
            edge_entity_ids.append(node_id)

    adjacency = build_adjacency(links)

    face_id_to_new_index = {
        face_id: index
        for index, face_id in enumerate(sorted(face_ids))
    }

    total_area = sum(max(get_float(id_to_node[face_id], "area", 0.0), 0.0) for face_id in face_ids)
    total_length = sum(max(get_float(id_to_node[edge_id], "length", 0.0), 0.0) for edge_id in edge_entity_ids)

    face_features = []
    face_type_counter = Counter()

    for face_id in sorted(face_ids):
        node = id_to_node[face_id]

        surface_type = node.get("surface_type", "UnknownSurfaceType")
        if surface_type not in FACE_TYPES:
            surface_type = "UnknownSurfaceType"

        area = max(get_float(node, "area", 0.0), 0.0)
        radius = get_float(node, "radius", 0.0)
        has_radius = 1.0 if "radius" in node and node.get("radius") is not None else 0.0
        reversed_value = get_bool_float(node, "reversed", 0.0)

        area_ratio = area / total_area if total_area > 0 else 0.0

        feature = []
        feature.extend(one_hot(surface_type, FACE_TYPES, "UnknownSurfaceType"))
        feature.extend(
            [
                math.log1p(area),
                area_ratio,
                math.log1p(max(radius, 0.0)),
                has_radius,
                reversed_value,
            ]
        )

        face_features.append(feature)
        face_type_counter[surface_type] += 1

    face_edges = []
    edge_features = []

    edge_curve_type_counter = Counter()
    edge_convexity_counter = Counter()

    skipped_edge_entity_no_face = 0
    skipped_edge_entity_one_face = 0
    edge_entity_more_than_two_faces = 0

    for edge_id in sorted(edge_entity_ids):
        edge_node = id_to_node[edge_id]

        adjacent_face_ids = [
            neighbor_id
            for neighbor_id in adjacency.get(edge_id, [])
            if neighbor_id in face_id_to_new_index
        ]

        if len(adjacent_face_ids) == 0:
            skipped_edge_entity_no_face += 1
            continue

        if len(adjacent_face_ids) == 1:
            skipped_edge_entity_one_face += 1
            continue

        if len(adjacent_face_ids) > 2:
            edge_entity_more_than_two_faces += 1

        curve_type = edge_node.get("curve_type", "UnknownCurveType")
        if curve_type not in CURVE_TYPES:
            curve_type = "UnknownCurveType"

        convexity = edge_node.get("convexity", "UnknownConvexity")
        if convexity not in CONVEXITY_TYPES:
            convexity = "UnknownConvexity"

        length = max(get_float(edge_node, "length", 0.0), 0.0)
        length_ratio = length / total_length if total_length > 0 else 0.0
        dihedral_angle = get_float(edge_node, "dihedral_angle", 0.0)
        radius = get_float(edge_node, "radius", 0.0)
        has_radius = 1.0 if "radius" in edge_node and edge_node.get("radius") is not None else 0.0

        feature = []
        feature.extend(one_hot(curve_type, CURVE_TYPES, "UnknownCurveType"))
        feature.extend(one_hot(convexity, CONVEXITY_TYPES, "UnknownConvexity"))
        feature.extend(
            [
                math.log1p(length),
                length_ratio,
                dihedral_angle,
                math.log1p(max(radius, 0.0)),
                has_radius,
            ]
        )

        for face_a, face_b in combinations(sorted(adjacent_face_ids), 2):
            index_a = face_id_to_new_index[face_a]
            index_b = face_id_to_new_index[face_b]

            face_edges.append([index_a, index_b])
            edge_features.append(feature)

            face_edges.append([index_b, index_a])
            edge_features.append(feature)

        edge_curve_type_counter[curve_type] += 1
        edge_convexity_counter[convexity] += 1

    result = {
        "num_original_nodes": len(nodes),
        "num_original_links": len(links),
        "num_face_nodes": len(face_ids),
        "num_edge_entity_nodes": len(edge_entity_ids),
        "num_converted_directed_edges": len(face_edges),
        "num_converted_undirected_edges": len(face_edges) // 2,
        "face_feature_dim": len(face_features[0]) if face_features else 0,
        "edge_feature_dim": len(edge_features[0]) if edge_features else 0,
        "skipped_edge_entity_no_face": skipped_edge_entity_no_face,
        "skipped_edge_entity_one_face": skipped_edge_entity_one_face,
        "edge_entity_more_than_two_faces": edge_entity_more_than_two_faces,
        "face_type_counter": dict(face_type_counter),
        "edge_curve_type_counter": dict(edge_curve_type_counter),
        "edge_convexity_counter": dict(edge_convexity_counter),
    }

    return result


def find_body_files(dataset_dir: Path, max_files: int):
    body_files = []

    for path in sorted(dataset_dir.rglob("*.json")):
        data, error = safe_load_json(path)

        if error is not None:
            continue

        if is_body_graph_file(data):
            body_files.append(path)

        if len(body_files) >= max_files:
            break

    return body_files


def save_markdown_report(results, output_path: Path):
    total = Counter()

    face_type_total = Counter()
    curve_type_total = Counter()
    convexity_total = Counter()

    for item in results:
        for key in [
            "num_original_nodes",
            "num_original_links",
            "num_face_nodes",
            "num_edge_entity_nodes",
            "num_converted_directed_edges",
            "num_converted_undirected_edges",
            "skipped_edge_entity_no_face",
            "skipped_edge_entity_one_face",
            "edge_entity_more_than_two_faces",
        ]:
            total[key] += item[key]

        face_type_total.update(item["face_type_counter"])
        curve_type_total.update(item["edge_curve_type_counter"])
        convexity_total.update(item["edge_convexity_counter"])

    face_feature_dims = Counter(item["face_feature_dim"] for item in results)
    edge_feature_dims = Counter(item["edge_feature_dim"] for item in results)

    lines = []
    lines.append("# Face Graph Conversion Validation Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Inspected body files: {len(results)}")
    lines.append(f"- Total original nodes: {total['num_original_nodes']}")
    lines.append(f"- Total original links: {total['num_original_links']}")
    lines.append(f"- Total face nodes: {total['num_face_nodes']}")
    lines.append(f"- Total edge entity nodes: {total['num_edge_entity_nodes']}")
    lines.append(f"- Total converted directed face-face edges: {total['num_converted_directed_edges']}")
    lines.append(f"- Total converted undirected face-face edges: {total['num_converted_undirected_edges']}")
    lines.append("")
    lines.append("## Feature Dimensions")
    lines.append("")
    lines.append(f"- Face feature dimensions: {dict(face_feature_dims)}")
    lines.append(f"- Edge feature dimensions: {dict(edge_feature_dims)}")
    lines.append("")
    lines.append("## Skipped Edge Entity Nodes")
    lines.append("")
    lines.append(f"- No adjacent face: {total['skipped_edge_entity_no_face']}")
    lines.append(f"- Only one adjacent face: {total['skipped_edge_entity_one_face']}")
    lines.append(f"- More than two adjacent faces: {total['edge_entity_more_than_two_faces']}")
    lines.append("")
    lines.append("## Face Surface Types")
    lines.append("")
    for key, count in face_type_total.most_common():
        lines.append(f"- `{key}`: {count}")
    lines.append("")
    lines.append("## Edge Curve Types")
    lines.append("")
    for key, count in curve_type_total.most_common():
        lines.append(f"- `{key}`: {count}")
    lines.append("")
    lines.append("## Edge Convexity Types")
    lines.append("")
    for key, count in convexity_total.most_common():
        lines.append(f"- `{key}`: {count}")
    lines.append("")
    lines.append("## Per-file Samples")
    lines.append("")
    lines.append("| File | Original Nodes | Face Nodes | Edge Entity Nodes | Converted Undirected Edges | Face Dim | Edge Dim |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for item in results[:30]:
        file_name = Path(item["file"]).name
        lines.append(
            f"| `{file_name}` | "
            f"{item['num_original_nodes']} | "
            f"{item['num_face_nodes']} | "
            f"{item['num_edge_entity_nodes']} | "
            f"{item['num_converted_undirected_edges']} | "
            f"{item['face_feature_dim']} | "
            f"{item['edge_feature_dim']} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default=r"C:\so_zero\j1.0.0_all\joint",
    )

    parser.add_argument(
        "--max_files",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default=r"C:\so_zero\cad_pair_gnn\outputs\reports\face_graph_conversion_validation.json",
    )

    parser.add_argument(
        "--output_md",
        type=str,
        default=r"C:\so_zero\cad_pair_gnn\outputs\reports\face_graph_conversion_validation.md",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    print("=" * 80)
    print("[STEP] Finding body graph files")
    print("=" * 80)

    body_files = find_body_files(dataset_dir, max_files=args.max_files)

    print(f"Selected body graph files: {len(body_files)}")

    results = []

    print("\n" + "=" * 80)
    print("[STEP] Validating face graph conversion")
    print("=" * 80)

    for path in body_files:
        data, error = safe_load_json(path)

        if error is not None:
            continue

        result = convert_to_face_graph(data)
        result["file"] = str(path)
        results.append(result)

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    save_markdown_report(results, output_md)

    print("\n" + "=" * 80)
    print("[DONE] Face graph conversion validation completed")
    print("=" * 80)
    print(f"JSON report: {output_json}")
    print(f"Markdown report: {output_md}")


if __name__ == "__main__":
    main()
