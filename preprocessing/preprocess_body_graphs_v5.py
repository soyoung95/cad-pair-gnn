import argparse
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations


import torch
from torch_geometric.data import Data
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# 카테고리 정의 (v4 와 동일)
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# 교수님 지정 feature 차원
#
# Face (10):
#   surface_type one-hot (7) + reversed (1) + log_area (1) + log_radius (1)
#
# Edge (14):
#   curve_type one-hot (7) + reversed (1) + log_length (1)
#   + convexity one-hot (4) + log_radius (1)
# ──────────────────────────────────────────────────────────────────────────────
FACE_FEATURE_DIM = 10
EDGE_FEATURE_DIM = 14




# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────────────────────
def safe_load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)




def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)




def is_body_graph_file(data):
    if not isinstance(data, dict):
        return False
    return {"nodes", "links", "graph", "properties"}.issubset(set(data.keys()))




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
    return [1.0 if value == c else 0.0 for c in categories]




def get_raw_array_2d(node, key):
    """JSON 배열 → float32 [N, 3] tensor. 실패 시 [0, 3] 빈 tensor."""
    import numpy as np
    value = node.get(key, [])
    if not value:
        return torch.zeros((0, 3), dtype=torch.float32)
    try:
        arr = np.array(value, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 3)
        if arr.ndim != 2 or arr.shape[1] != 3:
            return torch.zeros((0, 3), dtype=torch.float32)
        return torch.from_numpy(arr)
    except Exception:
        return torch.zeros((0, 3), dtype=torch.float32)




def get_raw_array_1d(node, key):
    """JSON 배열 → float32 [N] tensor. 실패 시 [0] 빈 tensor."""
    import numpy as np
    value = node.get(key, [])
    if not value:
        return torch.zeros((0,), dtype=torch.float32)
    try:
        arr = np.array(value, dtype=np.float32).flatten()
        return torch.from_numpy(arr)
    except Exception:
        return torch.zeros((0,), dtype=torch.float32)




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




# ──────────────────────────────────────────────────────────────────────────────
# 메인 변환 함수
# ──────────────────────────────────────────────────────────────────────────────
def convert_body_graph_to_pyg(data):
    nodes = data.get("nodes", [])
    links = data.get("links", [])


    # ── 노드 분류 ──────────────────────────────────────────────────────────────
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


    face_ids = sorted(face_ids)
    edge_entity_ids = sorted(edge_entity_ids)
    face_id_to_new_index = {fid: idx for idx, fid in enumerate(face_ids)}


    # ── Face 노드 피처 추출 ────────────────────────────────────────────────────
    face_features = []
    face_points_list        = []   # list of Tensor [N_i, 3]  (교수님 필수)
    face_normals_list       = []   # list of Tensor [N_i, 3]  (교수님 필수)
    face_trimming_mask_list = []   # list of Tensor [N_i]     (교수님 필수)
    face_type_counter = Counter()


    for face_id in face_ids:
        node = id_to_node[face_id]


        surface_type = node.get("surface_type", "UnknownSurfaceType")
        if surface_type not in FACE_TYPES:
            surface_type = "UnknownSurfaceType"


        area   = max(get_float(node, "area",   0.0), 0.0)
        radius = max(get_float(node, "radius", 0.0), 0.0)
        rev    = get_bool_float(node, "reversed", 0.0)


        # ── 교수님 지정 face feature (10 dims) ────────────────────────────────
        feat = []
        feat.extend(one_hot(surface_type, FACE_TYPES, "UnknownSurfaceType"))  # 7
        feat.append(rev)                  # 1  reversed
        feat.append(math.log1p(area))     # 1  area  (log 스케일로 안정화)
        feat.append(math.log1p(radius))   # 1  radius (0이면 log1p(0)=0)
        # 합계: 7 + 1 + 1 + 1 = 10


        assert len(feat) == FACE_FEATURE_DIM, f"Face dim mismatch: {len(feat)}"
        face_features.append(feat)
        face_type_counter[surface_type] += 1


        # 교수님 필수 항목: 배열 저장 (학습 시 PointNet 인코딩용)
        face_points_list.append(get_raw_array_2d(node, "points"))
        face_normals_list.append(get_raw_array_2d(node, "normals"))
        face_trimming_mask_list.append(get_raw_array_1d(node, "trimming_mask"))


    # ── Edge entity 피처 추출 ──────────────────────────────────────────────────
    adjacency = build_adjacency(links)


    edge_index_list        = []
    edge_feature_list      = []
    edge_to_entity_idx_list = []
    edge_entity_points_list   = []   # list of Tensor [M_j, 3]  (교수님 필수)
    edge_entity_tangents_list = []   # list of Tensor [M_j, 3]  (교수님 필수)


    edge_curve_type_counter = Counter()
    edge_convexity_counter  = Counter()
    skipped_no_adjacent_face  = 0
    skipped_one_adjacent_face = 0
    edge_entity_more_than_two = 0


    valid_entity_counter = 0   # dense index into edge_entity_points_list
    for entity_idx, edge_id in enumerate(edge_entity_ids):
        edge_node = id_to_node[edge_id]


        adjacent_face_ids = [
            nid for nid in adjacency.get(edge_id, [])
            if nid in face_id_to_new_index
        ]


        if len(adjacent_face_ids) == 0:
            skipped_no_adjacent_face += 1
            continue
        if len(adjacent_face_ids) == 1:
            skipped_one_adjacent_face += 1
            continue
        if len(adjacent_face_ids) > 2:
            edge_entity_more_than_two += 1


        curve_type = edge_node.get("curve_type", "UnknownCurveType")
        if curve_type not in CURVE_TYPES:
            curve_type = "UnknownCurveType"


        convexity = edge_node.get("convexity", "UnknownConvexity")
        if convexity not in CONVEXITY_TYPES:
            convexity = "UnknownConvexity"


        length = max(get_float(edge_node, "length", 0.0), 0.0)
        radius = max(get_float(edge_node, "radius", 0.0), 0.0)
        rev    = get_bool_float(edge_node, "reversed", 0.0)


        # ── 교수님 지정 edge feature (14 dims) ───────────────────────────────
        edge_feat = []
        edge_feat.extend(one_hot(curve_type, CURVE_TYPES, "UnknownCurveType"))       # 7
        edge_feat.append(rev)                   # 1  reversed
        edge_feat.append(math.log1p(length))    # 1  length
        edge_feat.extend(one_hot(convexity, CONVEXITY_TYPES, "UnknownConvexity"))    # 4
        edge_feat.append(math.log1p(radius))    # 1  radius
        # 합계: 7 + 1 + 1 + 4 + 1 = 14


        assert len(edge_feat) == EDGE_FEATURE_DIM, f"Edge dim mismatch: {len(edge_feat)}"


        # Required point cloud data stored per valid edge entity
        edge_entity_points_list.append(get_raw_array_2d(edge_node, "points"))
        edge_entity_tangents_list.append(get_raw_array_2d(edge_node, "tangents"))


        for face_a, face_b in combinations(sorted(adjacent_face_ids), 2):
            ia = face_id_to_new_index[face_a]
            ib = face_id_to_new_index[face_b]
            edge_index_list.append([ia, ib])
            edge_feature_list.append(edge_feat)
            edge_to_entity_idx_list.append(valid_entity_counter)  # dense index


            edge_index_list.append([ib, ia])
            edge_feature_list.append(edge_feat)
            edge_to_entity_idx_list.append(valid_entity_counter)  # dense index


        edge_curve_type_counter[curve_type] += 1
        edge_convexity_counter[convexity]   += 1
        valid_entity_counter += 1  # increment only for valid entities


    # ── Tensor 변환 ────────────────────────────────────────────────────────────
    if len(face_features) == 0:
        x = torch.empty((0, FACE_FEATURE_DIM), dtype=torch.float32)
    else:
        x = torch.tensor(face_features, dtype=torch.float32)


    if len(edge_index_list) == 0:
        edge_index         = torch.empty((2, 0), dtype=torch.long)
        edge_attr          = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)
        edge_to_entity_idx = torch.empty((0,), dtype=torch.long)
    else:
        edge_index         = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr          = torch.tensor(edge_feature_list, dtype=torch.float32)
        edge_to_entity_idx = torch.tensor(edge_to_entity_idx_list, dtype=torch.long)


    # ── PyG Data 구성 ───────────────────────────────────────────────────────────
    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_to_entity_idx=edge_to_entity_idx,
        # ── 교수님 필수 항목 (학습 시 반드시 포함) ────────────────────────────
        face_points=face_points_list,                  # list of Tensor [N_i, 3]
        face_normals=face_normals_list,                # list of Tensor [N_i, 3]
        face_trimming_mask=face_trimming_mask_list,    # list of Tensor [N_i]
        edge_entity_points=edge_entity_points_list,    # list of Tensor [M_j, 3]
        edge_entity_tangents=edge_entity_tangents_list, # list of Tensor [M_j, 3]
    )


    stats = {
        "num_face_nodes":                len(face_ids),
        "num_edge_entity_nodes":         len(edge_entity_ids),
        "num_directed_edges":            int(edge_index.size(1)),
        "face_feature_dim":              FACE_FEATURE_DIM,
        "edge_feature_dim":              EDGE_FEATURE_DIM,
        "skipped_no_adjacent_face":      skipped_no_adjacent_face,
        "skipped_one_adjacent_face":     skipped_one_adjacent_face,
        "edge_entity_more_than_two":     edge_entity_more_than_two,
        "face_type_counter":             dict(face_type_counter),
        "edge_curve_type_counter":       dict(edge_curve_type_counter),
        "edge_convexity_counter":        dict(edge_convexity_counter),
    }
    return pyg_data, stats




# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Body graph preprocessing v5 (교수님 지정 slim 피처)"
    )
    parser.add_argument(
        "--body_graph_map",
        type=str,
        default=r"D:\soyoung\cad_pair_gnn\outputs\pair_index\body_graph_files.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"D:\soyoung\cad_pair_gnn\processed\body_graphs_v5_slim",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default=r"D:\soyoung\cad_pair_gnn\outputs\reports\pyg_preprocess_v5_summary.json",
    )
    parser.add_argument("--limit",     type=int,  default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    body_graph_map = safe_load_json(Path(args.body_graph_map))
    items = sorted(body_graph_map.items())
    if args.limit is not None:
        items = items[:args.limit]


    total      = len(items)
    processed  = 0
    skipped    = 0
    failed     = 0
    failed_ex  = []


    total_face_nodes     = 0
    total_directed_edges = 0
    graphs_zero_edges    = 0


    print("=" * 70)
    print("[v5] Body Graph Preprocessing — 교수님 지정 slim 피처")
    print(f"  Face x    : {FACE_FEATURE_DIM} dims  "
          f"(surface_type OH7 + reversed + log_area + log_radius)")
    print(f"  Edge attr : {EDGE_FEATURE_DIM} dims  "
          f"(curve_type OH7 + reversed + log_length + convexity OH4 + log_radius)")
    print(f"  + face_points / face_normals / face_trimming_mask")
    print(f"  + edge_entity_points / edge_entity_tangents  ← 교수님 필수")
    print("=" * 70)
    print(f"Total: {total}  |  Output: {output_dir}")


    for body_id, json_path_str in tqdm(items):
        json_path   = Path(json_path_str)
        output_path = output_dir / f"{body_id}.pt"


        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue


        try:
            raw_data = safe_load_json(json_path)
            if not is_body_graph_file(raw_data):
                raise ValueError("Not a valid body graph file.")


            pyg_data, stats = convert_body_graph_to_pyg(raw_data)
            torch.save(pyg_data, output_path)
            processed += 1


            total_face_nodes     += stats["num_face_nodes"]
            total_directed_edges += stats["num_directed_edges"]
            if stats["num_directed_edges"] == 0:
                graphs_zero_edges += 1


        except Exception as e:
            failed += 1
            if len(failed_ex) < 20:
                failed_ex.append({"body_id": body_id, "error": str(e)})


    summary = {
        "face_feature_dim":    FACE_FEATURE_DIM,
        "edge_feature_dim":    EDGE_FEATURE_DIM,
        "total":               total,
        "processed":           processed,
        "skipped":             skipped,
        "failed":              failed,
        "failed_examples":     failed_ex,
        "total_face_nodes":    total_face_nodes,
        "total_directed_edges": total_directed_edges,
        "graphs_zero_edges":   graphs_zero_edges,
        "output_dir":          str(output_dir),
    }
    save_json(Path(args.report_json), summary)


    print("\n" + "=" * 70)
    print("[DONE] v5 preprocessing completed")
    print(f"  Processed : {processed}")
    print(f"  Skipped   : {skipped}")
    print(f"  Failed    : {failed}")
    print(f"  Report    : {args.report_json}")




if __name__ == "__main__":
    main()


