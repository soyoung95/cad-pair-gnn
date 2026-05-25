from __future__ import annotations


import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple


import torch
from tqdm import tqdm




DEFAULT_BASE_DIR = Path("processed/body_graphs_v2_geometry")
DEFAULT_EXTRA_DIR = Path("processed/body_graphs_v6_full")
DEFAULT_OUTPUT_DIR = Path("processed/body_graphs_v8_v2_plus_v6_teacher")
DEFAULT_REPORT_PATH = Path("outputs/reports/v8_merge_v2_v6_summary.json")


EPS = 1e-8




FACE_EXTRA_NAMES = [
    "valid_ratio_from_trimming_mask",
    "normal_concentration",
    "mean_point_x",
    "mean_point_y",
    "mean_point_z",
    "mean_normal_x",
    "mean_normal_y",
    "mean_normal_z",
    "point_planarity",
    "log_spread_mean",
    "log_spread_max",
]


EDGE_EXTRA_NAMES = [
    "log_edge_spread_mean",
    "mean_tangent_x",
    "mean_tangent_y",
    "mean_tangent_z",
    "tangent_concentration",
    "mean_edge_point_x",
    "mean_edge_point_y",
    "mean_edge_point_z",
]




def load_graph(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)




def get_extra_slices(extra_graph) -> Tuple[slice, slice]:
    face_dim = int(extra_graph.x.shape[1])
    edge_dim = int(extra_graph.edge_attr.shape[1])


    if face_dim >= 21 and edge_dim >= 22:
        return slice(10, 21), slice(14, 22)


    raise ValueError(
        f"Unsupported extra graph dims: face_dim={face_dim}, edge_dim={edge_dim}. "
        "Expected at least face_dim=21 and edge_dim=22."
    )




def validate_pair(base_graph, extra_graph, filename: str) -> None:
    if not hasattr(base_graph, "x") or not hasattr(base_graph, "edge_attr"):
        raise ValueError(f"{filename}: base graph missing x or edge_attr")
    if not hasattr(extra_graph, "x") or not hasattr(extra_graph, "edge_attr"):
        raise ValueError(f"{filename}: extra graph missing x or edge_attr")


    if base_graph.x.dim() != 2 or extra_graph.x.dim() != 2:
        raise ValueError(f"{filename}: x must be 2D")
    if base_graph.edge_attr.dim() != 2 or extra_graph.edge_attr.dim() != 2:
        raise ValueError(f"{filename}: edge_attr must be 2D")


    if base_graph.x.shape[0] != extra_graph.x.shape[0]:
        raise ValueError(
            f"{filename}: node count mismatch: "
            f"base={base_graph.x.shape[0]}, extra={extra_graph.x.shape[0]}"
        )


    if base_graph.edge_attr.shape[0] != extra_graph.edge_attr.shape[0]:
        raise ValueError(
            f"{filename}: edge count mismatch: "
            f"base={base_graph.edge_attr.shape[0]}, extra={extra_graph.edge_attr.shape[0]}"
        )


    if hasattr(base_graph, "edge_index") and hasattr(extra_graph, "edge_index"):
        if base_graph.edge_index.shape != extra_graph.edge_index.shape:
            raise ValueError(
                f"{filename}: edge_index shape mismatch: "
                f"base={tuple(base_graph.edge_index.shape)}, "
                f"extra={tuple(extra_graph.edge_index.shape)}"
            )
        if not torch.equal(base_graph.edge_index.cpu(), extra_graph.edge_index.cpu()):
            raise ValueError(f"{filename}: edge_index mismatch")




def update_stats(
    sum_vec: torch.Tensor,
    sum_sq_vec: torch.Tensor,
    count: int,
    values: torch.Tensor,
):
    if values.numel() == 0:
        return sum_vec, sum_sq_vec, count


    v = values.float()
    sum_vec += v.sum(dim=0).double()
    sum_sq_vec += (v.double() ** 2).sum(dim=0)
    count += int(v.shape[0])
    return sum_vec, sum_sq_vec, count




def finalize_stats(
    sum_vec: torch.Tensor,
    sum_sq_vec: torch.Tensor,
    count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if count <= 0:
        mean = torch.zeros_like(sum_vec, dtype=torch.float32)
        std = torch.ones_like(sum_vec, dtype=torch.float32)
        return mean, std


    mean = sum_vec / count
    var = (sum_sq_vec / count) - (mean ** 2)
    var = torch.clamp(var, min=EPS)
    std = torch.sqrt(var)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)


    return mean.float(), std.float()




def standardize(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.float()
    return (values.float() - mean.view(1, -1)) / std.view(1, -1)




def make_schema(base_node_dim: int, base_edge_dim: int, standardize_extra: bool) -> Dict[str, object]:
    return {
        "version": "v8_v2_plus_v6_teacher",
        "description": (
            "Base v2_geometry features are preserved. Teacher-requested v6 features "
            "from face points/normals/trimming_mask and edge points/tangents are appended."
        ),
        "base_node_dim": base_node_dim,
        "base_edge_dim": base_edge_dim,
        "extra_node_dim": len(FACE_EXTRA_NAMES),
        "extra_edge_dim": len(EDGE_EXTRA_NAMES),
        "total_node_dim": base_node_dim + len(FACE_EXTRA_NAMES),
        "total_edge_dim": base_edge_dim + len(EDGE_EXTRA_NAMES),
        "standardize_extra": standardize_extra,
        "node_feature_layout": {
            "base_v2_geometry": [0, base_node_dim],
            "v6_teacher_features": {
                name: base_node_dim + i for i, name in enumerate(FACE_EXTRA_NAMES)
            },
        },
        "edge_feature_layout": {
            "base_v2_geometry": [0, base_edge_dim],
            "v6_teacher_features": {
                name: base_edge_dim + i for i, name in enumerate(EDGE_EXTRA_NAMES)
            },
        },
    }




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge v2_geometry graphs with v6 teacher-required features."
    )
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--extra_dir", type=Path, default=DEFAULT_EXTRA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report_path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no_standardize_extra",
        action="store_true",
        help="Do not standardize appended v6 teacher features.",
    )
    args = parser.parse_args()


    base_dir: Path = args.base_dir
    extra_dir: Path = args.extra_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)


    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    if not extra_dir.exists():
        raise FileNotFoundError(f"Extra directory not found: {extra_dir}")


    base_files = sorted([p for p in base_dir.iterdir() if p.suffix == ".pt"])
    if args.limit and args.limit > 0:
        base_files = base_files[: args.limit]


    if not base_files:
        raise FileNotFoundError(f"No .pt files found in base_dir: {base_dir}")


    print("=" * 80)
    print("[v8] Merge v2_geometry + v6 teacher features")
    print("=" * 80)
    print(f"Base dir     : {base_dir}")
    print(f"Extra dir    : {extra_dir}")
    print(f"Output dir   : {output_dir}")
    print(f"Files        : {len(base_files)}")
    print(f"Overwrite    : {args.overwrite}")
    print(f"Std extra    : {not args.no_standardize_extra}")
    print("=" * 80)


    standardize_extra = not args.no_standardize_extra


    first_base = load_graph(base_files[0])
    first_extra_path = extra_dir / base_files[0].name
    if not first_extra_path.exists():
        raise FileNotFoundError(f"Matching extra file not found: {first_extra_path}")


    first_extra = load_graph(first_extra_path)
    validate_pair(first_base, first_extra, base_files[0].name)


    face_slice, edge_slice = get_extra_slices(first_extra)


    base_node_dim = int(first_base.x.shape[1])
    base_edge_dim = int(first_base.edge_attr.shape[1])
    extra_node_dim = int(first_extra.x[:, face_slice].shape[1])
    extra_edge_dim = int(first_extra.edge_attr[:, edge_slice].shape[1])


    if extra_node_dim != len(FACE_EXTRA_NAMES):
        raise ValueError(f"Unexpected face extra dim: {extra_node_dim}")
    if extra_edge_dim != len(EDGE_EXTRA_NAMES):
        raise ValueError(f"Unexpected edge extra dim: {extra_edge_dim}")


    print(f"Base node dim : {base_node_dim}")
    print(f"Base edge dim : {base_edge_dim}")
    print(f"Extra node dim: {extra_node_dim}")
    print(f"Extra edge dim: {extra_edge_dim}")
    print(f"Total node dim: {base_node_dim + extra_node_dim}")
    print(f"Total edge dim: {base_edge_dim + extra_edge_dim}")


    node_sum = torch.zeros(extra_node_dim, dtype=torch.float64)
    node_sum_sq = torch.zeros(extra_node_dim, dtype=torch.float64)
    edge_sum = torch.zeros(extra_edge_dim, dtype=torch.float64)
    edge_sum_sq = torch.zeros(extra_edge_dim, dtype=torch.float64)
    node_count = 0
    edge_count = 0


    missing_extra = 0
    failed_stats = 0
    stats_examples: List[Dict[str, str]] = []


    for base_path in tqdm(base_files, desc="Pass 1/2 stats", dynamic_ncols=True):
        extra_path = extra_dir / base_path.name
        if not extra_path.exists():
            missing_extra += 1
            if len(stats_examples) < 20:
                stats_examples.append({"file": base_path.name, "error": "missing extra file"})
            continue


        try:
            base_graph = load_graph(base_path)
            extra_graph = load_graph(extra_path)
            validate_pair(base_graph, extra_graph, base_path.name)


            node_extra = extra_graph.x[:, face_slice]
            edge_extra = extra_graph.edge_attr[:, edge_slice]


            node_sum, node_sum_sq, node_count = update_stats(
                node_sum, node_sum_sq, node_count, node_extra
            )
            edge_sum, edge_sum_sq, edge_count = update_stats(
                edge_sum, edge_sum_sq, edge_count, edge_extra
            )
        except Exception as exc:
            failed_stats += 1
            if len(stats_examples) < 20:
                stats_examples.append({"file": base_path.name, "error": repr(exc)})


    node_mean, node_std = finalize_stats(node_sum, node_sum_sq, node_count)
    edge_mean, edge_std = finalize_stats(edge_sum, edge_sum_sq, edge_count)


    if not standardize_extra:
        node_mean = torch.zeros_like(node_mean)
        node_std = torch.ones_like(node_std)
        edge_mean = torch.zeros_like(edge_mean)
        edge_std = torch.ones_like(edge_std)


    processed = 0
    skipped = 0
    failed = 0
    missing = 0
    examples: List[Dict[str, str]] = []


    schema = make_schema(base_node_dim, base_edge_dim, standardize_extra)


    for base_path in tqdm(base_files, desc="Pass 2/2 merge", dynamic_ncols=True):
        out_path = output_dir / base_path.name


        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue


        extra_path = extra_dir / base_path.name
        if not extra_path.exists():
            missing += 1
            if len(examples) < 20:
                examples.append({"file": base_path.name, "error": "missing extra file"})
            continue


        try:
            base_graph = load_graph(base_path)
            extra_graph = load_graph(extra_path)
            validate_pair(base_graph, extra_graph, base_path.name)


            node_extra = standardize(extra_graph.x[:, face_slice], node_mean, node_std)
            edge_extra = standardize(extra_graph.edge_attr[:, edge_slice], edge_mean, edge_std)


            merged = copy.deepcopy(base_graph)
            merged.x = torch.cat([base_graph.x.float(), node_extra], dim=1)
            merged.edge_attr = torch.cat([base_graph.edge_attr.float(), edge_extra], dim=1)


            merged.feature_schema = schema
            merged.v8_source_base = str(base_dir)
            merged.v8_source_extra = str(extra_dir)
            merged.v8_standardize_extra = standardize_extra


            torch.save(merged, out_path)
            processed += 1
        except Exception as exc:
            failed += 1
            if len(examples) < 20:
                examples.append({"file": base_path.name, "error": repr(exc)})


    report = {
        "base_dir": str(base_dir),
        "extra_dir": str(extra_dir),
        "output_dir": str(output_dir),
        "files": len(base_files),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "missing": missing,
        "missing_extra_in_stats": missing_extra,
        "failed_stats": failed_stats,
        "node_count_for_stats": node_count,
        "edge_count_for_stats": edge_count,
        "base_node_dim": base_node_dim,
        "base_edge_dim": base_edge_dim,
        "extra_node_dim": extra_node_dim,
        "extra_edge_dim": extra_edge_dim,
        "total_node_dim": base_node_dim + extra_node_dim,
        "total_edge_dim": base_edge_dim + extra_edge_dim,
        "standardize_extra": standardize_extra,
        "face_extra_names": FACE_EXTRA_NAMES,
        "edge_extra_names": EDGE_EXTRA_NAMES,
        "node_extra_mean": node_mean.tolist(),
        "node_extra_std": node_std.tolist(),
        "edge_extra_mean": edge_mean.tolist(),
        "edge_extra_std": edge_std.tolist(),
        "examples": examples,
        "stats_examples": stats_examples,
    }


    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


    print("=" * 80)
    print("[DONE] v8 merge completed")
    print(f"  Processed : {processed}")
    print(f"  Skipped   : {skipped}")
    print(f"  Missing   : {missing}")
    print(f"  Failed    : {failed}")
    print(f"  Output    : {output_dir}")
    print(f"  Report    : {args.report_path}")
    print("=" * 80)




if __name__ == "__main__":
    main()


