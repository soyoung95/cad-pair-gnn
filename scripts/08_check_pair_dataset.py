
import argparse
import json
import math
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm


FACE_FEATURE_DIM = 12
EDGE_FEATURE_DIM = 16


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def torch_load_graph(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def check_graph_data(data, body_id):
    errors = []

    if not hasattr(data, "x"):
        errors.append("missing x")
        return errors

    if not hasattr(data, "edge_index"):
        errors.append("missing edge_index")
        return errors

    if not hasattr(data, "edge_attr"):
        errors.append("missing edge_attr")
        return errors

    if data.x.dim() != 2:
        errors.append(f"invalid x dim: {tuple(data.x.shape)}")

    if data.x.size(1) != FACE_FEATURE_DIM:
        errors.append(f"invalid x feature dim: {tuple(data.x.shape)}")

    if data.edge_index.dim() != 2:
        errors.append(f"invalid edge_index dim: {tuple(data.edge_index.shape)}")

    if data.edge_index.size(0) != 2:
        errors.append(f"invalid edge_index shape: {tuple(data.edge_index.shape)}")

    if data.edge_attr.dim() != 2:
        errors.append(f"invalid edge_attr dim: {tuple(data.edge_attr.shape)}")

    if data.edge_attr.size(1) != EDGE_FEATURE_DIM:
        errors.append(f"invalid edge_attr feature dim: {tuple(data.edge_attr.shape)}")

    if data.edge_index.size(1) != data.edge_attr.size(0):
        errors.append(
            f"edge count mismatch: edge_index={data.edge_index.size(1)}, "
            f"edge_attr={data.edge_attr.size(0)}"
        )

    if torch.isnan(data.x).any():
        errors.append("x contains NaN")

    if torch.isinf(data.x).any():
        errors.append("x contains Inf")

    if torch.isnan(data.edge_attr).any():
        errors.append("edge_attr contains NaN")

    if torch.isinf(data.edge_attr).any():
        errors.append("edge_attr contains Inf")

    if data.x.size(0) == 0:
        errors.append("graph has zero face nodes")

    if data.edge_index.numel() > 0:
        max_index = int(data.edge_index.max().item())
        min_index = int(data.edge_index.min().item())

        if min_index < 0:
            errors.append(f"edge_index has negative index: {min_index}")

        if max_index >= data.x.size(0):
            errors.append(
                f"edge_index out of range: max={max_index}, num_nodes={data.x.size(0)}"
            )

    return errors


class PairGraphDataset(Dataset):
    def __init__(self, pair_json_path, graph_dir, limit=None):
        self.pair_json_path = Path(pair_json_path)
        self.graph_dir = Path(graph_dir)

        self.samples = load_json(self.pair_json_path)

        if limit is not None:
            self.samples = self.samples[:limit]

    def __len__(self):
        return len(self.samples)

    def get_graph_path(self, body_id):
        return self.graph_dir / f"{body_id}.pt"

    def __getitem__(self, index):
        sample = self.samples[index]

        body_one = sample["body_one"]
        body_two = sample["body_two"]
        label = float(sample["label"])

        graph_one_path = self.get_graph_path(body_one)
        graph_two_path = self.get_graph_path(body_two)

        graph_one = torch_load_graph(graph_one_path)
        graph_two = torch_load_graph(graph_two_path)

        graph_one.body_id = body_one
        graph_two.body_id = body_two

        y = torch.tensor([label], dtype=torch.float32)

        return graph_one, graph_two, y


def inspect_pair_file(pair_json_path, graph_dir, split_name, limit=None):
    pair_json_path = Path(pair_json_path)
    graph_dir = Path(graph_dir)

    samples = load_json(pair_json_path)

    if limit is not None:
        samples_to_check = samples[:limit]
    else:
        samples_to_check = samples

    label_counter = Counter()
    missing_graphs = []
    graph_error_examples = []

    total_pairs = len(samples_to_check)
    checked_graph_count = 0

    total_nodes_a = 0
    total_nodes_b = 0
    total_edges_a = 0
    total_edges_b = 0

    zero_edge_graphs = 0

    print("=" * 80)
    print(f"[STEP] Checking split: {split_name}")
    print("=" * 80)
    print(f"Pair file: {pair_json_path}")
    print(f"Samples to check: {total_pairs}")

    for sample in tqdm(samples_to_check):
        body_one = sample["body_one"]
        body_two = sample["body_two"]
        label = int(sample["label"])

        label_counter[label] += 1

        for body_id, side in [(body_one, "body_one"), (body_two, "body_two")]:
            graph_path = graph_dir / f"{body_id}.pt"

            if not graph_path.exists():
                if len(missing_graphs) < 30:
                    missing_graphs.append(
                        {
                            "split": split_name,
                            "body_id": body_id,
                            "side": side,
                            "path": str(graph_path),
                        }
                    )
                continue

            try:
                graph = torch_load_graph(graph_path)
                errors = check_graph_data(graph, body_id)

                if errors:
                    if len(graph_error_examples) < 30:
                        graph_error_examples.append(
                            {
                                "split": split_name,
                                "body_id": body_id,
                                "side": side,
                                "path": str(graph_path),
                                "errors": errors,
                            }
                        )
                else:
                    checked_graph_count += 1

                    if side == "body_one":
                        total_nodes_a += int(graph.x.size(0))
                        total_edges_a += int(graph.edge_index.size(1))
                    else:
                        total_nodes_b += int(graph.x.size(0))
                        total_edges_b += int(graph.edge_index.size(1))

                    if graph.edge_index.size(1) == 0:
                        zero_edge_graphs += 1

            except Exception as e:
                if len(graph_error_examples) < 30:
                    graph_error_examples.append(
                        {
                            "split": split_name,
                            "body_id": body_id,
                            "side": side,
                            "path": str(graph_path),
                            "errors": [str(e)],
                        }
                    )

    avg_nodes_a = total_nodes_a / total_pairs if total_pairs > 0 else 0.0
    avg_nodes_b = total_nodes_b / total_pairs if total_pairs > 0 else 0.0
    avg_edges_a = total_edges_a / total_pairs if total_pairs > 0 else 0.0
    avg_edges_b = total_edges_b / total_pairs if total_pairs > 0 else 0.0

    result = {
        "split": split_name,
        "pair_file": str(pair_json_path),
        "total_pairs_checked": total_pairs,
        "label_counts": dict(label_counter),
        "missing_graph_count": len(missing_graphs),
        "missing_graph_examples": missing_graphs,
        "graph_error_count": len(graph_error_examples),
        "graph_error_examples": graph_error_examples,
        "checked_graph_count": checked_graph_count,
        "zero_edge_graphs": zero_edge_graphs,
        "avg_nodes_body_one": avg_nodes_a,
        "avg_nodes_body_two": avg_nodes_b,
        "avg_directed_edges_body_one": avg_edges_a,
        "avg_directed_edges_body_two": avg_edges_b,
    }

    print(f"Label counts: {dict(label_counter)}")
    print(f"Missing graph examples: {len(missing_graphs)}")
    print(f"Graph error examples: {len(graph_error_examples)}")
    print(f"Zero-edge graphs: {zero_edge_graphs}")
    print(f"Average nodes body_one: {avg_nodes_a:.2f}")
    print(f"Average nodes body_two: {avg_nodes_b:.2f}")
    print(f"Average directed edges body_one: {avg_edges_a:.2f}")
    print(f"Average directed edges body_two: {avg_edges_b:.2f}")

    return result


def check_dataloader(pair_json_path, graph_dir, batch_size, limit, num_workers):
    print("\n" + "=" * 80)
    print("[STEP] Checking PyG DataLoader")
    print("=" * 80)

    dataset = PairGraphDataset(
        pair_json_path=pair_json_path,
        graph_dir=graph_dir,
        limit=limit,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    batch = next(iter(loader))

    graph_a, graph_b, y = batch

    print(f"Batch graph A x shape: {tuple(graph_a.x.shape)}")
    print(f"Batch graph A edge_index shape: {tuple(graph_a.edge_index.shape)}")
    print(f"Batch graph A edge_attr shape: {tuple(graph_a.edge_attr.shape)}")
    print(f"Batch graph A batch shape: {tuple(graph_a.batch.shape)}")

    print(f"Batch graph B x shape: {tuple(graph_b.x.shape)}")
    print(f"Batch graph B edge_index shape: {tuple(graph_b.edge_index.shape)}")
    print(f"Batch graph B edge_attr shape: {tuple(graph_b.edge_attr.shape)}")
    print(f"Batch graph B batch shape: {tuple(graph_b.batch.shape)}")

    print(f"Batch y shape: {tuple(y.shape)}")
    print(f"Batch y values: {y.view(-1).tolist()}")

    result = {
        "batch_size": batch_size,
        "graph_a_x_shape": list(graph_a.x.shape),
        "graph_a_edge_index_shape": list(graph_a.edge_index.shape),
        "graph_a_edge_attr_shape": list(graph_a.edge_attr.shape),
        "graph_a_batch_shape": list(graph_a.batch.shape),
        "graph_b_x_shape": list(graph_b.x.shape),
        "graph_b_edge_index_shape": list(graph_b.edge_index.shape),
        "graph_b_edge_attr_shape": list(graph_b.edge_attr.shape),
        "graph_b_batch_shape": list(graph_b.batch.shape),
        "y_shape": list(y.shape),
        "y_values": y.view(-1).tolist(),
    }

    return result


def make_markdown_report(summary, output_path):
    lines = []

    lines.append("# Pair Dataset Check Report")
    lines.append("")

    lines.append("## Split Checks")
    lines.append("")

    for item in summary["split_results"]:
        lines.append(f"### {item['split']}")
        lines.append("")
        lines.append(f"- Pair file: `{item['pair_file']}`")
        lines.append(f"- Total pairs checked: {item['total_pairs_checked']}")
        lines.append(f"- Label counts: `{item['label_counts']}`")
        lines.append(f"- Missing graph examples: {item['missing_graph_count']}")
        lines.append(f"- Graph error examples: {item['graph_error_count']}")
        lines.append(f"- Checked graph count: {item['checked_graph_count']}")
        lines.append(f"- Zero-edge graphs: {item['zero_edge_graphs']}")
        lines.append(f"- Average nodes body_one: {item['avg_nodes_body_one']:.4f}")
        lines.append(f"- Average nodes body_two: {item['avg_nodes_body_two']:.4f}")
        lines.append(f"- Average directed edges body_one: {item['avg_directed_edges_body_one']:.4f}")
        lines.append(f"- Average directed edges body_two: {item['avg_directed_edges_body_two']:.4f}")
        lines.append("")

        if item["missing_graph_examples"]:
            lines.append("#### Missing Graph Examples")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(item["missing_graph_examples"], ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")

        if item["graph_error_examples"]:
            lines.append("#### Graph Error Examples")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(item["graph_error_examples"], ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")

    lines.append("## DataLoader Check")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["dataloader_result"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pair_index_dir",
        type=str,
        default=r"C:\so_zero\cad_pair_gnn\outputs\pair_index",
    )

    parser.add_argument(
        "--graph_dir",
        type=str,
        default=r"C:\so_zero\cad_pair_gnn\processed\body_graphs",
    )

    parser.add_argument(
        "--report_json",
        type=str,
        default=r"C:\so_zero\cad_pair_gnn\outputs\reports\pair_dataset_check.json",
    )

    parser.add_argument(
        "--report_md",
        type=str,
        default=r"C:\so_zero\cad_pair_gnn\outputs\reports\pair_dataset_check.md",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pairs to check per split. Use no value to check all samples.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    pair_index_dir = Path(args.pair_index_dir)
    graph_dir = Path(args.graph_dir)
    report_json_path = Path(args.report_json)
    report_md_path = Path(args.report_md)

    split_files = {
        "train": pair_index_dir / "train_pairs.json",
        "validation": pair_index_dir / "validation_pairs.json",
        "test": pair_index_dir / "test_pairs.json",
    }

    split_results = []

    for split_name, pair_path in split_files.items():
        result = inspect_pair_file(
            pair_json_path=pair_path,
            graph_dir=graph_dir,
            split_name=split_name,
            limit=args.limit,
        )
        split_results.append(result)

    dataloader_result = check_dataloader(
        pair_json_path=split_files["train"],
        graph_dir=graph_dir,
        batch_size=args.batch_size,
        limit=64,
        num_workers=args.num_workers,
    )

    summary = {
        "pair_index_dir": str(pair_index_dir),
        "graph_dir": str(graph_dir),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "split_results": split_results,
        "dataloader_result": dataloader_result,
    }

    save_json(report_json_path, summary)
    make_markdown_report(summary, report_md_path)

    print("\n" + "=" * 80)
    print("[DONE] Pair dataset check completed")
    print("=" * 80)
    print(f"Report JSON: {report_json_path}")
    print(f"Report Markdown: {report_md_path}")


if __name__ == "__main__":
    main()
