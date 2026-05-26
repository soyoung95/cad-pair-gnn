
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
from pprint import pprint


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


def summarize_value(value):
    if isinstance(value, dict):
        return {
            "type": "dict",
            "num_keys": len(value),
            "keys": list(value.keys())[:30],
        }

    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "sample_types": [type(x).__name__ for x in value[:5]],
            "sample_values": value[:5] if len(value) <= 10 else value[:5],
        }

    return {
        "type": type(value).__name__,
        "value": value,
    }


def collect_keys_from_dict(obj, prefix=""):
    keys = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            keys.append(full_key)

            if isinstance(value, dict):
                keys.extend(collect_keys_from_dict(value, full_key))

    return keys


def find_body_graph_files(dataset_dir: Path, max_files: int):
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


def inspect_body_file(path: Path):
    data, error = safe_load_json(path)

    if error is not None:
        return None

    nodes = data.get("nodes", [])
    links = data.get("links", [])
    graph = data.get("graph", {})
    properties = data.get("properties", {})

    node_key_counter = Counter()
    link_key_counter = Counter()
    node_nested_key_counter = Counter()
    link_nested_key_counter = Counter()

    node_type_counter = Counter()
    link_type_counter = Counter()

    for node in nodes:
        if isinstance(node, dict):
            node_key_counter.update(node.keys())

            nested_keys = collect_keys_from_dict(node)
            node_nested_key_counter.update(nested_keys)

            for candidate_key in ["type", "entity_type", "surface_type", "geometry_type", "label"]:
                if candidate_key in node:
                    node_type_counter[str(node[candidate_key])] += 1

    for link in links:
        if isinstance(link, dict):
            link_key_counter.update(link.keys())

            nested_keys = collect_keys_from_dict(link)
            link_nested_key_counter.update(nested_keys)

            for candidate_key in ["type", "entity_type", "curve_type", "geometry_type", "convexity"]:
                if candidate_key in link:
                    link_type_counter[str(link[candidate_key])] += 1

    sample_nodes = nodes[:5]
    sample_links = links[:5]

    result = {
        "file": str(path),
        "num_nodes": len(nodes),
        "num_links": len(links),
        "graph_summary": summarize_value(graph),
        "properties_summary": summarize_value(properties),
        "node_key_counter": dict(node_key_counter),
        "link_key_counter": dict(link_key_counter),
        "node_nested_key_counter": dict(node_nested_key_counter),
        "link_nested_key_counter": dict(link_nested_key_counter),
        "node_type_counter": dict(node_type_counter),
        "link_type_counter": dict(link_type_counter),
        "sample_nodes": sample_nodes,
        "sample_links": sample_links,
    }

    return result


def make_markdown_report(results, output_path: Path):
    lines = []

    lines.append("# Body Graph Feature Inspection Report")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Number of inspected body graph files: {len(results)}")
    lines.append("")

    total_node_key_counter = Counter()
    total_link_key_counter = Counter()
    total_node_nested_key_counter = Counter()
    total_link_nested_key_counter = Counter()
    total_node_type_counter = Counter()
    total_link_type_counter = Counter()

    for item in results:
        total_node_key_counter.update(item["node_key_counter"])
        total_link_key_counter.update(item["link_key_counter"])
        total_node_nested_key_counter.update(item["node_nested_key_counter"])
        total_link_nested_key_counter.update(item["link_nested_key_counter"])
        total_node_type_counter.update(item["node_type_counter"])
        total_link_type_counter.update(item["link_type_counter"])

    lines.append("## Node Top-Level Keys")
    lines.append("")
    for key, count in total_node_key_counter.most_common(50):
        lines.append(f"- `{key}`: {count}")
    lines.append("")

    lines.append("## Link Top-Level Keys")
    lines.append("")
    for key, count in total_link_key_counter.most_common(50):
        lines.append(f"- `{key}`: {count}")
    lines.append("")

    lines.append("## Node Nested Keys")
    lines.append("")
    for key, count in total_node_nested_key_counter.most_common(80):
        lines.append(f"- `{key}`: {count}")
    lines.append("")

    lines.append("## Link Nested Keys")
    lines.append("")
    for key, count in total_link_nested_key_counter.most_common(80):
        lines.append(f"- `{key}`: {count}")
    lines.append("")

    lines.append("## Node Type Candidates")
    lines.append("")
    if len(total_node_type_counter) == 0:
        lines.append("- No direct node type candidate was found.")
    else:
        for key, count in total_node_type_counter.most_common(50):
            lines.append(f"- `{key}`: {count}")
    lines.append("")

    lines.append("## Link Type Candidates")
    lines.append("")
    if len(total_link_type_counter) == 0:
        lines.append("- No direct link type candidate was found.")
    else:
        for key, count in total_link_type_counter.most_common(50):
            lines.append(f"- `{key}`: {count}")
    lines.append("")

    lines.append("## Sample Files")
    lines.append("")

    for index, item in enumerate(results[:5], start=1):
        lines.append(f"### Sample {index}")
        lines.append("")
        lines.append(f"- File: `{item['file']}`")
        lines.append(f"- Number of nodes: {item['num_nodes']}")
        lines.append(f"- Number of links: {item['num_links']}")
        lines.append("")

        lines.append("#### Sample Nodes")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(item["sample_nodes"], ensure_ascii=False, indent=2)[:6000])
        lines.append("```")
        lines.append("")

        lines.append("#### Sample Links")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(item["sample_links"], ensure_ascii=False, indent=2)[:6000])
        lines.append("```")
        lines.append("")

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
        default=20,
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default=r"C:\so_zero\cad_pair_gnn\outputs\reports\body_graph_feature_report.json",
    )

    parser.add_argument(
        "--output_md",
        type=str,
        default=r"C:\so_zero\cad_pair_gnn\outputs\reports\body_graph_feature_report.md",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    print("=" * 80)
    print("[STEP] Finding body graph files")
    print("=" * 80)

    body_files = find_body_graph_files(
        dataset_dir=dataset_dir,
        max_files=args.max_files,
    )

    print(f"Body graph files selected: {len(body_files)}")

    results = []

    print("\n" + "=" * 80)
    print("[STEP] Inspecting body graph files")
    print("=" * 80)

    for path in body_files:
        print(f"Inspecting: {path}")
        result = inspect_body_file(path)

        if result is not None:
            results.append(result)

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    make_markdown_report(results, output_md)

    print("\n" + "=" * 80)
    print("[DONE] Body graph feature inspection completed")
    print("=" * 80)
    print(f"JSON report: {output_json}")
    print(f"Markdown report: {output_md}")


if __name__ == "__main__":
    main()
