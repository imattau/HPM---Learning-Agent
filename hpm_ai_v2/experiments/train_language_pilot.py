"""Run the encoder-free HPM language learning pilot."""
from __future__ import annotations

import argparse
import json
from hpm_ai_v2.training.language_prediction import run_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, action="append", help="source JSONL; repeat to mix corpora")
    parser.add_argument("--output", required=True, help="directory for checkpoint and held-out metrics")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1,
                        help="bounded worker count for document counting and independent validation/evaluation")
    parser.add_argument("--min-word-count", type=int, default=1,
                        help="map training words below this frequency to <unk> in HPM and matched controls")
    parser.add_argument("--class-nodes", action="store_true",
                        help="enable opt-in HFN word-class nodes; reports validation selection and surface metrics")
    args = parser.parse_args()
    corpora = args.corpus if len(args.corpus) > 1 else args.corpus[0]
    result = run_pilot(corpora, args.output, args.seed, args.order, args.workers,
                       args.min_word_count, class_nodes=args.class_nodes)
    summary = {
        "state": "complete",
        "metrics": f"{args.output}/metrics.json",
        "checkpoint": f"{args.output}/hpm_language_checkpoint.json",
        "workers": result["workers"],
        "min_word_count": result["min_word_count"],
        "split_documents": result["split_documents"],
        "selected_hpm_alpha": result["selected_hpm_alpha"],
        "selected_character_alpha": result["selected_character_alpha"],
        "hpm_test_cross_entropy_nats": result["hpm"]["test"]["cross_entropy_nats"],
        "checkpoint_roundtrip": result["checkpoint_roundtrip"],
    }
    if args.class_nodes:
        summary["class_node_training"] = result["class_node_training"]
        summary["hpm_test_surface_cross_entropy_nats"] = result["hpm"]["test"]["surface_cross_entropy_nats"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
