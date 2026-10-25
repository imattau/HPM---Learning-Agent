"""Independent noisy role-grammar check for distributional class nodes.

This experiment does not modify or tune the composition testbed gates. It
creates overlapping verb roles, injects cross-domain follower noise, and
evaluates held-out verb/object pairs after the model chooses classes using its
normal training-only validation path.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

from hpm_ai_v2.training.language_prediction import HierarchicalNextUnitModel
from hpm_ai_v2.training.language_corpus import tokenize


SUBJECTS = ("alice", "bruno", "cora", "drew")
FOOD = ("apple", "pear", "peach", "plum")
SPORTS = ("ball", "disc", "bat", "frisbee")
VERB_OBJECTS = {
    "eats": FOOD,
    "likes": FOOD,
    "buys": FOOD,
    "kicks": SPORTS,
    "throws": SPORTS,
    # This verb belongs to both semantic roles; the hard clustering must
    # decide whether to group it with one role or leave it separate.
    "carries": FOOD + SPORTS,
}
HELDOUT_PAIRS = {
    ("eats", "apple"),
    ("likes", "pear"),
    ("buys", "peach"),
    ("kicks", "ball"),
    ("throws", "disc"),
    ("carries", "plum"),
    ("carries", "frisbee"),
}
NOISE = {
    "eats": ("ball",),
    "likes": ("disc",),
    "buys": ("bat",),
    "kicks": ("apple",),
    "throws": ("peach",),
    "carries": (),
}


def build_corpus() -> tuple[list[str], list[str], list[str]]:
    training: list[str] = []
    testing: list[str] = []
    test_groups: list[str] = []
    for subject in SUBJECTS:
        for verb, objects in VERB_OBJECTS.items():
            for obj in objects:
                if (verb, obj) in HELDOUT_PAIRS:
                    continue
                # Distinct tails provide repeated lexical evidence without
                # introducing duplicate document groups.
                training.extend((
                    f"the {subject} {verb} {obj} today",
                    f"the {subject} {verb} {obj} now",
                ))
            for obj in NOISE[verb]:
                training.append(f"the {subject} {verb} {obj} today")
    for subject in SUBJECTS:
        for verb, obj in sorted(HELDOUT_PAIRS):
            testing.append(f"the {subject} {verb} {obj} today")
            test_groups.append(f"subject-{subject}")
    random.Random(81).shuffle(training)
    return training, testing, test_groups


def run(*, bootstrap_samples: int = 2000) -> dict:
    training, testing, test_groups = build_corpus()
    started = time.monotonic()
    model = HierarchicalNextUnitModel(max_order=2, class_nodes=True).fit(training)
    evaluation = model.evaluate_class_generalization(
        testing, group_ids=test_groups, bootstrap_samples=bootstrap_samples,
    )
    pair_losses: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for text in testing:
        tokens = tokenize(text)
        for index, target in enumerate(tokens):
            if index == 0 or (tokens[index - 1], target) not in HELDOUT_PAIRS:
                continue
            raw_context = tuple(tokens[max(0, index - model.max_order + 1):index])
            context = tuple(model._map(token) for token in raw_context)
            mapped_target = model._map(target)
            plain = model._weighted_probability(
                context, mapped_target, apply_class_mix=False, apply_character_mix=False,
            )
            class_mixed = model._weighted_probability(
                context, mapped_target, apply_character_mix=False,
            )
            losses = pair_losses[(tokens[index - 1], target)]
            losses[0] += -math.log(max(plain, 1e-300))
            losses[1] += -math.log(max(class_mixed, 1e-300))
            losses[2] += 1
    unseen_pair_breakdown = [
        {
            "verb": verb,
            "object": obj,
            "events": int(values[2]),
            "plain_cross_entropy_nats": values[0] / values[2],
            "class_cross_entropy_nats": values[1] / values[2],
            "class_minus_plain_delta_nats_per_event": (values[1] - values[0]) / values[2],
        }
        for (verb, obj), values in sorted(pair_losses.items())
    ]
    active_class_ids = {
        class_id for class_id in model.class_labels.values()
        if model.class_lambdas.get(class_id, 0.0) > 0.0
    }
    class_sizes = {
        class_id: sum(label == class_id for label in model.class_labels.values())
        for class_id in sorted(active_class_ids)
    }
    final_class_members = {
        class_id: sorted(word for word, label in model.class_labels.items() if label == class_id)
        for class_id in sorted(active_class_ids)
    }
    return {
        "protocol": (
            "independent synthetic role grammar; no existing gates changed; "
            "training-only class validation; train/test disjoint at verb-object pair level"
        ),
        "grammar": {
            "subjects": list(SUBJECTS),
            "food_objects": list(FOOD),
            "sports_objects": list(SPORTS),
            "verbs": {verb: list(objects) for verb, objects in VERB_OBJECTS.items()},
            "overlapping_verb": "carries",
            "cross_domain_noise": {verb: list(objects) for verb, objects in NOISE.items()},
        },
        "training_documents": len(training),
        "test_documents": len(testing),
        "heldout_verb_object_pairs": [list(pair) for pair in sorted(HELDOUT_PAIRS)],
        "selected_tau": model.class_tau,
        "maximum_selected_class_lambda": model.class_lambda,
        "active_class_count": len(active_class_ids),
        "active_class_sizes": class_sizes,
        "final_class_members_after_refit": final_class_members,
        "class_quality": (model.class_selection or {}).get("class_quality", []),
        "class_quality_scope": "selection-stage clusters; final memberships are listed separately after full-data refit",
        "unseen_pair_comparison": evaluation,
        "unseen_pair_breakdown": unseen_pair_breakdown,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_groups": len(set(test_groups)),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "Synthetic grammar is diagnostic, not evidence of broad language generalization.",
            "Only four independent subject groups are available for bootstrap resampling.",
            "The overlap case tests one hard single-membership cluster, not soft membership.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("hpm_ai_v2/experiments/results/2026-10-25-class-nodes-fresh-grammar-v1/comparison.json"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    result = run(bootstrap_samples=args.bootstrap_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
