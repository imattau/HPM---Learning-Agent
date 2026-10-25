"""Class-node learning and prediction mixin for the HFN language pilot."""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict, OrderedDict
from typing import Iterable

import numpy as np

from hfn.hfn import HFN
from hpm_ai_v2.training.language_constants import BOUNDARY, UNK
from hpm_ai_v2.training.language_corpus import split_sentences, tokenize

CLASS_MAX_SEEN_REGRESSION_NATS = 0.05
CLASS_RESERVED_UNSEEN_MASS = 0.04
CLASS_MAX_MEMBERS = 64
CLASS_MIN_COHESION = 0.10
CLASS_MIN_VALIDATION_PAIRS = 1
CLASS_LAMBDA_GRID = (0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0)
CLASS_MIX_CACHE_MAX_ENTRIES = 32
CLASS_MIX_MASS_CACHE_MAX_ENTRIES = 4096


class LanguageClassMixin:
    @staticmethod
    def _project_follower_profile(profile: np.ndarray, features: list[str]) -> np.ndarray:
        """Deterministically embed follower profiles into the Forest's D=4 space.

        A signed feature hash makes projection stable for a token across training
        rounds and independent of the current vocabulary's ordering.
        """
        projected = np.zeros(4, dtype=float)
        for feature, value in zip(features, profile):
            if value == 0:
                continue
            digest = hashlib.sha256(f"hpm-language-profile-projection-v1:{feature}".encode()).digest()
            bucket = int.from_bytes(digest[:4], "big") % len(projected)
            sign = 1.0 if digest[4] & 1 else -1.0
            projected[bucket] += sign * float(value)
        return projected

    def _profiles_from_counts(self, raw_counts: dict) -> dict[str, np.ndarray]:
        """IDF-weighted follower profiles conditioned on the preceding role cue.

        When the forest has at least a two-token context, the profile feature
        is ``(word_before_context_word, next_word)``. This prevents a word
        used in distinct positions (for example a subject noun before a verb
        and an object noun before sentence end) from having its role signals
        collapsed into one unconditional follower vector.
        """
        followers: dict[str, Counter] = defaultdict(Counter)
        for context, targets in raw_counts.items():
            if len(context) == 2 and context[-1] not in (BOUNDARY, UNK):
                role_cue = context[-2]
                for target, count in targets.items():
                    followers[context[-1]][f"{role_cue}\0{target}"] += count
        # Preserve order-2 models' original behavior, where there is no
        # preceding role cue available to condition the follower profile.
        if not followers:
            for context, targets in raw_counts.items():
                if len(context) == 1 and context[0] not in (BOUNDARY, UNK):
                    followers[context[0]].update(targets)
        eligible = {word: counts for word, counts in followers.items()
                    if sum(counts.values()) >= self.class_min_observations}
        if len(eligible) > self.class_profile_limit:
            eligible = dict(sorted(eligible.items(), key=lambda item: (-sum(item[1].values()), item[0]))[
                            :self.class_profile_limit])
        if not eligible:
            self._current_profile_features = []
            return {}
        follower_totals = Counter()
        for counts in eligible.values():
            follower_totals.update(counts)
        target_vocab = [word for word, _ in sorted(follower_totals.items(),
                                                    key=lambda item: (-item[1], item[0]))[
                        :self.class_follower_limit]]
        self._current_profile_features = list(target_vocab)
        target_index = {word: index for index, word in enumerate(target_vocab)}
        global_counts = Counter()
        for counts in eligible.values():
            global_counts.update({word: count for word, count in counts.items()
                                  if word in target_index})
        global_total = sum(global_counts.values())
        global_prob = np.asarray([(global_counts.get(word, 0) + 1e-9) /
                                  (global_total + 1e-9 * len(target_vocab))
                                  for word in target_vocab])
        profiles = {}
        for word, counts in eligible.items():
            retained = {target: count for target, count in counts.items() if target in target_index}
            total = sum(retained.values())
            if not total:
                continue
            observed = np.zeros(len(target_vocab), dtype=float)
            for target, count in retained.items():
                observed[target_index[target]] = count / total
            pmi = np.maximum(0.0, np.log((observed + 1e-12) / global_prob))
            profiles[word] = pmi
        df = np.sum(np.asarray([profile > 0 for profile in profiles.values()]), axis=0)
        idf = np.log((len(profiles) + 1.0) / (df + 1.0))
        weighted = {word: profile * idf for word, profile in profiles.items()}
        # Single-follower words mostly encode sentence templates (for example
        # every synthetic verb followed by "a"), not a useful lexical class.
        return {word: profile for word, profile in weighted.items()
                if np.count_nonzero(profile > 1e-12) >= 2}

    @staticmethod
    def _average_link_clusters(profiles: dict[str, np.ndarray], tau: float | None) -> list[list[str]]:
        words = sorted(profiles)
        if tau is None or len(words) < 2:
            return [[word] for word in words]
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist, squareform
        matrix = np.asarray([profiles[word] for word in words], dtype=float)
        norms = np.linalg.norm(matrix, axis=1)
        zero = norms <= 1e-12
        matrix[~zero] /= norms[~zero, None]
        distances = pdist(matrix, metric="cosine")
        distances = np.nan_to_num(distances, nan=1.0, posinf=1.0, neginf=0.0)
        if np.any(zero):
            matrix_distance = squareform(distances)
            matrix_distance[zero, :] = 1.0
            matrix_distance[:, zero] = 1.0
            np.fill_diagonal(matrix_distance, 0.0)
            distances = squareform(matrix_distance)
        tree = linkage(distances, method="average")
        labels = fcluster(tree, t=float(tau), criterion="distance")
        grouped: dict[int, list[str]] = defaultdict(list)
        for word, label in zip(words, labels):
            grouped[int(label)].append(word)
        return sorted((sorted(group) for group in grouped.values()), key=lambda group: group[0])

    @staticmethod
    def _abstract_prediction_slot(context: tuple[str, ...], labels: dict[str, str]) -> tuple[str, ...]:
        """Replace only the immediately preceding unit by its learned class.

        Follower profiles are learned for P(next_unit | word), so their class
        membership applies to the context's final position. Applying the same
        global label to older context slots would treat a word as
        interchangeable in roles for which it was never evaluated.
        """
        if not context:
            return ()
        return (*context[:-1], labels.get(context[-1], context[-1]))

    def _validation_class_choice(self) -> tuple[float | None, float]:
        """Select coherent classes on document-heldout unseen word pairs only."""
        documents = self._class_validation_documents or []
        if not documents or self.max_order < 2:
            self.class_lambdas = {}
            self.class_selection = {
                "selection_reason": "no-class fallback: document-heldout validation unavailable",
                "validation_method": "source_grouped_document_holdout_unseen_adjacent_pairs",
                "held_pairs": 0,
                "candidate_diagnostics": [],
            }
            return None, 0.0

        # Validation counts are built from complete held-out documents; the
        # fit count tables and follower profiles below never contain these
        # documents. Hash-edge holdout from the training counts rewarded
        # memorized adjacency statistics and is intentionally removed.
        validation_counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        for text in documents:
            for sentence in split_sentences(text):
                sequence = tokenize(sentence) + [BOUNDARY]
                for index, target in enumerate(sequence):
                    context = tuple(sequence[max(0, index - self.max_order + 1):index])
                    validation_counts[context][target] += 1
        train_tables = self.raw_word_context_counts
        profiles = self._profiles_from_counts(train_tables)
        train_pairs = {
            (context[-1], target)
            for context, targets in train_tables.items() if len(context) == 1
            for target, count in targets.items() if count > 0
        }
        validation_events = []
        for context, targets in validation_counts.items():
            if not context:
                continue
            for target, count in targets.items():
                pair = (context[-1], target)
                validation_events.append((context, target, count, pair not in train_pairs))
        unseen_events = [event for event in validation_events
                         if event[3] and event[0][-1] in profiles]
        if not unseen_events:
            self.class_lambdas = {}
            self.class_selection = {
                "selection_reason": "no-class fallback: no held-out unseen pairs with learned profiles",
                "validation_method": "source_grouped_document_holdout_unseen_adjacent_pairs",
                "held_pairs": 0,
                "validation_events": len(validation_events),
                "candidate_diagnostics": [],
            }
            return None, 0.0

        # Train profiles are high-dimensional and are not retained after this
        # fit. Keep candidate cluster analysis cached by tau.
        thresholds = (0.7, 0.8, 0.9, 0.95, 0.98)
        candidate_diagnostics = []
        candidates = []
        no_class_pair_losses: dict[tuple[str, str], list[float]] = defaultdict(list)

        mapped_counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        for context, targets in train_tables.items():
            mapped_context = tuple(self._map(word) for word in context)
            for target, count in targets.items():
                mapped_counts[mapped_context][self._map(target)] += count

        def make_dist(tables, cache, *, observer_weighted=False):
            def dist(context, target):
                key = (context, target)
                if key in cache:
                    return cache[key]
                if not context:
                    unigram = mapped_counts.get((), Counter())
                    total = sum(unigram.values())
                    value = (unigram.get(target, 0) + self.alpha) / (
                        total + self.alpha * len(self.vocab)
                    )
                    if observer_weighted and self.observer is not None:
                        weight = self.observer.get_weight(self.word_level_nodes[0].id)
                        value = weight * value + (1.0 - weight) / len(self.vocab)
                else:
                    base = dist(context[1:], target)
                    local = tables.get(context, Counter())
                    total = sum(local.values())
                    if not total:
                        value = base
                    else:
                        mass = total / (total + self.alpha)
                        if observer_weighted and self.observer is not None:
                            order = min(len(context), self.max_order - 1)
                            mass *= self.observer.get_weight(self.word_level_nodes[order].id)
                        value = mass * local.get(target, 0) / total + (1.0 - mass) * base
                cache[key] = value
                return value
            return dist

        def mix_probability(context, target, class_context, lam, class_dist):
            """Use class mass only to refine the exact context's backoff mass.

            Pair targets already observed after the word keep their plain
            probability. The class expert redistributes only the remaining
            probability among unseen targets, so class induction cannot erase
            a learned edge. This is the same rule used at inference.
            """
            seen_targets = set()
            if context:
                for order in range(len(context), 0, -1):
                    local = mapped_counts.get(context[-order:], Counter())
                    if local:
                        seen_targets = set(local)
                        break
            plain = no_class_dist(context, target)
            if not seen_targets:
                return plain
            key = (context, class_context, tuple(sorted(seen_targets)))
            stats = mix_cache.get(key)
            if stats is None:
                plain_seen_mass = sum(no_class_dist(context, word) for word in seen_targets)
                class_seen_mass = sum(class_dist(class_context, word) for word in seen_targets)
                stats = (plain_seen_mass, max(0.0, 1.0 - plain_seen_mass),
                         max(0.0, 1.0 - class_seen_mass))
                mix_cache[key] = stats
            plain_seen_mass, plain_unseen_mass, class_unseen_mass = stats
            if class_unseen_mass <= 1e-12:
                return plain
            reserved_unseen_mass = max(
                plain_unseen_mass, CLASS_RESERVED_UNSEEN_MASS * lam,
            )
            if target in seen_targets:
                if reserved_unseen_mass <= plain_unseen_mass or plain_seen_mass <= 0.0:
                    return plain
                return plain * (1.0 - reserved_unseen_mass) / plain_seen_mass
            base_component = (plain / plain_unseen_mass if plain_unseen_mass > 1e-12
                              else class_dist(class_context, target) / class_unseen_mass)
            class_component = class_dist(class_context, target) / class_unseen_mass
            return reserved_unseen_mass * (
                (1.0 - lam) * base_component + lam * class_component
            )

        # The plain expert must match actual inference, including Observer
        # reliability weights. The class expert remains an interpolated
        # pooled-count distribution, mixed by its own Observer weight.
        no_class_dist = make_dist(mapped_counts, {}, observer_weighted=True)
        for context, target, count, _unseen in unseen_events:
            mapped_context = tuple(self._map(word) for word in context)
            mapped_target = self._map(target)
            pair = (context[-1], target)
            no_class_pair_losses[pair].extend(
                [-math.log(max(no_class_dist(mapped_context, mapped_target), 1e-12))] * count
            )
        no_class_pair_means = np.asarray([
            float(np.mean(losses)) for losses in no_class_pair_losses.values()
        ])
        no_class_se = (float(np.std(no_class_pair_means, ddof=1) / math.sqrt(len(no_class_pair_means)))
                       if len(no_class_pair_means) > 1 else float("inf"))

        for tau in thresholds:
            raw_clusters = self._average_link_clusters(profiles, tau)
            kept_clusters = []
            for members in raw_clusters:
                if len(members) < 2 or len(members) > CLASS_MAX_MEMBERS:
                    continue
                matrix = np.asarray([profiles[word] for word in members], dtype=float)
                norms = np.linalg.norm(matrix, axis=1)
                matrix = matrix / np.maximum(norms[:, None], 1e-12)
                centroid = matrix.mean(axis=0)
                centroid_norm = float(np.linalg.norm(centroid))
                cohesion = (float(np.mean(matrix @ centroid / max(centroid_norm, 1e-12)))
                            if centroid_norm else 0.0)
                if cohesion >= CLASS_MIN_COHESION:
                    kept_clusters.append((members, cohesion))

            labels = {}
            class_members = {}
            class_cohesion = {}
            for index, (members, cohesion) in enumerate(kept_clusters):
                class_id = f"language_class_{index:06d}"
                labels.update({word: class_id for word in members})
                class_members[class_id] = members
                class_cohesion[class_id] = cohesion

            pooled: dict[tuple[str, ...], Counter] = defaultdict(Counter)
            for context, targets in train_tables.items():
                class_context = self._abstract_prediction_slot(context, labels)
                if context and class_context != context:
                    for target, count in targets.items():
                        pooled[class_context][self._map(target)] += count
            class_dist = make_dist(pooled, {})
            mix_cache = {}

            class_lambdas = {}
            class_diagnostics = []
            for class_id, members in class_members.items():
                events = [event for event in unseen_events if labels.get(event[0][-1]) == class_id]
                pairs = sorted({(event[0][-1], event[1]) for event in events})
                if len(pairs) < CLASS_MIN_VALIDATION_PAIRS:
                    # An explicit zero distinguishes a rejected/unvalidated
                    # class from a legacy class whose weight should inherit
                    # the global compatibility default.
                    class_lambdas[class_id] = 0.0
                    class_diagnostics.append({
                        "class_id": class_id, "members": len(members),
                        "example_members": members[:12], "cohesion": class_cohesion[class_id],
                        "validation_unseen_pairs": len(pairs), "selected_weight": 0.0,
                        "selection_reason": "insufficient held-out unseen-pair evidence",
                    })
                    continue
                per_lambda: dict[float, dict[tuple[str, str], list[float]]] = {
                    lam: defaultdict(list) for lam in (0.0, *CLASS_LAMBDA_GRID)
                }
                per_seen_lambda: dict[float, dict[tuple[str, str], list[float]]] = {
                    lam: defaultdict(list) for lam in (0.0, *CLASS_LAMBDA_GRID)
                }
                for raw_context, raw_target, count, _ in events:
                    context = tuple(self._map(word) for word in raw_context)
                    target = self._map(raw_target)
                    plain = no_class_dist(context, target)
                    abstract = self._abstract_prediction_slot(raw_context, labels)
                    class_probability = class_dist(abstract, target)
                    pair = (raw_context[-1], raw_target)
                    for lam, pair_losses in per_lambda.items():
                        mixture = mix_probability(context, target, abstract, lam, class_dist)
                        pair_losses[pair].extend(
                            [-math.log(max(mixture, 1e-12))] * count
                        )
                seen_events = [
                    event for event in validation_events
                    if not event[3] and event[0]
                    and labels.get(event[0][-1]) == class_id
                ]
                for raw_context, raw_target, count, _ in seen_events:
                    context = tuple(self._map(word) for word in raw_context)
                    target = self._map(raw_target)
                    plain = no_class_dist(context, target)
                    abstract = self._abstract_prediction_slot(raw_context, labels)
                    class_probability = class_dist(abstract, target)
                    pair = (raw_context[-1], raw_target)
                    for lam, pair_losses in per_seen_lambda.items():
                        mixture = mix_probability(context, target, abstract, lam, class_dist)
                        pair_losses[pair].extend(
                            [-math.log(max(mixture, 1e-12))] * count
                        )
                means = {
                    lam: np.asarray([float(np.mean(losses)) for losses in pair_losses.values()])
                    for lam, pair_losses in per_lambda.items()
                }
                losses = {lam: float(values.mean()) for lam, values in means.items()}
                seen_means = {
                    lam: np.asarray([float(np.mean(values)) for values in pair_losses.values()])
                    for lam, pair_losses in per_seen_lambda.items()
                }
                plain_seen_loss = float(seen_means[0.0].mean()) if len(seen_means[0.0]) else 0.0
                seen_deltas_by_lambda = {
                    lam: (float(values.mean()) - plain_seen_loss if len(values) else 0.0)
                    for lam, values in seen_means.items()
                }
                eligible_lambdas = [
                    lam for lam in losses
                    if seen_deltas_by_lambda[lam] <= CLASS_MAX_SEEN_REGRESSION_NATS
                ]
                best_lam = min(eligible_lambdas, key=losses.get) if eligible_lambdas else 0.0
                best_values = means[best_lam]
                best_se = (float(np.std(best_values, ddof=1) / math.sqrt(len(best_values)))
                           if len(best_values) > 1 else 0.0)
                if losses[0.0] - losses[best_lam] <= best_se:
                    best_lam = 0.0
                else:
                    # One-standard-error selection favors the lightest mix
                    # whose held-out risk is statistically indistinguishable
                    # from the minimum.
                    cutoff = losses[best_lam] + best_se
                    best_lam = min(
                        lam for lam in eligible_lambdas if losses[lam] <= cutoff
                    )
                if len(pairs) == 1:
                    # A single held-out edge cannot estimate class-level
                    # variance. Keep its influence deliberately small; the
                    # aggregate edge-level gate below still has to establish
                    # a reliable gain for the complete clustering.
                    best_lam = min(best_lam, 0.5)
                class_lambdas[class_id] = best_lam
                class_diagnostics.append({
                    "class_id": class_id, "members": len(members),
                    "example_members": members[:12], "cohesion": class_cohesion[class_id],
                    "validation_unseen_pairs": len(pairs),
                    "validation_events": sum(event[2] for event in events),
                    "selected_weight": best_lam,
                    "validation_seen_pairs": len(seen_means[0.0]),
                    "validation_seen_pair_loss_delta_nats": seen_deltas_by_lambda[best_lam],
                    "validation_gain_nats": losses[0.0] - losses[best_lam],
                    "paired_edge_standard_error_nats": best_se if len(pairs) > 1 else None,
                    "selection_reason": "held-out unseen-pair improvement" if best_lam else
                        "no reliable held-out unseen-pair improvement",
                })

            # A class can improve its withheld combinations while blurring
            # already learned pairs. Let each class earn its own seen-pair
            # safety decision before judging the remaining classes globally.
            seen_events_by_class: dict[str, list[tuple[float, int]]] = defaultdict(list)
            for raw_context, raw_target, count, is_unseen in validation_events:
                if is_unseen or not raw_context:
                    continue
                class_id = labels.get(raw_context[-1])
                lam = class_lambdas.get(class_id, 0.0)
                if not lam:
                    continue
                context = tuple(self._map(word) for word in raw_context)
                target = self._map(raw_target)
                plain = no_class_dist(context, target)
                abstract = self._abstract_prediction_slot(raw_context, labels)
                class_probability = class_dist(abstract, target)
                mixture = mix_probability(context, target, abstract, lam, class_dist)
                loss_delta = math.log(max(plain, 1e-12) / max(mixture, 1e-12))
                seen_events_by_class[class_id].append((loss_delta, count))
            for diagnostic in class_diagnostics:
                class_id = diagnostic["class_id"]
                seen_rows = seen_events_by_class.get(class_id, [])
                if not seen_rows:
                    diagnostic["validation_seen_pairs"] = 0
                    continue
                seen_delta = sum(delta * count for delta, count in seen_rows) / sum(
                    count for _delta, count in seen_rows
                )
                diagnostic["validation_seen_pairs"] = len(seen_rows)
                diagnostic["validation_seen_pair_loss_delta_nats"] = seen_delta
                if (class_lambdas.get(class_id, 0.0) > 0.0
                        and seen_delta > CLASS_MAX_SEEN_REGRESSION_NATS):
                    class_lambdas[class_id] = 0.0
                    diagnostic["selected_weight"] = 0.0
                    diagnostic["selection_reason"] = "class harms held-out seen pairs"

            # Estimate aggregate unseen-pair gain and seen-pair harm for the
            # candidate partition. A class is active only with its own
            # evidence; global tau is chosen on pooled held-out unseen pairs.
            unseen_deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
            seen_deltas: list[float] = []
            for raw_context, raw_target, count, is_unseen in validation_events:
                if not raw_context:
                    continue
                class_id = labels.get(raw_context[-1])
                lam = class_lambdas.get(class_id, 0.0)
                if not lam:
                    continue
                context = tuple(self._map(word) for word in raw_context)
                target = self._map(raw_target)
                plain = no_class_dist(context, target)
                abstract = self._abstract_prediction_slot(raw_context, labels)
                mixture = mix_probability(context, target, abstract, lam, class_dist)
                delta = math.log(max(plain, 1e-12) / max(mixture, 1e-12))
                if is_unseen:
                    # Improvement is positive when the class mixture assigns
                    # greater likelihood to the held-out target.
                    unseen_deltas[(raw_context[-1], raw_target)].extend([-delta] * count)
                else:
                    # Loss delta is mixture minus plain, so positive values
                    # indicate seen-pair harm and trigger the safety bound.
                    seen_deltas.extend([delta] * count)
            edge_gains = np.asarray([float(np.mean(values)) for values in unseen_deltas.values()])
            gain = float(edge_gains.mean()) if len(edge_gains) else 0.0
            gain_se = (float(np.std(edge_gains, ddof=1) / math.sqrt(len(edge_gains)))
                       if len(edge_gains) > 1 else float("inf"))
            seen_delta = float(np.mean(seen_deltas)) if seen_deltas else 0.0
            rejected_for_seen_harm = seen_delta > CLASS_MAX_SEEN_REGRESSION_NATS
            if rejected_for_seen_harm or gain <= gain_se:
                class_lambdas = {}
            candidates.append({"tau": tau, "gain": gain, "gain_se": gain_se,
                               "seen_pair_loss_delta_nats": seen_delta,
                               "class_lambdas": class_lambdas,
                               "class_diagnostics": class_diagnostics,
                               "class_count": sum(value > 0.0
                                                  for value in class_lambdas.values())})
            candidate_diagnostics.append({
                "tau": tau, "unseen_pair_gain_nats": gain,
                "paired_edge_standard_error_nats": gain_se,
                "seen_pair_loss_delta_nats": seen_delta,
                "active_class_count": sum(value > 0.0 for value in class_lambdas.values()),
                "rejected_for_seen_pair_harm": rejected_for_seen_harm,
                "rejection_reason": (
                    "seen-pair loss exceeded safety bound" if rejected_for_seen_harm else
                    "unseen-pair gain did not exceed its paired standard error"
                    if gain <= gain_se else None
                ),
                "class_quality": class_diagnostics,
            })
            if (rejected_for_seen_harm or gain <= gain_se or not class_lambdas):
                candidates.pop()

        if candidates:
            # Each active class has already earned a per-class held-out gain,
            # and the complete candidate has passed the paired-edge and
            # seen-pair gates. Prefer the validated partition that retains the
            # most independently useful class nodes; aggregate gain breaks
            # ties between equally expressive candidates.
            selected = max(candidates, key=lambda row: (
                row["class_count"], row["gain"], -float(row["tau"]),
            ))
        else:
            selected = {"tau": None, "gain": 0.0, "gain_se": no_class_se,
                        "seen_pair_loss_delta_nats": 0.0,
                        "class_lambdas": {}, "class_diagnostics": [], "class_count": 0}
        if selected["gain"] <= selected["gain_se"] or not selected["class_lambdas"]:
            self.class_lambdas = {}
            self.class_selection = {
                "validation_method": "source_grouped_document_holdout_unseen_adjacent_pairs",
                "selection_reason": "no-class fallback: no reliable held-out unseen-pair gain",
                "held_pairs": len({(event[0][-1], event[1]) for event in unseen_events}),
                "validation_events": len(validation_events),
                "unseen_validation_events": sum(event[2] for event in unseen_events),
                "unseen_pair_gain_nats": selected["gain"],
                "paired_edge_standard_error_nats": selected["gain_se"],
                "candidate_diagnostics": candidate_diagnostics,
            }
            return None, 0.0
        self.class_lambdas = dict(selected["class_lambdas"])
        self.class_lambda = max(self.class_lambdas.values(), default=0.0)
        self.class_selection = {
            "validation_method": "source_grouped_document_holdout_unseen_adjacent_pairs",
            "selection_reason": "per-class held-out unseen-pair reinforcement",
            "held_pairs": len({(event[0][-1], event[1]) for event in unseen_events}),
            "validation_events": len(validation_events),
            "unseen_validation_events": sum(event[2] for event in unseen_events),
            "selected_tau": selected["tau"],
            "selected_lambda_by_class": self.class_lambdas,
            "selected_validation_gain_nats": selected["gain"],
            "paired_edge_standard_error_nats": selected["gain_se"],
            "selected_seen_pair_loss_delta_nats": selected["seen_pair_loss_delta_nats"],
            "class_quality": selected["class_diagnostics"],
            "candidate_diagnostics": candidate_diagnostics,
        }
        return selected["tau"], max(self.class_lambdas.values(), default=0.0)

    def _fit_class_nodes_from_counts(self, *, refit: bool) -> None:
        """Induce/freeze classes, refresh compact projections, and pool counts."""
        profiles = self._profiles_from_counts(self.raw_word_context_counts)
        if refit:
            self.class_tau, self.class_lambda = self._validation_class_choice()
            profiles = self._profiles_from_counts(self.raw_word_context_counts)
            clusters = self._average_link_clusters(profiles, self.class_tau)
            self.class_labels = {}
            valid_clusters = []
            for members in clusters:
                if len(members) < 2 or len(members) > CLASS_MAX_MEMBERS:
                    continue
                matrix = np.asarray([profiles[word] for word in members], dtype=float)
                norms = np.linalg.norm(matrix, axis=1)
                matrix = matrix / np.maximum(norms[:, None], 1e-12)
                centroid = matrix.mean(axis=0)
                centroid_norm = float(np.linalg.norm(centroid))
                cohesion = (float(np.mean(matrix @ centroid / max(centroid_norm, 1e-12)))
                            if centroid_norm else 0.0)
                if cohesion >= CLASS_MIN_COHESION:
                    valid_clusters.append(members)
            for index, members in enumerate(valid_clusters):
                class_id = f"language_class_{index:06d}"
                if class_id not in self.class_lambdas and self.class_lambda > 0:
                    # Compatibility for callers that inject a class-choice
                    # policy. Normal training always records per-class values.
                    self.class_lambdas[class_id] = self.class_lambda
                if self.class_lambdas.get(class_id, 0.0) <= 0:
                    continue
                self.class_labels.update({word: class_id for word in members})
            self._class_membership_frozen = True
        # The full follower vectors are transient clustering inputs. Persisting
        # one mean and variance vector per class duplicates thousands of
        # dimensions even though HFN nodes use only the stable four-dimensional
        # projection. Recompute that compact summary after load and partial_fit.
        self.class_profile_features = list(self._current_profile_features)
        self.class_profiles = {}
        self.class_profile_variances = {}
        for class_id in sorted(set(self.class_labels.values())):
            members = sorted(word for word, label in self.class_labels.items() if label == class_id)
            projected = np.asarray([
                self._project_follower_profile(
                    profiles.get(word, np.zeros(len(self.class_profile_features))),
                    self.class_profile_features,
                ) for word in members
            ])
            self.class_profiles[class_id] = projected.mean(axis=0)
            self.class_profile_variances[class_id] = projected.var(axis=0) + 0.05 ** 2
        self.class_context_counts = defaultdict(Counter)
        if not self.class_labels or not any(self.class_lambdas.values()):
            return
        for raw_context, targets in self.raw_word_context_counts.items():
            if not raw_context:
                continue
            class_context = self._abstract_prediction_slot(raw_context, self.class_labels)
            if class_context == raw_context:
                continue
            for target, count in targets.items():
                self.class_context_counts[class_context][self._map(target)] += count

    def _refit_selected_classes_on_full_training_counts(self) -> dict:
        """Refit selected class profiles on all initial-fit documents.

        Selection and lambda tuning use document-heldout pairs. After that
        choice is frozen, the reserved fit documents return to the training
        counts and class profiles are recomputed on the complete training
        partition, as in ordinary train/validation refit. Active weights are
        transferred by member overlap; classes with no earned weight stay at
        zero. Subsequent ``partial_fit`` calls still preserve membership.
        """
        prior_labels = dict(self.class_labels)
        prior_weights = dict(self.class_lambdas)
        profiles = self._profiles_from_counts(self.raw_word_context_counts)
        clusters = self._average_link_clusters(profiles, self.class_tau)
        self.class_labels = {}
        self.class_lambdas = {}
        valid_clusters = []
        for members in clusters:
            if len(members) < 2 or len(members) > CLASS_MAX_MEMBERS:
                continue
            matrix = np.asarray([profiles[word] for word in members], dtype=float)
            matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            centroid = matrix.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            cohesion = float(np.mean(matrix @ centroid / max(norm, 1e-12))) if norm else 0.0
            if cohesion >= CLASS_MIN_COHESION:
                valid_clusters.append(members)
        for index, members in enumerate(valid_clusters):
            class_id = f"language_class_{index:06d}"
            inherited = [prior_weights.get(prior_labels[word], 0.0)
                         for word in members if word in prior_labels]
            weight = sum(inherited) / len(members) if inherited else 0.0
            self.class_lambdas[class_id] = weight
            if weight > 0.0:
                self.class_labels.update({word: class_id for word in members})
        self.class_lambda = max(self.class_lambdas.values(), default=0.0)
        self._class_membership_frozen = True
        self.class_profile_features = list(self._current_profile_features)
        self.class_profiles = {}
        self.class_profile_variances = {}
        for class_id in sorted(set(self.class_labels.values())):
            members = sorted(word for word, label in self.class_labels.items()
                             if label == class_id)
            projected = np.asarray([
                self._project_follower_profile(
                    profiles.get(word, np.zeros(len(self.class_profile_features))),
                    self.class_profile_features,
                ) for word in members
            ])
            self.class_profiles[class_id] = projected.mean(axis=0)
            self.class_profile_variances[class_id] = projected.var(axis=0) + 0.05 ** 2
        self.class_context_counts = defaultdict(Counter)
        for raw_context, targets in self.raw_word_context_counts.items():
            if not raw_context:
                continue
            class_context = self._abstract_prediction_slot(raw_context, self.class_labels)
            if class_context == raw_context:
                continue
            for target, count in targets.items():
                self.class_context_counts[class_context][self._map(target)] += count
        return {
            "method": "full_training_profile_refit_after_heldout_selection",
            "classes": len(valid_clusters),
            "active_classes": len(set(self.class_labels.values())),
            "members": len(self.class_labels),
            "selected_tau": self.class_tau,
        }

    def _reinforce_class_observer_weights(self, documents: Iterable[str]) -> dict:
        """Prequentially reinforce classes on heldout unseen adjacent pairs.

        Each target is scored before its class weight changes. The class expert
        receives positive reinforcement when its likelihood beats the exact
        word-context expert and a penalty otherwise. Validation documents are
        never added to the count tables until this pass has completed.
        """
        if not self.class_node_registry or self.observer is None:
            return {"events": 0, "updated_classes": 0, "weights": {}}
        train_pairs = {
            (context[-1], target)
            for context, targets in self.raw_word_context_counts.items()
            if len(context) == 1 for target, count in targets.items() if count > 0
        }
        before = {class_id: self.observer.get_weight(class_id)
                  for class_id in self.class_node_registry}
        events = 0
        for text in documents:
            for sentence in split_sentences(text):
                sequence = tokenize(sentence) + [BOUNDARY]
                for index, raw_target in enumerate(sequence):
                    raw_context = tuple(sequence[max(0, index - self.max_order + 1):index])
                    if not raw_context:
                        continue
                    class_id = self.class_labels.get(raw_context[-1])
                    if class_id not in self.class_node_registry:
                        continue
                    if (raw_context[-1], raw_target) in train_pairs:
                        continue
                    mapped_context = tuple(self._map(word) for word in raw_context)
                    target = self._map(raw_target)
                    abstract = self._abstract_prediction_slot(raw_context, self.class_labels)
                    class_context = tuple(
                        token if token == class_id else self._map(token) for token in abstract
                    )
                    plain_probability = self._weighted_probability(
                        mapped_context, target, apply_class_mix=False
                    )
                    class_probability = self._class_probability_for_context(class_context, target)
                    if class_probability > plain_probability:
                        self.observer.boost_id(class_id, gain=0.1)
                    else:
                        self.observer.penalize_id(class_id, penalty=0.02)
                    events += 1
        weights = {class_id: self.observer.get_weight(class_id)
                   for class_id in self.class_node_registry}
        dropped = []
        for class_id, weight in list(weights.items()):
            if weight < 0.005:
                self.observer.set_weight(class_id, 0.0)
                weights[class_id] = 0.0
                dropped.append(class_id)
            self.class_lambdas[class_id] = weights[class_id]
        self.class_lambda = max(self.class_lambdas.values(), default=0.0)
        return {
            "events": events,
            "updated_classes": sum(abs(weights[key] - before[key]) > 1e-12 for key in weights),
            "weights_before": before,
            "weights": weights,
            "dropped_classes": dropped,
        }

    def _class_mixed_probability(
        self, context: tuple[str, ...], target: str, plain_probability: float,
    ) -> float | None:
        """Score one class-mixed target without materializing a vocabulary map.

        Class mixing reserves probability mass for unseen continuations while
        preserving the base predictor on observed continuations. Only the mass
        of the observed set under the plain and class experts is needed for a
        single target. Computing those sums scales with observed followers,
        rather than vocabulary size, and avoids allocating a dense class map
        for every token in a corpus evaluation.
        """
        class_context = self._abstract_prediction_slot(context, self.class_labels)
        if class_context == context or class_context not in self.class_context_nodes:
            return None
        class_id = class_context[-1]
        weight = (self.observer.get_weight(class_id)
                  if self.observer is not None and class_id in self.class_node_registry
                  else self.class_lambdas.get(class_id, self.class_lambda))
        if weight <= 0.0:
            return None

        seen_targets = set()
        for order in range(len(context), 0, -1):
            local = self.counts.get(context[-order:], Counter())
            if local:
                seen_targets = set(local)
                break
        if not seen_targets:
            return None

        order_weights = (tuple(self.observer.get_weight(node.id)
                               for _, node in sorted(self.word_level_nodes.items()))
                         if self.observer is not None else ())
        key = (context, class_context, weight, order_weights, self.alpha)
        cached = self._class_mix_mass_cache.get(key)
        if cached is None:
            plain_seen_mass = sum(
                self._weighted_probability(
                    context, word, apply_class_mix=False, apply_character_mix=False,
                ) for word in seen_targets
            )
            class_seen_mass = sum(
                self._class_probability_for_context(class_context, word)
                for word in seen_targets
            )
            cached = (plain_seen_mass, max(0.0, 1.0 - plain_seen_mass),
                      max(0.0, 1.0 - class_seen_mass))
            self._class_mix_mass_cache[key] = cached
            while len(self._class_mix_mass_cache) > CLASS_MIX_MASS_CACHE_MAX_ENTRIES:
                self._class_mix_mass_cache.popitem(last=False)
        else:
            self._class_mix_mass_cache.move_to_end(key)

        plain_seen_mass, plain_unseen_mass, class_unseen_mass = cached
        if class_unseen_mass <= 1e-12:
            return plain_probability
        reserved_unseen_mass = max(
            plain_unseen_mass, CLASS_RESERVED_UNSEEN_MASS * weight,
        )
        if target in seen_targets:
            if reserved_unseen_mass > plain_unseen_mass and plain_seen_mass > 0.0:
                return plain_probability * (1.0 - reserved_unseen_mass) / plain_seen_mass
            return plain_probability
        if reserved_unseen_mass > plain_unseen_mass and plain_seen_mass <= 0.0:
            return plain_probability
        class_probability = self._class_probability_for_context(class_context, target)
        if plain_unseen_mass > 1e-12:
            plain_component = plain_probability / plain_unseen_mass
        else:
            plain_component = class_probability / class_unseen_mass
        class_component = class_probability / class_unseen_mass
        return reserved_unseen_mass * (
            (1.0 - weight) * plain_component + weight * class_component
        )

    def _class_mixed_distribution(
        self, context: tuple[str, ...], *, plain_distribution: dict[str, float] | None = None,
    ) -> dict[str, float] | None:
        """Apply a class prior only over probability mass reserved for unseen edges.

        The exact context retains its learned target probabilities. The class
        node redistributes only the remaining mass across unseen targets.
        This protects observed word pairs while allowing the hierarchy to
        rank plausible but unobserved followers.
        """
        class_context = self._abstract_prediction_slot(context, self.class_labels)
        if class_context == context or class_context not in self.class_context_nodes:
            return None
        class_id = class_context[-1]
        weight = (self.observer.get_weight(class_id)
                  if self.observer is not None and class_id in self.class_node_registry
                  else self.class_lambdas.get(class_id, self.class_lambda))
        if weight <= 0.0:
            return None
        if self.observer is not None and plain_distribution is None:
            order_weights = tuple(
                self.observer.get_weight(node.id)
                for _, node in sorted(self.word_level_nodes.items())
            )
        else:
            order_weights = ()
        mode = "count" if plain_distribution is not None else "observer"
        cache_key = (mode, context, class_context, weight, order_weights, self.alpha)
        cache = getattr(self, "_class_mix_distribution_cache", None)
        if cache is None:
            cache = self._class_mix_distribution_cache = OrderedDict()
        cached = cache.get(cache_key)
        if cached is not None:
            cache.move_to_end(cache_key)
            return cached
        if plain_distribution is None:
            plain_distribution = {
                word: self._weighted_probability(
                    context, word, apply_class_mix=False, apply_character_mix=False,
                ) for word in self.vocab
            }
        seen_targets = set()
        if context:
            for order in range(len(context), 0, -1):
                local = self.counts.get(context[-order:], Counter())
                if local:
                    seen_targets = set(local)
                    break
        if not seen_targets:
            return None
        seen_mass = sum(plain_distribution.get(word, 0.0) for word in seen_targets)
        plain_unseen_mass = max(0.0, 1.0 - seen_mass)
        class_distribution = self._class_distribution_for_context(class_context)
        class_unseen_mass = max(
            0.0, 1.0 - sum(class_distribution.get(word, 0.0) for word in seen_targets)
        )
        if class_unseen_mass <= 1e-12:
            return plain_distribution
        reserved_unseen_mass = max(
            plain_unseen_mass, CLASS_RESERVED_UNSEEN_MASS * weight,
        )
        if reserved_unseen_mass > plain_unseen_mass and seen_mass <= 0.0:
            return plain_distribution
        mixed = {}
        for word in self.vocab:
            if word in seen_targets:
                mixed[word] = (
                    plain_distribution.get(word, 0.0)
                    * (1.0 - reserved_unseen_mass) / seen_mass
                    if reserved_unseen_mass > plain_unseen_mass else
                    plain_distribution.get(word, 0.0)
                )
            else:
                base_component = (
                    plain_distribution.get(word, 0.0) / plain_unseen_mass
                    if plain_unseen_mass > 1e-12 else
                    class_distribution.get(word, 0.0) / class_unseen_mass
                )
                class_component = class_distribution.get(word, 0.0) / class_unseen_mass
                mixed[word] = reserved_unseen_mass * (
                    (1.0 - weight) * base_component + weight * class_component
                )
        cache[cache_key] = mixed
        while len(cache) > CLASS_MIX_CACHE_MAX_ENTRIES:
            cache.popitem(last=False)
        return mixed

    def _class_distribution_for_context(self, context: tuple[str, ...]) -> dict[str, float]:
        """Smoothed distribution backed off through pooled class contexts."""
        if not context:
            counts = self.counts.get((), Counter())
            denom = sum(counts.values()) + self.alpha * len(self.vocab)
            return {word: (counts.get(word, 0) + self.alpha) / denom for word in self.vocab}
        base = self._class_distribution_for_context(context[1:])
        node = self.class_context_nodes.get(context)
        counts = (node.metadata.get("target_counts", {})
                  if node is not None else self.class_context_counts.get(context, Counter()))
        total = sum(counts.values())
        if not total:
            return base
        local_mass = total / (total + self.alpha)
        return {word: local_mass * counts.get(word, 0) / total + (1.0 - local_mass) * base[word]
                for word in self.vocab}

    def _class_probability_for_context(self, context: tuple[str, ...], target: str) -> float:
        """Return one pooled-class target probability without building a dense vocabulary map."""
        if not context:
            counts = self.counts.get((), Counter())
            denom = sum(counts.values()) + self.alpha * len(self.vocab)
            return (counts.get(target, 0) + self.alpha) / denom
        base = self._class_probability_for_context(context[1:], target)
        node = self.class_context_nodes.get(context)
        counts = (node.metadata.get("target_counts", {})
                  if node is not None else self.class_context_counts.get(context, Counter()))
        total = sum(counts.values())
        if not total:
            return base
        local_mass = total / (total + self.alpha)
        return (local_mass * counts.get(target, 0) / total
                + (1.0 - local_mass) * base)

    def _count_probability_for_context(self, context: tuple[str, ...], target: str) -> float:
        """Raw interpolated-count probability used by the exact flat fallback."""
        if not context:
            counts = self.counts.get((), Counter())
            denominator = sum(counts.values()) + self.alpha * len(self.vocab)
            return (counts.get(target, 0) + self.alpha) / denominator
        base = self._count_probability_for_context(context[1:], target)
        counts = self.counts.get(context, Counter())
        total = sum(counts.values())
        if not total:
            return base
        mass = total / (total + self.alpha)
        return mass * counts.get(target, 0) / total + (1.0 - mass) * base

    def evaluate_class_generalization(
        self, documents: Iterable[str], *, group_ids: Iterable[str] | None = None,
        bootstrap_samples: int = 1000,
    ) -> dict:
        """Compare class-on/off loss on test pairs unseen in HFN training.

        Both predictions use this checkpoint's identical word Observer state;
        only class mixing changes. Confidence intervals resample independent
        source groups, not tokens. A pair is unseen when its adjacent
        ``(previous_word, target_word)`` combination is absent from training.
        """
        documents = list(documents)
        if group_ids is None:
            groups = [str(index) for index in range(len(documents))]
        else:
            groups = list(group_ids)
            if len(groups) != len(documents):
                raise ValueError("group_ids must contain one identifier per document")
        if bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be nonnegative")
        trained_pairs = {
            (context[-1], target)
            for context, targets in self.raw_word_context_counts.items()
            if len(context) == 1 for target, count in targets.items() if count > 0
        }
        # group -> split -> [plain NLL, class NLL, events, distinct pairs]
        grouped: dict[str, dict[str, list]] = defaultdict(lambda: {
            "unseen": [0.0, 0.0, 0, set()],
            "seen": [0.0, 0.0, 0, set()],
        })
        totals = {"unseen": [0.0, 0.0, 0, set()], "seen": [0.0, 0.0, 0, set()]}
        for text, group in zip(documents, groups):
            for sentence in split_sentences(text):
                sequence = tokenize(sentence) + [BOUNDARY]
                for index, target in enumerate(sequence):
                    raw_context = tuple(sequence[max(0, index - self.max_order + 1):index])
                    if not raw_context:
                        continue
                    pair = (raw_context[-1], target)
                    split_name = "unseen" if pair not in trained_pairs else "seen"
                    context = tuple(self._map(word) for word in raw_context)
                    mapped_target = self._map(target)
                    plain_probability = self._weighted_probability(
                        context, mapped_target, apply_class_mix=False,
                        apply_character_mix=False,
                    )
                    class_probability = self._weighted_probability(
                        context, mapped_target, apply_character_mix=False,
                    )
                    plain_nll = -math.log(max(plain_probability, 1e-300))
                    class_nll = -math.log(max(class_probability, 1e-300))
                    for bucket in (grouped[group][split_name], totals[split_name]):
                        bucket[0] += plain_nll
                        bucket[1] += class_nll
                        bucket[2] += 1
                        bucket[3].add(pair)

        def summarize(split_name: str) -> dict:
            total_plain, total_class, count, pairs = totals[split_name]
            group_deltas = []
            for values in grouped.values():
                p_nll, c_nll, events, _pairs = values[split_name]
                if events:
                    group_deltas.append((c_nll - p_nll) / events)
            interval = None
            mean_delta = float(np.mean(group_deltas)) if group_deltas else None
            if group_deltas and bootstrap_samples:
                rng = np.random.default_rng(20261025)
                samples = [float(np.mean(rng.choice(group_deltas, size=len(group_deltas), replace=True)))
                           for _ in range(bootstrap_samples)]
                interval = [float(np.quantile(samples, 0.025)),
                            float(np.quantile(samples, 0.975))]
            return {
                "events": count,
                "distinct_adjacent_pairs": len(pairs),
                **({"distinct_unseen_adjacent_pairs": len(pairs)}
                   if split_name == "unseen" else {}),
                "plain_cross_entropy_nats": total_plain / count if count else None,
                "class_cross_entropy_nats": total_class / count if count else None,
                "paired_class_minus_plain_delta_nats_per_event": mean_delta,
                "source_group_bootstrap_95ci": interval,
                "source_groups": len(group_deltas),
            }

        return {
            "comparison": "same word Observer weights; class mixing is the only predictor difference",
            "pair_definition": "adjacent raw-token pair absent from training bigram counts",
            "unseen_pairs": summarize("unseen"),
            "seen_pairs": summarize("seen"),
        }
