"""Check that a study trainer's HFN storage has the intended structure.

``storage_problems`` returns human-readable problems (empty when sound). It is
read-only and cheap enough to run after ingestion, after loading a checkpoint,
and in tests.
"""
from __future__ import annotations

from hpm_ai_v2.agents.text_structure import content_stems


def storage_problems(trainer) -> list[str]:
    from hpm_ai_v2.training.study import _sentences_with_offsets

    if getattr(trainer, "storage_version", 1) < 2:
        return ["legacy storage (version 1): sentences are a flat copy outside the hierarchy; "
                "re-ingest the sources to rebuild"]
    agent, problems = trainer.agent, []
    sentence_owner: dict[str, str] = {}
    for source_id, source in trainer.sources.items():
        passage = agent.patterns.get(trainer.source_to_node[source_id])
        if passage is None:
            problems.append(f"{source_id}: passage node is missing")
            continue
        paragraphs = passage.children()
        if not paragraphs:
            problems.append(f"{source_id}: passage has no paragraphs")
        _check_chain(problems, passage, f"{source_id}: passage")
        spans = _sentences_with_offsets(source.text)
        expected = [trainer._sentence_node_id(source_id, s, start) for s, start, _ in spans]
        found = [child.id for paragraph in paragraphs for child in paragraph.children()]
        if found != expected:
            problems.append(f"{source_id}: sentence nodes {len(found)} do not match source sentences {len(expected)} in order")
        if trainer.source_to_answer_nodes.get(source_id) != expected:
            problems.append(f"{source_id}: source_to_answer_nodes does not match the hierarchy")
        for paragraph in paragraphs:
            _check_chain(problems, paragraph, f"{source_id}: paragraph {paragraph.metadata.get('ordinal')}")
            for child in paragraph.children():
                if child.id in sentence_owner:
                    problems.append(f"{child.id}: belongs to two paragraphs")
                sentence_owner[child.id] = paragraph.id
                meta = child.metadata
                if child.relation_type != "sentence" or meta.get("type") != "sentence":
                    problems.append(f"{child.id}: not a sentence node")
                    continue
                if source.text[meta["start"]:meta["end"]] != trainer.answer_node_data.get(child.id, {}).get("text"):
                    problems.append(f"{child.id}: offsets do not reproduce the source sentence")
                if meta.get("stems") != content_stems(source.text[meta["start"]:meta["end"]]):
                    problems.append(f"{child.id}: stored stems differ from the sentence")
                if not child.children():
                    problems.append(f"{child.id}: sentence has no word children")
    for node_id in trainer.answer_node_data:
        if node_id not in sentence_owner:
            problems.append(f"{node_id}: sentence is outside the document hierarchy")
    argument_terms: set[str] = set()
    for node_id, node in agent.patterns.items():
        if node.relation_type == "proposition":
            _check_proposition(problems, agent, node, sentence_owner, argument_terms)
    for node_id, node in agent.patterns.items():
        if node.relation_type != "term":
            continue
        children = node.children()
        stem_ = node.metadata.get("stem")
        if not children or (len(children) < 2 and stem_ not in argument_terms):
            problems.append(f"{node_id}: term has fewer than two sentences and is not a relation argument")
        for child in children:
            if stem_ not in child.metadata.get("stems", ()):
                problems.append(f"{node_id}: child {child.id} does not mention the stem")
            if child.id not in sentence_owner:
                problems.append(f"{node_id}: child {child.id} is not a stored sentence")
    shared = {t for t, ids in agent.term_sentences.items() if len(ids) >= 2}
    stored = {n.metadata["stem"] for n in agent.patterns.values() if n.relation_type == "term"}
    if not shared <= stored:
        problems.append(f"terms in two or more sentences have no node: {sorted(shared - stored)[:5]}")
    if not argument_terms <= stored:
        problems.append(f"relation arguments have no term node: {sorted(argument_terms - stored)[:5]}")
    return problems


def _check_proposition(problems, agent, node, sentence_owner, argument_terms) -> None:
    from hpm_ai_v2.agents.propositions import COPULA, ROLE_ORDER

    meta, label = node.metadata, node.id
    children = node.children()
    if len(children) != 1 or children[0].id != meta.get("sentence_id") or children[0].id not in sentence_owner:
        problems.append(f"{label}: proposition is not supported by exactly one stored sentence")
        return
    if not any(e.relation == "supported_by" and e.target is children[0] for e in node.edges()):
        problems.append(f"{label}: missing supported_by edge")
    sentence = children[0]
    stems = set(sentence.metadata.get("stems", ()))
    roles = meta.get("roles", {})
    ordered = [r for r in ROLE_ORDER if r in roles]
    if not roles or set(roles) - set(ROLE_ORDER) or node.relation_params.get("roles") != ordered:
        problems.append(f"{label}: roles are malformed")
        return
    if [i.metadata.get("stem") for i in node.inputs] != [roles[r] for r in ordered]:
        problems.append(f"{label}: inputs do not match the role fillers")
    if meta.get("source_id") != sentence.metadata.get("source_id"):
        problems.append(f"{label}: source differs from its sentence")
    if meta.get("predicate") != COPULA and meta.get("predicate") not in stems:
        problems.append(f"{label}: predicate {meta.get('predicate')!r} is not in its sentence")
    for role, key in roles.items():
        argument_terms.add(key)
        if key not in stems:
            problems.append(f"{label}: {role} {key!r} is not in its sentence")


def _check_chain(problems: list[str], node, label: str) -> None:
    children = node.children()
    chain = [(e.source.id, e.target.id) for e in node.edges() if e.relation == "next"]
    expected = [(a.id, b.id) for a, b in zip(children, children[1:])]
    if chain != expected:
        problems.append(f"{label}: next edges do not chain its {len(children)} children in order")
