"""Regression gate for the default relation extractor.

The sentences and expected relations below were written by hand in three
rounds. The first set was used to develop the rules, the second to correct
them after its first run, the third was run once, untouched, to estimate
quality (precision 0.93, recall 0.93 on 15 relations). They are small, so these
thresholds guard against regressions; they are not a benchmark.
"""
import pytest

from hpm_ai_v2.agents.propositions import HeuristicRoleExtractor
from hpm_ai_v2.agents.text_structure import stem

SET1 = [
 ("Solar panels convert sunlight into electricity.", [("panels", "convert", "sunlight")]),
 ("Wind turbines convert moving air into electricity.", [("turbines", "convert", "air")]),
 ("Photosynthesis is the process plants use to convert light energy into chemical energy.", [("photosynthesis", "be", "process")]),
 ("The Calvin cycle then uses ATP and NADPH to fix carbon dioxide into glucose.", [("cycle", "uses", "atp"), ("cycle", "uses", "nadph")]),
 ("Oxygen is released as a by-product when water molecules are split.", [(None, "released", "oxygen"), (None, "split", "molecules")]),
 ("It takes place in the chloroplasts, which contain the pigment chlorophyll.", [("chloroplasts", "contain", "chlorophyll")]),
 ("The light-dependent reactions capture sunlight and produce ATP and NADPH.", [("reactions", "capture", "sunlight"), ("reactions", "produce", "atp"), ("reactions", "produce", "nadph")]),
 ("Cellular respiration releases energy from glucose using oxygen in the mitochondria.", [("respiration", "releases", "energy")]),
 ("Bats navigate in darkness using echolocation.", []),
 ("Honeybees perform a waggle dance to communicate the direction and distance of food sources to hive mates.", [("honeybees", "perform", "dance")]),
 ("Neurons release neurotransmitters at synapses.", [("neurons", "release", "neurotransmitters")]),
 ("The brain communicates through electrochemical signals across neurons and synapses.", []),
 ("Bees use visual cues to point hive mates toward nectar sources.", [("bees", "use", "cues")]),
 ("Synaptic plasticity strengthens connections between neurons.", [("plasticity", "strengthens", "connections")]),
 ("Plants use light to make glucose.", [("plants", "use", "light")]),
 ("Chlorophyll absorbs light.", [("chlorophyll", "absorbs", "light")]),
 ("The enzyme breaks down the sugar.", [("enzyme", "breaks", "sugar")]),
 ("Glucose is a simple sugar.", [("glucose", "be", "sugar")]),
 ("Factors such as light intensity limit the rate.", [("intensity", "limit", "rate")]),
 ("Mitochondria produce ATP.", [("mitochondria", "produce", "atp")]),
 ("Water is wet.", []),
 ("The results were published by the team.", [("team", "published", "results")]),
 ("Heat melts the ice.", [("heat", "melts", "ice")]),
 ("Wind farms are often built offshore.", [(None, "built", "farms")]),
 ("What does the cat chase?", []),
 ("Rain falls.", []),
 ("The sun heats the earth and the oceans.", [("sun", "heats", "earth"), ("sun", "heats", "oceans")]),
 ("Plants absorb carbon dioxide from the air.", [("plants", "absorb", "dioxide")]),
 ("Fungi decompose dead organisms.", [("fungi", "decompose", "organisms")]),
 ("The cell contains a nucleus.", [("cell", "contains", "nucleus")]),
]
SET2 = [
 ("Volcanoes erupt molten rock.", [("volcanoes", "erupt", "rock")]),
 ("The heart pumps blood through the body.", [("heart", "pumps", "blood")]),
 ("Antibiotics kill bacteria.", [("antibiotics", "kill", "bacteria")]),
 ("Enzymes speed up chemical reactions.", [("enzymes", "speed", "reactions")]),
 ("The moon orbits the Earth.", [("moon", "orbits", "earth")]),
 ("Red blood cells carry oxygen to the tissues.", [("cells", "carry", "oxygen")]),
 ("Insulin is a hormone.", [("insulin", "be", "hormone")]),
 ("The Nile is a long river in Africa.", [("nile", "be", "river")]),
 ("Vaccines were developed by scientists.", [("scientists", "developed", "vaccines")]),
 ("Glaciers carve valleys.", [("glaciers", "carve", "valleys")]),
 ("The treaty was signed by both nations.", [("nations", "signed", "treaty")]),
 ("Bacteria cause many diseases.", [("bacteria", "cause", "diseases")]),
 ("Electrons orbit the nucleus.", [("electrons", "orbit", "nucleus")]),
 ("The committee approved the budget and rejected the proposal.", [("committee", "approved", "budget"), ("committee", "rejected", "proposal")]),
 ("Soil contains minerals, which plants absorb.", [("soil", "contains", "minerals")]),
 ("Rainfall.", []),
 ("Why do plants need water?", []),
 ("Tides are caused by the gravity of the moon.", [("gravity", "caused", "tides")]),
 ("The company builds electric cars.", [("company", "builds", "cars")]),
 ("Earthquakes occur along fault lines.", []),
 ("The river flows into the sea.", []),
 ("Lions hunt zebras.", [("lions", "hunt", "zebras")]),
 ("A catalyst lowers the activation energy of a reaction.", [("catalyst", "lowers", "energy")]),
 ("Photons carry energy.", [("photons", "carry", "energy")]),
 ("Chlorophyll is the pigment that gives leaves their green colour.", [("chlorophyll", "be", "pigment")]),
 ("The professor wrote a textbook.", [("professor", "wrote", "textbook")]),
 ("Dogs bark.", []),
 ("The bridge was destroyed in the storm.", [(None, "destroyed", "bridge")]),
 ("Muscles convert chemical energy into motion.", [("muscles", "convert", "energy")]),
 ("Roots absorb water and minerals from the soil.", [("roots", "absorb", "water"), ("roots", "absorb", "minerals")]),
]
SET3 = [
 ("Hurricanes form over warm ocean water.", []),
 ("The immune system fights infections.", [("system", "fights", "infections")]),
 ("Coal produces carbon dioxide when it burns.", [("coal", "produces", "dioxide")]),
 ("The telescope was invented in the seventeenth century.", [(None, "invented", "telescope")]),
 ("Honey bees collect nectar from flowers.", [("bees", "collect", "nectar")]),
 ("Magma cools into igneous rock.", []),
 ("Calcium strengthens bones and teeth.", [("calcium", "strengthens", "bones"), ("calcium", "strengthens", "teeth")]),
 ("A thermometer measures temperature.", [("thermometer", "measures", "temperature")]),
 ("Sharks are fish.", []),
 ("The senate passed the law.", [("senate", "passed", "law")]),
 ("What does a thermostat control?", []),
 ("Wolves hunt in packs.", []),
 ("The experiment was repeated by independent laboratories.", [("laboratories", "repeated", "experiment")]),
 ("Sunlight heats the surface of the planet.", [("sunlight", "heats", "surface")]),
 ("Carbon forms strong bonds with other atoms.", [("carbon", "forms", "bonds")]),
 ("The drug reduces inflammation.", [("drug", "reduces", "inflammation")]),
 ("Satellites orbit the planet.", [("satellites", "orbit", "planet")]),
 ("Cheetahs run fast.", []),
 ("The author published three novels.", [("author", "published", "novels")]),
 ("Algae produce oxygen.", [("algae", "produce", "oxygen")]),
]


def _expected(rows):
    return {(a and stem(a), "be" if p == "be" else stem(p), b and stem(b)) for a, p, b in rows}


def _extracted(extractor, sentence):
    return {(p.roles.get("AGENT", (None,))[0], p.predicate, p.roles.get("PATIENT", (None,))[0])
            for p in extractor.extract(sentence)}


def _score(sets):
    extractor = HeuristicRoleExtractor()
    tp = fp = fn = 0
    for sentence, rows in (row for s in sets for row in s):
        want, got = _expected(rows), _extracted(extractor, sentence)
        tp, fp, fn = tp + len(want & got), fp + len(got - want), fn + len(want - got)
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def test_precision_and_recall_hold_on_the_authored_sets():
    for sets, min_precision, min_recall in [((SET3,), 0.9, 0.9), ((SET1, SET2, SET3), 0.95, 0.9)]:
        precision, recall = _score(sets)
        assert precision >= min_precision and recall >= min_recall, (precision, recall)


@pytest.mark.parametrize("sentence", [
    "What does the cat chase?", "Why do plants need water?", "It takes place somewhere.",
    "They convert it.", "Rain falls.", "Dogs bark.", "",
])
def test_it_emits_nothing_when_no_clear_relation_exists(sentence):
    assert HeuristicRoleExtractor().extract(sentence) == []


def test_roles_carry_head_stems_and_phrase_text():
    [prop] = HeuristicRoleExtractor().extract("Solar panels convert sunlight into electricity.")
    assert prop.predicate == stem("convert") and prop.voice == "active" and prop.lemma == "convert"
    assert prop.roles == {"AGENT": (stem("panels"), "solar panels"), "PATIENT": (stem("sunlight"), "sunlight")}
    [passive] = HeuristicRoleExtractor().extract("The treaty was signed by both nations.")
    assert passive.voice == "passive" and set(passive.roles) == {"AGENT", "PATIENT"}
    assert HeuristicRoleExtractor().extract("Glucose is a simple sugar.")[0].voice == "copula"
