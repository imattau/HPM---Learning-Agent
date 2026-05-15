# Quiz Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Quiz section to the web demo that generates multi-choice questions, scores answers with immediate feedback, and auto-trains on weak topics after quiz completion.

**Architecture:** `QuizAgent` (new) composes `ReasoningAgent` and `MultiAgentReader` to generate and self-evaluate questions. Four new Flask API endpoints in `web_demo.py` handle generate/submit/train/download. A new collapsible Quiz section with two pill sub-tabs (Take Quiz, Download Banks) mirrors the existing UI pattern.

**Tech Stack:** Python 3.10+, Flask, dataclasses, vanilla JS (matching existing web_demo.py style)

---

## File Map

| File | Action |
|---|---|
| `hpm_ai_v6/agents/quiz_agent.py` | Create — QuizQuestion dataclass + QuizAgent |
| `hpm_ai_v6/agents/__init__.py` | Modify — add QuizAgent export |
| `hpm_ai_v6/data/quiz_banks/easy.json` | Create — 20 easy general knowledge questions |
| `hpm_ai_v6/data/quiz_banks/medium.json` | Create — 20 medium questions |
| `hpm_ai_v6/data/quiz_banks/hard.json` | Create — 20 hard questions |
| `hpm_ai_v6/web/web_demo.py` | Modify — Quiz UI section + 4 API endpoints + `_quiz_state` |
| `hpm_ai_v6/tests/test_quiz_agent.py` | Create — unit tests for QuizAgent |

---

## Task 1: QuizQuestion dataclass and QuizAgent skeleton

**Files:**
- Create: `hpm_ai_v6/agents/quiz_agent.py`
- Test: `hpm_ai_v6/tests/test_quiz_agent.py`

- [ ] **Step 1: Write failing test for QuizQuestion dataclass**

```python
# hpm_ai_v6/tests/test_quiz_agent.py
from hpm_ai_v6.agents.quiz_agent import QuizQuestion

def test_quiz_question_fields():
    q = QuizQuestion(
        id="q1",
        question="What is the capital of France?",
        options=["London", "Paris", "Berlin", "Rome"],
        correct_index=1,
        topic="geography",
        explanation="Paris is the capital and largest city of France.",
        difficulty="easy",
        source="bank",
    )
    assert q.id == "q1"
    assert len(q.options) == 4
    assert q.correct_index == 1
    assert q.source == "bank"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v6/tests/test_quiz_agent.py::test_quiz_question_fields -v
```

Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Create `quiz_agent.py` with QuizQuestion and QuizAgent skeleton**

```python
# hpm_ai_v6/agents/quiz_agent.py
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional

from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent
from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader


@dataclass
class QuizQuestion:
    id: str
    question: str
    options: List[str]      # always exactly 4
    correct_index: int       # 0-3
    topic: str
    explanation: str
    difficulty: str          # "easy" | "medium" | "hard"
    source: str              # "model" | "bank"


_BANK_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "quiz_banks")


class QuizAgent:
    """Generates and self-evaluates multi-choice quiz questions using the learned corpus."""

    def __init__(self, reader: MultiAgentReader, reasoning_agent: ReasoningAgent) -> None:
        self._reader = reader
        self._reasoner = reasoning_agent

    def generate_quiz(self, n: int, difficulty: str, source: str) -> List[QuizQuestion]:
        """Generate n questions from model (corpus) or bank (static JSON)."""
        if source == "bank":
            return self._load_bank(difficulty, n)
        questions = []
        for _ in range(n):
            q = self.generate_question(topic=None, difficulty=difficulty)
            if q:
                questions.append(q)
        return questions

    def generate_question(self, topic: Optional[str], difficulty: str) -> Optional[QuizQuestion]:
        """Generate a single question with reasoning verification. Returns None if generation fails."""
        raise NotImplementedError

    def _load_bank(self, difficulty: str, n: int) -> List[QuizQuestion]:
        """Load up to n questions from the static JSON bank for the given difficulty."""
        path = os.path.join(_BANK_DIR, f"{difficulty}.json")
        with open(path) as f:
            raw = json.load(f)
        questions = []
        for item in raw[:n]:
            questions.append(QuizQuestion(
                id=item.get("id", str(uuid.uuid4())),
                question=item["question"],
                options=item["options"],
                correct_index=item["correct_index"],
                topic=item["topic"],
                explanation=item["explanation"],
                difficulty=difficulty,
                source="bank",
            ))
        return questions
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest hpm_ai_v6/tests/test_quiz_agent.py::test_quiz_question_fields -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v6/agents/quiz_agent.py hpm_ai_v6/tests/test_quiz_agent.py
git commit -m "feat: add QuizQuestion dataclass and QuizAgent skeleton"
```

---

## Task 2: Question bank JSON files

**Files:**
- Create: `hpm_ai_v6/data/quiz_banks/easy.json`
- Create: `hpm_ai_v6/data/quiz_banks/medium.json`
- Create: `hpm_ai_v6/data/quiz_banks/hard.json`
- Test: `hpm_ai_v6/tests/test_quiz_agent.py`

- [ ] **Step 1: Write failing test for bank loading**

Add to `hpm_ai_v6/tests/test_quiz_agent.py`:

```python
from unittest.mock import MagicMock
from hpm_ai_v6.agents.quiz_agent import QuizAgent, QuizQuestion

def _make_agent():
    return QuizAgent(MagicMock(), MagicMock())

def test_load_bank_easy():
    agent = _make_agent()
    questions = agent._load_bank("easy", 5)
    assert len(questions) == 5
    for q in questions:
        assert isinstance(q, QuizQuestion)
        assert len(q.options) == 4
        assert 0 <= q.correct_index <= 3
        assert q.difficulty == "easy"
        assert q.source == "bank"
```

- [ ] **Step 2: Run to confirm it fails**

```bash
python -m pytest hpm_ai_v6/tests/test_quiz_agent.py::test_load_bank_easy -v
```

Expected: `FileNotFoundError`

- [ ] **Step 3: Create quiz bank directory and easy.json**

```bash
mkdir -p hpm_ai_v6/data/quiz_banks
```

Create `hpm_ai_v6/data/quiz_banks/easy.json` with 20 questions:

```json
[
  {"id":"e1","question":"What is the capital of France?","options":["London","Paris","Berlin","Rome"],"correct_index":1,"topic":"geography","explanation":"Paris is the capital and largest city of France."},
  {"id":"e2","question":"How many days are in a week?","options":["5","6","7","8"],"correct_index":2,"topic":"general","explanation":"A week has 7 days."},
  {"id":"e3","question":"What colour is the sky on a clear day?","options":["Green","Red","Blue","Yellow"],"correct_index":2,"topic":"science","explanation":"The sky appears blue due to Rayleigh scattering of sunlight."},
  {"id":"e4","question":"Which planet is closest to the Sun?","options":["Venus","Earth","Mars","Mercury"],"correct_index":3,"topic":"astronomy","explanation":"Mercury is the innermost planet in the Solar System."},
  {"id":"e5","question":"How many sides does a triangle have?","options":["2","3","4","5"],"correct_index":1,"topic":"mathematics","explanation":"A triangle is a polygon with exactly three sides."},
  {"id":"e6","question":"What gas do plants absorb from the air?","options":["Oxygen","Nitrogen","Carbon dioxide","Hydrogen"],"correct_index":2,"topic":"biology","explanation":"Plants absorb carbon dioxide during photosynthesis."},
  {"id":"e7","question":"Who wrote Romeo and Juliet?","options":["Charles Dickens","William Shakespeare","Jane Austen","Mark Twain"],"correct_index":1,"topic":"literature","explanation":"Romeo and Juliet was written by William Shakespeare around 1594-1596."},
  {"id":"e8","question":"What is the largest ocean on Earth?","options":["Atlantic","Indian","Arctic","Pacific"],"correct_index":3,"topic":"geography","explanation":"The Pacific Ocean is the largest and deepest ocean, covering more than 165 million km2."},
  {"id":"e9","question":"How many continents are there on Earth?","options":["5","6","7","8"],"correct_index":2,"topic":"geography","explanation":"Earth has 7 continents: Africa, Antarctica, Asia, Australia, Europe, North America, South America."},
  {"id":"e10","question":"What is the chemical symbol for water?","options":["O2","CO2","H2O","NaCl"],"correct_index":2,"topic":"chemistry","explanation":"Water is H2O - two hydrogen atoms bonded to one oxygen atom."},
  {"id":"e11","question":"Which country invented the telephone?","options":["USA","UK","France","Germany"],"correct_index":0,"topic":"history","explanation":"Alexander Graham Bell, working in the USA, is credited with inventing the telephone in 1876."},
  {"id":"e12","question":"What is the boiling point of water at sea level in Celsius?","options":["90","95","100","110"],"correct_index":2,"topic":"science","explanation":"Water boils at 100 degrees Celsius at standard atmospheric pressure."},
  {"id":"e13","question":"What language is spoken in Brazil?","options":["Spanish","English","French","Portuguese"],"correct_index":3,"topic":"geography","explanation":"Brazil is the only Portuguese-speaking country in South America."},
  {"id":"e14","question":"How many legs does a spider have?","options":["6","8","10","12"],"correct_index":1,"topic":"biology","explanation":"Spiders are arachnids and have 8 legs."},
  {"id":"e15","question":"What is the largest mammal in the world?","options":["Elephant","Blue whale","Giraffe","Hippopotamus"],"correct_index":1,"topic":"biology","explanation":"The blue whale is the largest animal ever known to have existed."},
  {"id":"e16","question":"Which metal is liquid at room temperature?","options":["Iron","Gold","Mercury","Silver"],"correct_index":2,"topic":"chemistry","explanation":"Mercury (Hg) is the only metal that is liquid at standard room temperature."},
  {"id":"e17","question":"What shape is the Earth?","options":["Flat","Perfect sphere","Oblate spheroid","Cube"],"correct_index":2,"topic":"science","explanation":"Earth is an oblate spheroid - slightly flattened at the poles and bulging at the equator."},
  {"id":"e18","question":"What is 12 x 12?","options":["124","144","132","148"],"correct_index":1,"topic":"mathematics","explanation":"12 x 12 = 144."},
  {"id":"e19","question":"Which organ pumps blood around the body?","options":["Liver","Kidney","Lung","Heart"],"correct_index":3,"topic":"biology","explanation":"The heart is the muscular organ that pumps blood through the circulatory system."},
  {"id":"e20","question":"What is the currency of Japan?","options":["Yuan","Won","Yen","Ringgit"],"correct_index":2,"topic":"geography","explanation":"Japan currency is the Japanese Yen."}
]
```

- [ ] **Step 4: Create medium.json**

```json
[
  {"id":"m1","question":"What is the powerhouse of the cell?","options":["Nucleus","Ribosome","Mitochondria","Golgi apparatus"],"correct_index":2,"topic":"biology","explanation":"Mitochondria produce ATP through cellular respiration."},
  {"id":"m2","question":"In which year did World War II end?","options":["1943","1944","1945","1946"],"correct_index":2,"topic":"history","explanation":"World War II ended in 1945: Germany surrendered in May, Japan in September."},
  {"id":"m3","question":"What does DNA stand for?","options":["Deoxyribonucleic acid","Dinitrogen acid","Deoxyribose nucleotide array","Dynamic neural assembly"],"correct_index":0,"topic":"biology","explanation":"DNA stands for Deoxyribonucleic acid, the molecule that carries genetic information."},
  {"id":"m4","question":"Which element has atomic number 6?","options":["Nitrogen","Oxygen","Carbon","Boron"],"correct_index":2,"topic":"chemistry","explanation":"Carbon (C) has atomic number 6, meaning it has 6 protons."},
  {"id":"m5","question":"What is the speed of light in a vacuum (approximately)?","options":["300,000 km/s","150,000 km/s","3,000 km/s","30,000 km/s"],"correct_index":0,"topic":"physics","explanation":"Light travels at approximately 299,792 km/s in a vacuum."},
  {"id":"m6","question":"Which philosopher wrote The Republic?","options":["Aristotle","Socrates","Plato","Epicurus"],"correct_index":2,"topic":"philosophy","explanation":"Plato wrote The Republic, a Socratic dialogue on justice and the ideal state."},
  {"id":"m7","question":"What is the chemical formula for table salt?","options":["KCl","NaCl","CaCl2","MgCl2"],"correct_index":1,"topic":"chemistry","explanation":"Table salt is sodium chloride, NaCl."},
  {"id":"m8","question":"In computing, what does CPU stand for?","options":["Central Processing Unit","Core Power Unit","Central Program Utility","Computed Processing Uplink"],"correct_index":0,"topic":"technology","explanation":"CPU stands for Central Processing Unit."},
  {"id":"m9","question":"Which country has the longest coastline?","options":["Russia","USA","Australia","Canada"],"correct_index":3,"topic":"geography","explanation":"Canada has the world longest coastline at approximately 202,080 km."},
  {"id":"m10","question":"What is the Pythagorean theorem?","options":["a+b=c","a2+b2=c2","a*b=c2","a2-b2=c"],"correct_index":1,"topic":"mathematics","explanation":"In a right triangle, a squared plus b squared equals c squared, where c is the hypotenuse."},
  {"id":"m11","question":"Who developed the theory of general relativity?","options":["Isaac Newton","Niels Bohr","Albert Einstein","Max Planck"],"correct_index":2,"topic":"physics","explanation":"Albert Einstein published his theory of general relativity in 1915."},
  {"id":"m12","question":"What is the half-life of Carbon-14?","options":["570 years","5,730 years","57,300 years","573 years"],"correct_index":1,"topic":"chemistry","explanation":"Carbon-14 has a half-life of approximately 5,730 years, used in radiocarbon dating."},
  {"id":"m13","question":"In economics, what does GDP stand for?","options":["Gross Domestic Product","General Debt Position","Gross Development Plan","Global Distribution Potential"],"correct_index":0,"topic":"economics","explanation":"GDP stands for Gross Domestic Product."},
  {"id":"m14","question":"Which part of the brain controls balance and coordination?","options":["Cerebrum","Hippocampus","Amygdala","Cerebellum"],"correct_index":3,"topic":"biology","explanation":"The cerebellum coordinates voluntary movements, balance, and fine motor control."},
  {"id":"m15","question":"What is the main gas in Earth atmosphere?","options":["Oxygen","Carbon dioxide","Argon","Nitrogen"],"correct_index":3,"topic":"science","explanation":"Nitrogen makes up approximately 78% of Earth atmosphere."},
  {"id":"m16","question":"Which ancient wonder was located in Alexandria?","options":["Colossus of Rhodes","Lighthouse of Alexandria","Temple of Artemis","Hanging Gardens"],"correct_index":1,"topic":"history","explanation":"The Lighthouse of Alexandria (Pharos) was one of the Seven Wonders of the Ancient World."},
  {"id":"m17","question":"What unit measures electrical resistance?","options":["Volt","Ampere","Watt","Ohm"],"correct_index":3,"topic":"physics","explanation":"Electrical resistance is measured in Ohms, named after Georg Simon Ohm."},
  {"id":"m18","question":"In music, how many semitones are in an octave?","options":["8","10","12","14"],"correct_index":2,"topic":"music","explanation":"An octave contains 12 semitones in Western music equal temperament."},
  {"id":"m19","question":"Which programming language was created by Guido van Rossum?","options":["Java","Ruby","Python","Perl"],"correct_index":2,"topic":"technology","explanation":"Python was created by Guido van Rossum and first released in 1991."},
  {"id":"m20","question":"What is the Fibonacci sequence?","options":["Each number is double the previous","Each number is the sum of the two preceding","Each number is the square of its position","Each number alternates sign"],"correct_index":1,"topic":"mathematics","explanation":"In the Fibonacci sequence, each number is the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8..."}
]
```

- [ ] **Step 5: Create hard.json**

```json
[
  {"id":"h1","question":"What is the Riemann Hypothesis concerned with?","options":["Distribution of prime numbers","Convergence of infinite series","Topology of manifolds","Solutions to polynomial equations"],"correct_index":0,"topic":"mathematics","explanation":"The Riemann Hypothesis concerns the distribution of non-trivial zeros of the Riemann zeta function, which determines the distribution of prime numbers."},
  {"id":"h2","question":"What phenomenon does Bell theorem rule out in quantum mechanics?","options":["Wave-particle duality","Local hidden variable theories","Quantum entanglement","Wave function collapse"],"correct_index":1,"topic":"physics","explanation":"Bell theorem proves that no local hidden variable theory can reproduce all predictions of quantum mechanics."},
  {"id":"h3","question":"In information theory, what does Shannon entropy measure?","options":["Signal strength","Average information content of a message","Compression ratio","Channel capacity"],"correct_index":1,"topic":"mathematics","explanation":"Shannon entropy H = -sum p(x) log p(x) quantifies the average amount of information in a probability distribution."},
  {"id":"h4","question":"Which neurotransmitter is primarily implicated in reward prediction error signalling?","options":["Serotonin","Acetylcholine","Dopamine","GABA"],"correct_index":2,"topic":"neuroscience","explanation":"Dopamine neurons in the ventral tegmental area encode reward prediction error - firing for unexpected rewards and pausing for expected ones."},
  {"id":"h5","question":"What does the P vs NP problem ask?","options":["Whether polynomial algorithms exist for all problems","Whether every problem whose solution can be verified quickly can also be solved quickly","Whether parallel computation equals sequential computation","Whether infinite sets can be well-ordered"],"correct_index":1,"topic":"computer science","explanation":"P vs NP asks whether every problem verifiable in polynomial time can also be solved in polynomial time."},
  {"id":"h6","question":"In thermodynamics, what does the second law state cannot decrease in an isolated system?","options":["Enthalpy","Free energy","Entropy","Internal energy"],"correct_index":2,"topic":"physics","explanation":"The second law of thermodynamics states that entropy of an isolated system never decreases over time."},
  {"id":"h7","question":"What is the Cambrian Explosion?","options":["A mass extinction 540 million years ago","A rapid diversification of animal phyla 541 million years ago","The formation of the first oceans","The origin of photosynthesis"],"correct_index":1,"topic":"biology","explanation":"The Cambrian Explosion (~541 million years ago) was a rapid appearance of most major animal phyla in the fossil record."},
  {"id":"h8","question":"What does Godel incompleteness theorem prove about formal systems?","options":["All true statements are provable","No consistent system can prove its own consistency","Arithmetic is decidable","All mathematical truths are computable"],"correct_index":1,"topic":"mathematics","explanation":"Godel second incompleteness theorem shows that a sufficiently powerful consistent formal system cannot prove its own consistency."},
  {"id":"h9","question":"In linguistics, what is the Sapir-Whorf hypothesis?","options":["All languages share a universal grammar","Language influences or determines thought","Language evolved from gesture","Children have innate grammatical knowledge"],"correct_index":1,"topic":"linguistics","explanation":"The Sapir-Whorf (linguistic relativity) hypothesis proposes that the language one speaks influences or determines cognition."},
  {"id":"h10","question":"What distinguishes Type I from Type II errors in hypothesis testing?","options":["Type I is false negative, Type II is false positive","Type I is false positive, Type II is false negative","Type I tests means, Type II tests variances","Type I uses z-tests, Type II uses t-tests"],"correct_index":1,"topic":"statistics","explanation":"Type I error is rejecting a true null hypothesis (false positive). Type II error is failing to reject a false null hypothesis (false negative)."},
  {"id":"h11","question":"What is the holographic principle in theoretical physics?","options":["Light behaves as a hologram","A 3D region of space can be fully described by information on its 2D boundary","Black holes emit light","Quantum states are observer-dependent"],"correct_index":1,"topic":"physics","explanation":"The holographic principle proposes that the information content of a volume of space can be encoded on its lower-dimensional boundary."},
  {"id":"h12","question":"What does RNA polymerase do during transcription?","options":["Replicates DNA","Translates mRNA into protein","Synthesises an RNA strand complementary to a DNA template","Splices introns from pre-mRNA"],"correct_index":2,"topic":"biology","explanation":"RNA polymerase reads the DNA template strand and synthesises a complementary mRNA strand during transcription."},
  {"id":"h13","question":"In game theory, what is a Nash equilibrium?","options":["The outcome maximising total social welfare","A state where no player can improve their payoff by unilaterally changing strategy","The outcome reached by always cooperating","The strategy that guarantees minimum loss"],"correct_index":1,"topic":"economics","explanation":"A Nash equilibrium is a set of strategies where no player benefits from deviating unilaterally."},
  {"id":"h14","question":"What is the primary mechanism of CRISPR-Cas9 gene editing?","options":["Viral insertion of new genes","RNA-guided endonuclease cuts at a specific DNA sequence","Methylation of target genes","Homologous recombination without a guide"],"correct_index":1,"topic":"biology","explanation":"CRISPR-Cas9 uses a guide RNA to direct the Cas9 endonuclease to cut a specific DNA sequence."},
  {"id":"h15","question":"In philosophy of mind, what is the hard problem of consciousness?","options":["Explaining motor control","Explaining why physical processes give rise to subjective experience","Explaining memory formation","Explaining language acquisition"],"correct_index":1,"topic":"philosophy","explanation":"David Chalmers hard problem asks why and how physical brain processes give rise to subjective first-person experience."},
  {"id":"h16","question":"What does the Fourier transform decompose a signal into?","options":["Time-domain impulses","Spatial frequencies","Sinusoidal components at different frequencies","Wavelets"],"correct_index":2,"topic":"mathematics","explanation":"The Fourier transform decomposes a signal into its constituent sinusoidal frequencies, each with an amplitude and phase."},
  {"id":"h17","question":"In evolutionary biology, what is genetic drift?","options":["Directional selection pressure","Random changes in allele frequencies due to chance sampling","Migration of genes between populations","Mutation rate variation"],"correct_index":1,"topic":"biology","explanation":"Genetic drift is the random change in allele frequencies across generations due to chance sampling - most significant in small populations."},
  {"id":"h18","question":"What does the Coase theorem state about externalities?","options":["Government must regulate all externalities","If property rights are well-defined and transaction costs are zero, parties will bargain to an efficient outcome","Externalities always require Pigouvian taxes","Market failures cannot be corrected privately"],"correct_index":1,"topic":"economics","explanation":"The Coase theorem argues that with clear property rights and zero transaction costs, private bargaining leads to efficient outcomes."},
  {"id":"h19","question":"In machine learning, what does the bias-variance tradeoff describe?","options":["The tradeoff between training speed and accuracy","The tension between a model sensitivity to training data and its generalisation error","The relationship between learning rate and convergence","The balance between regularisation and loss"],"correct_index":1,"topic":"computer science","explanation":"The bias-variance tradeoff describes how reducing bias (underfitting) tends to increase variance (overfitting) and vice versa."},
  {"id":"h20","question":"What is the anthropic principle in cosmology?","options":["The universe was designed for life","Observations of the universe must be compatible with the existence of observers making those observations","Consciousness creates physical reality","The universe is infinite and eternal"],"correct_index":1,"topic":"philosophy","explanation":"The anthropic principle notes that observations about the universe are constrained by the requirement that observers exist to make them."}
]
```

- [ ] **Step 6: Run bank tests**

```bash
python -m pytest hpm_ai_v6/tests/test_quiz_agent.py -v
```

Expected: all PASSED

- [ ] **Step 7: Add tests for medium and hard banks**

Add to `hpm_ai_v6/tests/test_quiz_agent.py`:

```python
def test_load_bank_medium():
    agent = _make_agent()
    questions = agent._load_bank("medium", 10)
    assert len(questions) == 10
    assert all(q.difficulty == "medium" for q in questions)

def test_load_bank_hard():
    agent = _make_agent()
    questions = agent._load_bank("hard", 3)
    assert len(questions) == 3
    assert all(q.source == "bank" for q in questions)

def test_generate_quiz_bank_source():
    agent = _make_agent()
    questions = agent.generate_quiz(n=5, difficulty="easy", source="bank")
    assert len(questions) == 5
    assert all(isinstance(q, QuizQuestion) for q in questions)
```

- [ ] **Step 8: Commit**

```bash
git add hpm_ai_v6/data/quiz_banks/ hpm_ai_v6/tests/test_quiz_agent.py
git commit -m "feat: add question bank JSON files and bank-loading tests"
```

---

## Task 3: QuizAgent model-generated question logic

**Files:**
- Modify: `hpm_ai_v6/agents/quiz_agent.py`
- Test: `hpm_ai_v6/tests/test_quiz_agent.py`

- [ ] **Step 1: Write failing test for generate_question**

Add to `hpm_ai_v6/tests/test_quiz_agent.py`:

```python
def test_generate_question_returns_quiz_question():
    reader = MagicMock()
    reader.top_concepts.return_value = ["photosynthesis", "gravity", "democracy"]

    reasoner = MagicMock()
    reasoner.reason.return_value = (
        '{"question": "What process do plants use to make food?", '
        '"options": ["Respiration", "Photosynthesis", "Fermentation", "Digestion"], '
        '"correct_index": 1, '
        '"explanation": "Photosynthesis converts light energy into glucose."}'
    )

    agent = QuizAgent(reader, reasoner)
    q = agent.generate_question(topic=None, difficulty="easy")

    assert q is not None
    assert isinstance(q, QuizQuestion)
    assert len(q.options) == 4
    assert 0 <= q.correct_index <= 3
    assert q.source == "model"
    assert q.difficulty == "easy"
```

- [ ] **Step 2: Run to confirm it fails**

```bash
python -m pytest hpm_ai_v6/tests/test_quiz_agent.py::test_generate_question_returns_quiz_question -v
```

Expected: `FAILED` — `NotImplementedError`

- [ ] **Step 3: Implement `generate_question` in `quiz_agent.py`**

Add at the top of `quiz_agent.py` after existing imports:

```python
import json as _json
import re as _re
```

Add this dict and replace the `generate_question` stub:

```python
_DIFFICULTY_PROMPT = {
    "easy": "Focus on a single well-known fact. The question should test direct recall.",
    "medium": "Test application of a concept or moderate inference from known information.",
    "hard": "Test inference across multiple related concepts or relational reasoning.",
}

def generate_question(self, topic: Optional[str], difficulty: str) -> Optional[QuizQuestion]:
    """Generate a single question with reasoning verification. Returns None if generation fails."""
    if topic is None:
        topics = getattr(self._reader, "top_concepts", lambda n: [])(10)
        topic = topics[0] if topics else "general knowledge"

    difficulty_hint = _DIFFICULTY_PROMPT.get(difficulty, _DIFFICULTY_PROMPT["medium"])

    prompt = (
        f"You are a quiz question author. Generate a multiple-choice question about: {topic}\n"
        f"Difficulty: {difficulty}. {difficulty_hint}\n"
        "Return ONLY valid JSON with this exact structure:\n"
        '{"question": "...", "options": ["A", "B", "C", "D"], "correct_index": N, "explanation": "..."}\n'
        "Rules:\n"
        "- options must have exactly 4 items\n"
        "- correct_index is 0-3\n"
        "- distractors must be plausible, not obviously wrong\n"
        "- explanation must say WHY the correct answer is right\n"
    )

    raw = self._reasoner.reason(prompt)
    parsed = self._parse_question_json(raw)
    if parsed is None:
        return None

    # Verification: ask the reasoner to confirm the correct answer
    verify_prompt = (
        f"Question: {parsed['question']}\n"
        f"Options: {parsed['options']}\n"
        f"Claimed correct answer index: {parsed['correct_index']} = "
        f"'{parsed['options'][parsed['correct_index']]}'\n"
        "Is this correct? Reply with only YES or NO."
    )
    verdict = self._reasoner.reason(verify_prompt).strip().upper()
    if "NO" in verdict:
        return None

    return QuizQuestion(
        id=str(uuid.uuid4()),
        question=parsed["question"],
        options=parsed["options"],
        correct_index=parsed["correct_index"],
        topic=topic,
        explanation=parsed["explanation"],
        difficulty=difficulty,
        source="model",
    )

def _parse_question_json(self, raw: str) -> Optional[dict]:
    """Extract and validate JSON from reasoner output."""
    match = _re.search(r'\{.*\}', raw, _re.DOTALL)
    if not match:
        return None
    try:
        data = _json.loads(match.group())
    except _json.JSONDecodeError:
        return None
    if not all(k in data for k in ("question", "options", "correct_index", "explanation")):
        return None
    if len(data["options"]) != 4:
        return None
    if not isinstance(data["correct_index"], int) or not (0 <= data["correct_index"] <= 3):
        return None
    return data
```

- [ ] **Step 4: Run test**

```bash
python -m pytest hpm_ai_v6/tests/test_quiz_agent.py::test_generate_question_returns_quiz_question -v
```

Expected: `PASSED`

- [ ] **Step 5: Add test for verification rejection**

Add to `hpm_ai_v6/tests/test_quiz_agent.py`:

```python
def test_generate_question_rejected_by_verification():
    reader = MagicMock()
    reader.top_concepts.return_value = ["gravity"]

    reasoner = MagicMock()
    # First call: generation; second call: verification returns NO
    reasoner.reason.side_effect = [
        '{"question": "Q?", "options": ["A","B","C","D"], "correct_index": 0, "explanation": "E"}',
        "NO"
    ]

    agent = QuizAgent(reader, reasoner)
    q = agent.generate_question(topic="gravity", difficulty="easy")
    assert q is None
```

- [ ] **Step 6: Run all tests**

```bash
python -m pytest hpm_ai_v6/tests/test_quiz_agent.py -v
```

Expected: all PASSED

- [ ] **Step 7: Commit**

```bash
git add hpm_ai_v6/agents/quiz_agent.py hpm_ai_v6/tests/test_quiz_agent.py
git commit -m "feat: implement QuizAgent.generate_question with reasoning verification"
```

---

## Task 4: Export QuizAgent from agents package

**Files:**
- Modify: `hpm_ai_v6/agents/__init__.py`

- [ ] **Step 1: Write failing import test**

Add to `hpm_ai_v6/tests/test_quiz_agent.py`:

```python
def test_quiz_agent_importable_from_package():
    from hpm_ai_v6.agents import QuizAgent as QA
    assert QA is not None
```

- [ ] **Step 2: Run to confirm it fails**

```bash
python -m pytest hpm_ai_v6/tests/test_quiz_agent.py::test_quiz_agent_importable_from_package -v
```

Expected: `ImportError`

- [ ] **Step 3: Add to `hpm_ai_v6/agents/__init__.py`**

```python
from hpm_ai_v6.agents.quiz_agent import QuizAgent
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest hpm_ai_v6/tests/test_quiz_agent.py -v
```

Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v6/agents/__init__.py
git commit -m "feat: export QuizAgent from agents package"
```

---

## Task 5: API endpoints in web_demo.py

**Files:**
- Modify: `hpm_ai_v6/web/web_demo.py`

- [ ] **Step 1: Add imports and quiz state**

At the top of `web_demo.py`, ensure `send_file` and `os` are imported:

```python
from flask import Flask, request, jsonify, send_file
import os
```

After the existing `_gutenberg_state` dict, add:

```python
_quiz_state: dict = {}  # keyed by question id -> QuizQuestion
_quiz_agent = None
```

- [ ] **Step 2: Initialise QuizAgent at startup**

In the server startup block where `_reasoning_agent` is built (find by searching for where `_build_reader()` or `_reasoning_agent` is assigned), add:

```python
from hpm_ai_v6.agents.quiz_agent import QuizAgent as _QuizAgent

def _build_quiz_agent(local_reader, local_reasoner):
    return _QuizAgent(local_reader, local_reasoner)
```

And in the startup block:

```python
_quiz_agent = _build_quiz_agent(_reader, _reasoning_agent)
```

- [ ] **Step 3: Add four quiz API endpoints before `def main()`**

```python
@app.route("/api/quiz/generate", methods=["POST"])
def api_quiz_generate():
    global _quiz_state
    if _quiz_agent is None:
        return jsonify({"error": "Quiz agent not initialised"}), 503
    data = request.get_json(force=True)
    n = int(data.get("n", 5))
    difficulty = data.get("difficulty", "easy")
    source = data.get("source", "bank")
    questions = _quiz_agent.generate_quiz(n=n, difficulty=difficulty, source=source)
    _quiz_state = {q.id: q for q in questions}
    return jsonify({
        "questions": [
            {"id": q.id, "question": q.question, "options": q.options,
             "topic": q.topic, "difficulty": q.difficulty, "source": q.source}
            for q in questions
        ]
    })


@app.route("/api/quiz/submit", methods=["POST"])
def api_quiz_submit():
    data = request.get_json(force=True)
    question_id = data.get("question_id")
    answer_index = data.get("answer_index")
    if question_id not in _quiz_state:
        return jsonify({"error": "Unknown question id"}), 404
    q = _quiz_state[question_id]
    return jsonify({
        "correct": answer_index == q.correct_index,
        "correct_index": q.correct_index,
        "explanation": q.explanation,
    })


@app.route("/api/quiz/train_gaps", methods=["POST"])
def api_quiz_train_gaps():
    data = request.get_json(force=True)
    topics = data.get("topics", [])
    if not topics:
        return jsonify({"status": "ok", "message": "No topics to train on"})
    msg = _start_wikipedia_cycle(topics=topics)
    return jsonify({"status": "ok", "message": msg})


@app.route("/api/quiz/banks/<difficulty>", methods=["GET"])
def api_quiz_banks(difficulty):
    if difficulty not in ("easy", "medium", "hard"):
        return jsonify({"error": "Invalid difficulty"}), 400
    bank_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "quiz_banks", f"{difficulty}.json"
    )
    return send_file(
        bank_path,
        mimetype="application/json",
        as_attachment=True,
        download_name=f"quiz_bank_{difficulty}.json",
    )
```

- [ ] **Step 4: Verify server starts**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python hpm_ai_v6/web/web_demo.py &
sleep 4
curl -s http://localhost:5000/api/quiz/banks/easy | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d), 'questions OK')"
kill %1
```

Expected: `20 questions OK`

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v6/web/web_demo.py
git commit -m "feat: add quiz API endpoints (generate, submit, train_gaps, banks)"
```

---

## Task 6: Quiz UI section in web_demo.py

**Files:**
- Modify: `hpm_ai_v6/web/web_demo.py`

- [ ] **Step 1: Add Quiz collapsible section HTML**

In the HTML template, after the closing `</div>` of the Reason section (search for `sec-reason`), add:

```html
      <div class="section open" id="sec-quiz">
        <div class="section-header" onclick="toggleSection('sec-quiz')">
          <span class="chevron">&#9662;</span>
          <span class="section-title">Quiz</span>
          <span class="section-badge" id="quiz-badge">multi-choice</span>
        </div>
        <div class="section-body">
          <div class="section-inner">
            <div class="pill-tabs">
              <button class="pill active" onclick="showTab('quiz','take')">Take Quiz</button>
              <button class="pill" onclick="showTab('quiz','download')">Download Banks</button>
            </div>

            <div id="quiz-take" class="sub-panel active">
              <div id="quiz-setup">
                <div style="margin-bottom:12px">
                  <label style="font-weight:600;display:block;margin-bottom:6px">Source</label>
                  <label><input type="radio" name="quiz-source" value="bank" checked> Question Bank</label>
                  &nbsp;&nbsp;
                  <label><input type="radio" name="quiz-source" value="model"> Model-generated</label>
                </div>
                <div style="margin-bottom:12px">
                  <label style="font-weight:600;display:block;margin-bottom:6px">Difficulty</label>
                  <label><input type="radio" name="quiz-diff" value="easy" checked> Easy</label>
                  &nbsp;&nbsp;
                  <label><input type="radio" name="quiz-diff" value="medium"> Medium</label>
                  &nbsp;&nbsp;
                  <label><input type="radio" name="quiz-diff" value="hard"> Hard</label>
                </div>
                <div style="margin-bottom:16px">
                  <label style="font-weight:600;display:block;margin-bottom:6px">Questions</label>
                  <label><input type="radio" name="quiz-n" value="5" checked> 5</label>
                  &nbsp;&nbsp;
                  <label><input type="radio" name="quiz-n" value="10"> 10</label>
                  &nbsp;&nbsp;
                  <label><input type="radio" name="quiz-n" value="20"> 20</label>
                </div>
                <button class="btn" onclick="startQuiz()">Start Quiz</button>
              </div>

              <div id="quiz-play" style="display:none">
                <div id="quiz-progress" style="margin-bottom:12px;font-size:13px;color:var(--muted)"></div>
                <div style="height:4px;background:var(--surface2);border-radius:2px;margin-bottom:20px">
                  <div id="quiz-progress-fill" style="height:4px;background:var(--accent);border-radius:2px;width:0%;transition:width 0.3s"></div>
                </div>
                <div id="quiz-question" style="font-size:16px;font-weight:600;margin-bottom:16px"></div>
                <div id="quiz-options" style="display:flex;flex-direction:column;gap:8px"></div>
                <div id="quiz-feedback" style="display:none;margin-top:16px;padding:12px;border-radius:6px;font-size:14px"></div>
                <button id="quiz-next-btn" class="btn" style="display:none;margin-top:16px" onclick="nextQuestion()">Next &rarr;</button>
              </div>

              <div id="quiz-complete" style="display:none">
                <h3 style="margin-bottom:12px">Quiz Complete</h3>
                <div id="quiz-score" style="font-size:24px;font-weight:700;color:var(--accent);margin-bottom:16px"></div>
                <div id="quiz-gap-list" style="margin-bottom:16px"></div>
                <button id="train-gaps-btn" class="btn" style="display:none" onclick="trainGaps()">Train on gaps</button>
                <div id="train-gaps-status" style="margin-top:10px;font-size:13px;color:var(--muted)"></div>
              </div>
            </div>

            <div id="quiz-download" class="sub-panel" style="display:none">
              <table style="width:100%;border-collapse:collapse">
                <tr style="border-bottom:1px solid var(--border)">
                  <td style="padding:12px 0"><strong>Easy</strong> &mdash; Single-concept factual recall</td>
                  <td style="text-align:right"><a href="/api/quiz/banks/easy" download class="btn">Download</a></td>
                </tr>
                <tr style="border-bottom:1px solid var(--border)">
                  <td style="padding:12px 0"><strong>Medium</strong> &mdash; Concept application and moderate inference</td>
                  <td style="text-align:right"><a href="/api/quiz/banks/medium" download class="btn">Download</a></td>
                </tr>
                <tr>
                  <td style="padding:12px 0"><strong>Hard</strong> &mdash; Relational reasoning across multiple concepts</td>
                  <td style="text-align:right"><a href="/api/quiz/banks/hard" download class="btn">Download</a></td>
                </tr>
              </table>
            </div>
          </div>
        </div>
      </div>
```

- [ ] **Step 2: Add Quiz JavaScript before closing `</script>` tag**

```javascript
    // ── Quiz ──
    var _quizQuestions = [];
    var _quizIndex = 0;
    var _quizCorrect = 0;
    var _quizFailedTopics = [];

    async function startQuiz() {
      var source = document.querySelector('input[name="quiz-source"]:checked').value;
      var diff   = document.querySelector('input[name="quiz-diff"]:checked').value;
      var n      = parseInt(document.querySelector('input[name="quiz-n"]:checked').value);
      document.getElementById('quiz-badge').textContent = 'loading…';
      var res = await fetch('/api/quiz/generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({n: n, difficulty: diff, source: source})
      });
      var data = await res.json();
      if (data.error) { alert(data.error); return; }
      _quizQuestions = data.questions;
      _quizIndex = 0; _quizCorrect = 0; _quizFailedTopics = [];
      document.getElementById('quiz-setup').style.display = 'none';
      document.getElementById('quiz-complete').style.display = 'none';
      document.getElementById('quiz-play').style.display = 'block';
      document.getElementById('quiz-badge').textContent = diff + ' · ' + source;
      showQuestion();
    }

    function showQuestion() {
      var q = _quizQuestions[_quizIndex];
      var total = _quizQuestions.length;
      document.getElementById('quiz-progress').textContent = 'Question ' + (_quizIndex + 1) + ' of ' + total;
      document.getElementById('quiz-progress-fill').style.width = Math.round((_quizIndex / total) * 100) + '%';
      document.getElementById('quiz-question').textContent = q.question;
      document.getElementById('quiz-feedback').style.display = 'none';
      document.getElementById('quiz-next-btn').style.display = 'none';
      var labels = ['A', 'B', 'C', 'D'];
      var optDiv = document.getElementById('quiz-options');
      optDiv.innerHTML = '';
      q.options.forEach(function(opt, i) {
        var btn = document.createElement('button');
        btn.className = 'btn';
        btn.style.textAlign = 'left';
        btn.style.width = '100%';
        btn.textContent = labels[i] + '. ' + opt;
        btn.onclick = (function(idx){ return function(){ submitAnswer(idx); }; })(i);
        optDiv.appendChild(btn);
      });
    }

    async function submitAnswer(answerIndex) {
      var q = _quizQuestions[_quizIndex];
      document.getElementById('quiz-options').querySelectorAll('button').forEach(function(b){ b.disabled = true; });
      var res = await fetch('/api/quiz/submit', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question_id: q.id, answer_index: answerIndex})
      });
      var data = await res.json();
      var fb = document.getElementById('quiz-feedback');
      var labels = ['A', 'B', 'C', 'D'];
      fb.style.display = 'block';
      if (data.correct) {
        _quizCorrect++;
        fb.style.background = 'var(--success-dim)';
        fb.style.color = 'var(--accent)';
        fb.innerHTML = '✓ Correct! ' + data.explanation;
      } else {
        _quizFailedTopics.push(q.topic);
        fb.style.background = 'var(--danger-dim)';
        fb.style.color = 'var(--danger)';
        fb.innerHTML = '✗ Incorrect. Correct answer: ' + labels[data.correct_index] + '. ' + data.explanation;
      }
      document.getElementById('quiz-next-btn').style.display = 'inline-block';
    }

    function nextQuestion() {
      _quizIndex++;
      if (_quizIndex >= _quizQuestions.length) { showCompletion(); } else { showQuestion(); }
    }

    function showCompletion() {
      document.getElementById('quiz-play').style.display = 'none';
      document.getElementById('quiz-complete').style.display = 'block';
      document.getElementById('quiz-progress-fill').style.width = '100%';
      document.getElementById('quiz-score').textContent = _quizCorrect + ' / ' + _quizQuestions.length + ' correct';
      var uniqueTopics = _quizFailedTopics.filter(function(t, i, a){ return a.indexOf(t) === i; });
      var gapDiv = document.getElementById('quiz-gap-list');
      var trainBtn = document.getElementById('train-gaps-btn');
      if (uniqueTopics.length > 0) {
        gapDiv.innerHTML = '<p style="color:var(--muted);margin-bottom:8px">Weak topics:</p>' +
          uniqueTopics.map(function(t){ return '<span style="display:inline-block;padding:3px 10px;margin:3px;border-radius:4px;background:var(--surface2);font-size:13px">' + t + '</span>'; }).join('');
        trainBtn.style.display = 'inline-block';
        trainBtn.dataset.topics = JSON.stringify(uniqueTopics);
      } else {
        gapDiv.innerHTML = '<p style="color:var(--accent)">Perfect score — no gaps!</p>';
      }
    }

    async function trainGaps() {
      var btn = document.getElementById('train-gaps-btn');
      var topics = JSON.parse(btn.dataset.topics);
      btn.disabled = true;
      document.getElementById('train-gaps-status').textContent = 'Starting training…';
      var res = await fetch('/api/quiz/train_gaps', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({topics: topics})
      });
      var data = await res.json();
      document.getElementById('train-gaps-status').textContent = data.message || 'Training started.';
    }
```

- [ ] **Step 3: Verify Quiz section appears in served HTML**

```bash
python hpm_ai_v6/web/web_demo.py &
sleep 4
curl -s http://localhost:5000 | grep -c "sec-quiz"
kill %1
```

Expected: `1`

- [ ] **Step 4: Commit**

```bash
git add hpm_ai_v6/web/web_demo.py
git commit -m "feat: add Quiz UI section with Take Quiz and Download Banks tabs"
```
