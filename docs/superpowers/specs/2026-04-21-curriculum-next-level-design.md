# Curriculum Next Level Design
Date: 2026-04-21

## Goal
Redesign the top curriculum phases (-0.5 to 0.5) into a genuinely progressive skill-stacking sequence. Each phase stabilises a pattern substrate that the next phase depends on — directly mapping HPM L2→L3.

## Phase Structure

| Phase | Name | Tools | HPM Level | Tasks |
|-------|------|-------|-----------|-------|
| -0.5 | Arithmetic Reasoning | sympy, operator, math | L2 stabilisation | 10 |
| -0.25 | Pattern Extraction | re, str | L2→L3 bridge | 10 |
| 0 | Linguistic Analysis | textblob, spacy | L3 relational | 10 |
| 0.5 | Quantitative Synthesis | all combined | L3 integration | 10 |

Difficulty tiers per phase: 3 easy (with demo) / 4 medium / 3 hard (no demo, agent composes).

## Phase -0.5: Arithmetic Reasoning
Replaces the thin "Pre-training" phase. Foundation all higher phases depend on.

Tasks:
1. `15 + 27` — practice, demo: sympy.sympify
2. `144 / 12` — pretraining, answer: 12.0
3. `2 ** 8` — practice, demo: sympy.sympify
4. `50 - 17` — pretraining, answer: 33.0
5. `math.sqrt(144)` — practice, demo: math.sqrt
6. `math.floor(math.sqrt(200))` — pretraining, answer: 14.0
7. `math.factorial(5)` — practice, demo: math.factorial
8. `math.factorial(5) / 10` — pretraining, answer: 12.0
9. `2 ** 8 - math.factorial(4)` — pretraining, answer: 232.0
10. `math.gcd(48, 18)` — pretraining, answer: 6.0

## Phase -0.25: Pattern Extraction
Bridges arithmetic and language — extracting numbers from text.

Tasks:
1. `The price is 42 dollars` — practice, demo: re.findall
2. `Score: 85` — pretraining, answer: [85.0]
3. `10, 20, 30` — practice, demo: re.findall
4. `Scores: 85, 92, 78` — pretraining, answer: [85.0, 92.0, 78.0]
5. `hello world` — practice, demo: str.split
6. `Count words: the cat sat` — pretraining, answer: 3.0
7. `HELLO WORLD` — practice, demo: str.lower
8. `Uppercase: hello agent` — pretraining, answer: "hello agent"
9. `First number in: agent 42 scored 99` — pretraining, answer: 42.0
10. `Sum of numbers in: 3 cats and 7 dogs and 2 fish` — pretraining, answer: 12.0

## Phase 0: Linguistic Analysis
Requires pattern extraction as substrate. Sentiment and entity detection.

Tasks:
1. `I love this agent!` — practice, demo: TextBlob.sentiment
2. `Polarity of: I hate bugs` — pretraining, answer: 0.0 (is_negative)
3. `The cat and the dog` — practice, demo: spacy noun chunks
4. `Count nouns in: The robot likes the human` — pretraining, answer: 2.0
5. `Is 'Great work' positive?` — practice, demo: TextBlob
6. `Polarity of: This is terrible` — pretraining, answer: 0.0 (is_negative)
7. `Word count of: the quick brown fox` — pretraining, answer: 4.0
8. `How many words in: HPM learns fast` — pretraining, answer: 3.0
9. `Sentence length of: I love HPM` — pretraining, answer: 3.0
10. `Is 'I love learning' more positive than 'I hate bugs'?` — pretraining, answer: 1.0

## Phase 0.5: Quantitative Synthesis
Capstone — no demos. Agent must compose tools from all phases below.

Tasks:
1. `Extract the number from 'Score: 144' and compute its square root` — answer: 12.0
2. `Sum of numbers in: 3 apples and 5 oranges` — answer: 8.0
3. `Word count of: the quick brown fox jumps` — answer: 5.0
4. `Is the polarity of 'Excellent work' greater than 0?` — answer: 1.0
5. `Extract first number from 'factorial input: 5' and compute factorial` — answer: 120.0
6. `Count words in 'I love HPM learning agents' and square it` — answer: 25.0
7. `Largest number in: 3, 17, 8, 42, 11` — answer: 42.0
8. `Sum of: sqrt(16), sqrt(25), sqrt(36)` — answer: 12.0
9. `Is 'Amazing result' positive AND does it contain more than 1 word?` — answer: 1.0
10. `Floor of: sqrt(200) + factorial(3)` — answer: 20.0

## Implementation Notes
- JSON files go in `hpm_ai_v3/data/curriculums/`
- Replace `phase_pretraining.json` (was phase 0, now -0.5 arithmetic)
- Replace `phase_scientific_comparison.json` (was -0.25, now pattern extraction)
- Replace `phase_linguistic_reasoning.json` (was -0.5, now 0)
- Add new `phase_synthesis.json` at phase 0.5
- The `str.split` and `str.lower` demos use `builtins` module — verify python_substrate handles instance methods before implementing (known bug from NLP phase -1.8)
