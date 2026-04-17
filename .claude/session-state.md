# Gap Fix Planning Session - CONTINUATION STATE

## EXECUTION MODE
- **mode**: unattended
- **auto_continue**: true
- **task_status**: READY FOR CONTINUATION - GENERATE 4 GAP PLANS

## Task Objective
Generate detailed implementation plans for fixing 4 critical HPM gaps with exact file:line changes.

1. **Gap 1: No Decay/Forgetting** - Add exponential decay to pattern weights
2. **Gap 2: No Boredom Mechanism** - Add satiation tracking to curiosity evaluator  
3. **Gap 3: L5 Meta-Pattern Layer** - Expand MetaStrategyController strategy monitoring
4. **Gap 4: Temporal Pattern Field** - Add timestamp/recency to pattern weights

## Key Architecture Knowledge
- HFN nodes already store metadata: `D=4 [successes, attempts, total_oracle_calls, last_timestamp]`
- HFN is the pattern substrate - use it for all structured data
- **Recommended Implementation**: Store decay/usage metadata in Observer class (manages pattern dynamics)

## REQUIRED READS FOR CONTINUATION AGENT
Read these files COMPLETELY to generate gap plans:

1. **base_agent.py** key sections:
   - Lines 40-120 (class def, __init__, Observer setup)
   - Lines 220-240 (curiosity/select_next_task)
   - Lines 565-595 (get_weight mechanism)
   - Lines 140-160 (pattern storage)

2. **meta_controller.py**:
   - Lines 1-150 (full class, SolveRecord, methods)

3. **Search patterns** (expect NONE - gaps to fill):
   - "decay" or "forgetting"
   - "boredom" or "satiation"
   - "timestamp" in pattern storage
   - "last_used" or "recency"

## Required Output Format
For EACH gap, output:
```
## Gap N: [Name]

**File:** [absolute path]
**Location:** [class:method, line range]

**Current Code:**
[brief description or snippet]

**Changes:**
1. Add/Modify [specific class/method]
   - Lines: [exact line numbers]
   - Change: [before → after code snippet]
2. [next change if needed]

**Verification:**
- Test: [how to verify fix]
- Expected: [behavior]
```

## File Paths (Absolute)
- base_agent.py: `/home/mattthomson/workspace/HPM---Learning-Agent/hpm_ai_v2/agents/base_agent.py`
- meta_controller.py: `/home/mattthomson/workspace/HPM---Learning-Agent/hpm_ai_v2/utils/meta_controller.py`
- hfn_meta_controller.py: `/home/mattthomson/workspace/HPM---Learning-Agent/hpm_ai_v2/utils/hfn_meta_controller.py`
- hfn_forward_model.py: `/home/mattthomson/workspace/HPM---Learning-Agent/hpm_ai_v2/utils/hfn_forward_model.py`
- Project root: `/home/mattthomson/workspace/HPM---Learning-Agent/`

## Continuation Instructions (CRITICAL)
- **DO NOT PAUSE** - unattended mode active
- Spawn continuation agent to read this file FIRST
- Execute all required reads in parallel
- Generate ALL 4 gap plans COMPLETELY
- Output directly to user with absolute file paths
- Include exact line numbers and code diffs
- Do not ask for confirmation
