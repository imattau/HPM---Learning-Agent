# T1 Trial Debugging - Session State Handoff #4

## Task Objective
Debug T1 trial in experiment_constraint_violation.py to fix failing topology scoring in decoder.py

## Current Progress
- experiment_constraint_violation.py analyzed (lines 1-196 read)
- Line 87 identified: `decode_result = dec.decode(query)` 
- Next: Insert debug code before line 88 check

## Immediate Remaining Tasks (IN ORDER - DO NOT SKIP)

### STEP 1: Edit experiment_constraint_violation.py
**File**: /home/mattthomson/workspace/HPM---Learning-Agent/hpm_fractal_node/experiments/experiment_constraint_violation.py

Insert 8 lines BEFORE line 88 (before the `if isinstance(decode_result, ResolutionRequest):` line):
```python
    # DEBUG T1
    if name == "T1: Clean (A→B→C)":
        print(f"\n[DEBUG T1] decode_result type: {type(decode_result).__name__}")
        if isinstance(decode_result, ResolutionRequest):
            print(f"[DEBUG T1] ResolutionRequest. missing_mu[:4]: {decode_result.missing_mu[:4]}")
        elif isinstance(decode_result, list):
            print(f"[DEBUG T1] List length: {len(decode_result)}")
            if decode_result:
                print(f"[DEBUG T1] First: {decode_result[0].id}")
```

**Method**: Use Edit tool with old_string starting from line 87, new_string includes debug code before line 88.

### STEP 2: Run experiment to capture debug output
```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python3 hpm_fractal_node/experiments/experiment_constraint_violation.py 2>&1
```
Look for [DEBUG T1] lines to determine decode_result type.

### STEP 3: Analyze output
Determine if decode_result is:
- ResolutionRequest → no candidates found
- Empty list [] → candidates rejected  
- List with nodes → some node returned

### STEP 4: Fix decoder.py _score_topological_fit()
**File**: /home/mattthomson/workspace/HPM---Learning-Agent/hfn/decoder.py
Fix based on debug output analysis

### STEP 5: Remove debug code from experiment_constraint_violation.py
Delete the 8 debug lines

### STEP 6: Run clean verification
```bash
python3 /home/mattthomson/workspace/HPM---Learning-Agent/hpm_fractal_node/experiments/experiment_constraint_violation.py 2>&1
```

### STEP 7: Verify Success
Expected: OVERALL: PASS with T1=success, T2=resolution_request, T3=resolution_request

## Key Files
- Experiment: /home/mattthomson/workspace/HPM---Learning-Agent/hpm_fractal_node/experiments/experiment_constraint_violation.py
- Decoder: /home/mattthomson/workspace/HPM---Learning-Agent/hfn/decoder.py

## Execution Mode
- **Mode**: unattended, auto_continue: true
- **Behavior**: DO NOT PAUSE. Complete ALL remaining work.
- **Next Agent**: Read this file FIRST, then execute STEP 1-7 in order.
