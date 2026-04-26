# hpm_ai_v4 Exploration - CONTINUATION TASK

## PRIMARY OBJECTIVE
User wants: concise map of hpm_ai_v4 covering:
1. All files and their responsibilities
2. Key classes and methods
3. **How replicator dynamics/weight updates work** - specifically where pattern weights updated and how FlatPattern vs HierarchicalPattern compete
4. Focus: agents/, evaluators/, simulations/layered_agent.py

## CONFIRMED STRUCTURE
- /agents/: decoders.py, agent.py, reasoning.py, meta_decoder_policy.py
- /evaluators/: EXISTS - need to list files
- /simulations/: EXISTS - need to find layered_agent.py and other files
- /pattern.py: CORE - contains pattern classes
- /tools/, /io/, /system.py, /curriculum.py, /field.py exist
- 85 total .py files

## CRITICAL: INCOMPLETE WORK
1. List all files in evaluators/
2. List all files in simulations/
3. Read pattern.py - find FlatPattern, HierarchicalPattern classes
4. Read agents/agent.py - find Agent class, pattern weight update methods
5. Read simulations/layered_agent.py - understand layered pattern behavior
6. Search for "weight" or "fitness" or "replicator" in codebase to find update mechanism
7. Compile final map showing weight update flow

## OUTPUT REQUIRED
Return DIRECT MESSAGE (no files) with:
- Files/responsibilities table
- Key classes and methods summary
- DETAILED explanation: where weights updated, how FlatPattern vs Hierarchical compete
- Code snippets ONLY if load-bearing (actual update code)

## MODE
unattended - do NOT ask for confirmation
complete ALL work before returning
