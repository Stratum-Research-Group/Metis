# Deliberative Reasoning over Topological Graphs

Passive retrieval models treat memory as a static lookup table, returning top-k chunks in a single pass. Human cognition instead relies on active deliberation: formulating hypotheses, inspecting clues, backtracking from dead ends, and synthesizing evidence.

Dianoia implements deliberative search over topological memory substrates:
- The walk is seeded via stochastic resonance retrieval.
- At each step, a controller model inspects the current passage and its topological neighbours.
- The controller chooses among three discrete actions: EXPAND to traverse an edge, BACKTRACK to retrace steps upon encountering irrelevance, or ANSWER once sufficient grounded evidence is gathered.
- Traversal logs preserve the walk trajectory, creating training data for future learned walk policies.
