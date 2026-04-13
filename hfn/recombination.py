"""
HPM Recombination — structural executor for HFN node merging.

Pure structural executor: no decision logic, no evaluation, no side effects
beyond Forest registration/deregistration.  Observer decides *when* to
recombine; Recombination executes *how*.

Two operations:
  absorb(absorbed, dominant, forest) → HFN
    Deregisters both nodes, registers merged node, returns it.
    Observer transfers dominant's weight/score to the new node.

  compress(node_a, node_b, forest, compressed_id) → HFN
    Creates a compressed node from a co-occurring pair.
    Registers new node in Forest (originals remain active).
    Returns new node.
"""

from __future__ import annotations

from hfn.hfn import HFN
from hfn.forest import Forest


class Recombination:
    """
    Structural executor for HFN node merging.

    Stateless — no instance state is stored or mutated.
    All Forest mutations are explicit and visible in the method signatures.
    """

    def absorb(
        self,
        absorbed: HFN,
        dominant: HFN,
        forest: Forest,
    ) -> HFN:
        """
        Recombine *absorbed* into *dominant*.

        *dominant* is the surviving node (higher weight).  Both nodes are
        deregistered from *forest*; the merged node is registered and returned.

        Observer is responsible for:
        - Calling _init_node(new_node) on the returned node.
        - Copying weight and score from *dominant* to the new node.
        - Recording *absorbed*.id in absorbed_ids for test visibility.

        Parameters
        ----------
        absorbed : HFN
            The weaker node being absorbed.
        dominant : HFN
            The stronger node that survives (its Gaussian dominates).
        forest : Forest
            The active node registry.

        Returns
        -------
        HFN
            Newly merged node (bare structural HFN — no weight/score state).
        """
        new_node = dominant.recombine(absorbed)
        forest.deregister(dominant.id)
        forest.deregister(absorbed.id)
        forest.register(new_node)
        return new_node

    def compress(
        self,
        node_a: HFN,
        node_b: HFN,
        forest: Forest,
        compressed_id: str,
    ) -> HFN:
        """
        Create a compressed node from co-occurring pair (*node_a*, *node_b*).

        *compressed_id* is constructed by Observer before calling
        (format: ``compressed({id_a[:8]},{id_b[:8]})``).

        The new node is registered in *forest*; the originals remain active.

        Observer is responsible for calling _init_node(new_node) on the
        returned node to initialise its weight/score state.

        Parameters
        ----------
        node_a : HFN
            First co-occurring node.
        node_b : HFN
            Second co-occurring node.
        forest : Forest
            The active node registry.
        compressed_id : str
            Pre-constructed identifier for the compressed node.

        Returns
        -------
        HFN
            Newly created compressed node (bare structural HFN).
        """
        new_node = node_a.recombine(node_b)
        new_node.id = compressed_id  # type: ignore[misc]
        forest.register(new_node)
        return new_node

    def recombine_structural(self, macro_a: HFN, macro_b: HFN, forest: Forest,
                             new_id: str = None) -> HFN:
        """
        Structural recombination: concatenates the `inputs` lists of both macros,
        averages their mu and sigma, and preserves all children and edges.
        This creates a new macro that sequentially applies first macro_a's
        constituents, then macro_b's constituents.
        """
        import numpy as np
        # Build new mu and sigma (average of both)
        new_mu = 0.5 * (macro_a.mu + macro_b.mu)
        if macro_a.use_diag and macro_b.use_diag:
            new_sigma = 0.5 * (macro_a.sigma + macro_b.sigma)
            new_node = HFN(mu=new_mu, sigma=new_sigma, use_diag=True)
        else:
            s_sigma = np.diag(macro_a.sigma) if macro_a.use_diag else macro_a.sigma
            o_sigma = np.diag(macro_b.sigma) if macro_b.use_diag else macro_b.sigma
            new_sigma = 0.5 * (s_sigma + o_sigma)
            new_node = HFN(mu=new_mu, sigma=new_sigma)

        # Concatenate inputs
        a_inputs = macro_a.inputs if macro_a.inputs else []
        b_inputs = macro_b.inputs if macro_b.inputs else []
        new_node.inputs = list(a_inputs) + list(b_inputs)
        new_node.relation_type = "recombined"

        # Children: all children from both macros
        new_node._children = list(macro_a.children()) + list(macro_b.children())
        # Edges: preserve all edges from both
        new_node._edges = list(macro_a.edges()) + list(macro_b.edges())

        # Add an edge from the last of macro_a's inputs to the first of macro_b's inputs
        if a_inputs and b_inputs:
            new_node.add_edge(a_inputs[-1], b_inputs[0], "then")

        new_node.id = new_id or f"recomb_{macro_a.id[:4]}_{macro_b.id[:4]}"
        forest.register(new_node)
        return new_node
