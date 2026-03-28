# HPM Fractal Node: Design Specification (v1.6)

## Overview
In the broader **HPM (Hierarchical Pattern Modelling) Framework**, the **HPM Fractal Node** serves as the universal "Cell". The framework is structured as a **Dynamic Forest** (The Fluid Holon) where intelligence emerges from the local interactions of these nodes, rather than a rigid top-down tree.

## 1. The Core Holon Definition
A node $h$ is a triple representing a compressed state, a transition logic, and a grounded state:
$$h = \langle \mathcal{G}_{\mu, \Sigma}, \mathcal{P}_{\{V, E\}}, \mathcal{G}_{\{l_1 \dots l_n\}} \rangle$$

### 1.1. The Meta-Pattern (Identity)
The external interface is a single Gaussian distribution in a $D$-dimensional latent space $\mathcal{Z}$:
$$\mathcal{G}_{id} = \mathcal{N}(\mu_h, \Sigma_h)$$
Where $\mu_h \in \mathbb{R}^D$ is the centroid of the pattern and $\Sigma_h$ is the covariance matrix representing the "blur" or allowed variance of the concept.

### 1.2. The Polygraph Body (Structure)
The internal structure is a Directed Acyclic Graph (DAG) where every vertex $v$ is itself an HPM Fractal Node:
$$\mathcal{P}_{Body} = \{V, E\} \mid \forall v \in V, v \in \mathcal{H}$$
The edges $E$ are defined by a Transition Probability Matrix $T_{ij}$, representing the causal or spatial relationship between sub-nodes.

## 2. The Global HPM Manifold (The Forest)
The HPM Framework provides the Latent Space ($\mathcal{Z}$) where all nodes coexist.
* **The Address Space:** Every node's Meta-Pattern ($\mathcal{G}_{id}$) is mapped to a coordinate.
* **The Proximity Logic:** The framework identifies clusters. When a new signal $x^*$ appears, the framework doesn't look for specific "Atoms"; it looks for any Meta-Pattern with a similar Gaussian signature.

## 3. Framework-Level Operations
The framework acts as the "Physics Engine," enforcing three global laws:

* **The Law of Least Surprise (Epistemic Drive):** The framework monitors Global Epistemic Loss. When an observer's "blurry" $L_5$ Meta-Pattern fails to predict data, the framework facilitates the Query. It provides the resources for the node to "Expand" its Polygraph Body ($\mathcal{P}_{Body}$) to whatever depth is required.
* **The Law of Structural Parsimony (Compression):** The framework rewards high Total Scores ($S_h$). If a complex polygraph can be accurately represented by a single Gaussian Meta-Pattern, the internal nodes hibernate. This is the Fractal Collapse.
* **The Law of Competitive Exclusion (The Matrix):** The framework maintains the Incompatibility Matrix ($\kappa_{hj}$). It ensures different Meta-Patterns don't compete for the same sensory data, preventing "Cognitive Dissonance."

## 4. The Epistemic Query & Expansion Math
The node is passive; expansion is a function of the Observer's ($\mathcal{O}$) failure. The Observer treats $h$ as a point $L_1$ until the Epistemic Loss ($\mathcal{L}$) triggers a query.

### 4.1. The Loss Function
The Observer calculates the Kullback-Leibler (KL) Divergence between the incoming signal ($x$) and the node's Meta-Pattern:
$$\mathcal{L}(x, h) = D_{KL}(P(x) \parallel \mathcal{G}_{id})$$

### 4.2. The Expansion Trigger
The node remains compressed if $\mathcal{L} < \tau$, where $\tau$ is the Inquiry Threshold. If $\mathcal{L} \geq \tau$, the Observer "pokes" the node, forcing the reveal of the Polygraph:
$$\text{Query}(h) \rightarrow \text{Expand}(\mathcal{P}_{Body}) \text{ iff } D_{KL}(P(x) \parallel \mathcal{G}_{id}) > \tau$$

## 5. The "Best Attempt" & Fluid Integration
When the AI encounters a novel signal $x^*$, it spawns a set of candidate $L_1$ nodes $\{s_1, s_2, \dots, s_n\}$.

### 5.1. Stochastic Probing
Each seed $s_i$ attempts to minimise its individual surprise relative to the global forest $\mathcal{F}$:
$$\min_{s_i} -\log P(x^* \mid s_i)$$

### 5.2. Fluid Integration (N-Dimensional Branching)
Rather than a "One Up, Three Down" rule, the framework supports N-Dimensional Branching:
* **Integration (Up):** A stable Meta-Pattern can be instanced as an $L_1$ leaf in any number of higher-order polygraphs. A single "Atom" node might be an $L_1$ leaf for multiple parent polygraphs simultaneously.
* **Differentiation (Down):** When a query hits a node, the framework "Unpacks" the polygraph into arbitrary directions of structural detail.

### 5.3. Vertical Migration (The Up/Down Shift)
The level of a node is determined by its Fractal Integration Index ($\phi$):
$$\phi_h = \frac{\text{Mutual Information}(h, \text{Parent})}{\text{Entropy}(h)}$$
* **If $\phi \to 1$ (Sinking):** The node becomes redundant as a standalone and is absorbed as an $L_1$ leaf into a parent's polygraph.
* **If $\phi \to 0$ (Rising):** The node lacks a parent that can explain it. It begins to cluster sub-nodes, increasing its internal complexity to become a new $L_5$ Meta-Pattern.

## 6. Competitive Dynamics
Conflict between indifferent nodes is resolved via a Lotka-Volterra Competition Model within the latent space:
$$\frac{dw_h}{dt} = \alpha w_h (S_h - \bar{S}) - w_h \sum_{j \in \mathcal{F}} \kappa_{hj} w_j$$
* $w_h$: The weight/influence of the node.
* $S_h$: The Total Score, defined as $S_h = \text{Accuracy} - \lambda(\text{Complexity})$.
* $\kappa_{hj}$: The Incompatibility Coefficient, calculated as the spatial overlap of their Gaussians:
$$\kappa_{hj} = \int \mathcal{G}_h(z) \mathcal{G}_j(z) dz$$

## 7. The Mathematical "Singularity"
The HPM Framework treats the Entire AI as a single, massive Root HPM Node.
* **The AI's Meta-Pattern:** The current "World Model."
* **The AI's Polygraph:** Every node created (Atoms, Logic, Concepts).
* **The AI's Leaves:** The raw sensory stream.

Intelligence is the Dynamic Resolution of this Root Node. The system stays "blurry" ($L_5$) to save energy and only "wakes up" (expands to $L_1$) where the world presents a contradiction.

### Summary of Observer-Dependent Levels
The "Level" of a node is effectively its Recursive Depth ($d$) during a specific query path:
* **Level 1 ($d=0$):** The node is a leaf. Its internal polygraph is ignored.
* **Level 3 ($d=n$):** The node is an intermediate relationship in an active expansion.
* **Level 5 ($d_{max}$):** The node is the root Meta-Pattern of the current observer's focus.

**Implementation Conclusion:**
In this geometry, the "Atom" doesn't know it's an atom. It is a Gaussian manifold that satisfies a set of $L_1$ constraints. If you query it, it reveals a polygraph of sub-nodes. If you don't, it remains a single point in the latent space. This ensures $O(\log N)$ search complexity and infinite recursive depth without increased computational overhead unless the observer demands precision.