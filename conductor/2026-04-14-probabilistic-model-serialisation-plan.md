# Serialisation Design for Pluggable Probabilistic Models

## Objective
Extend the serialisation format of `TieredForest` to store arbitrary probabilistic models while maintaining full backward compatibility with existing saved data (which uses the old flat Gaussian format).

## Requirements
- Allow any `ProbabilisticModel` subclass to be saved and restored seamlessly.
- Preserve backward compatibility: old `.npz` files must load correctly as `FlatGaussianModel`s.
- Maintain efficiency and avoid duplicating large arrays where possible.

## Proposed Extended Format
Each `.npz` file created during `_evict_lru` will contain an additional entry:
- `model_state`: A serialized byte array (using `pickle.dumps` for simplicity in this pure Python environment) containing a dictionary with:
  - `model_type`: The registered string name of the probabilistic model class (e.g., `"flat_gaussian"`, `"gaussian_mixture"`).
  - `params`: A dictionary of parameters needed to reconstruct the specific model instance.

The existing arrays (`mu`, `sigma`, `use_diag`, `child_ids_str`) will remain in the `.npz` file for backward compatibility, ensuring older versions of the codebase can still attempt to read the core geometry.

## Implementation Steps

### 1. Extend `ProbabilisticModel` Interface
**File:** `hfn/probabilistic_models.py`
- Add two abstract methods to `ProbabilisticModel`:
  - `get_state(self) -> dict`: Returns a dictionary of parameters needed to reconstruct the model.
  - `@classmethod from_state(cls, state: dict) -> ProbabilisticModel`: Reconstructs the model from the state dictionary.
- Implement these methods for `FlatGaussianModel`:
  - `get_state`: Return `{"mu": self.mu, "sigma": self.sigma, "use_diag": self.use_diag}`.
  - `from_state`: Return `cls(state["mu"], state["sigma"], state["use_diag"])`.
- Implement these methods for `GaussianMixtureModel`:
  - `get_state`: Return `{"k": len(self.components), "weights": self.weights, "mus": [c.mu for c in self.components], "sigmas": [c.sigma for c in self.components], "use_diag": self.components[0].use_diag}`.
  - `from_state`: Reconstruct components using the lists and return `cls(components, weights)`.

### 2. Implement Model Registry
**File:** `hfn/probabilistic_models.py`
- Create a global `_MODEL_REGISTRY` dictionary.
- Add a `register_model(name: str, cls: type)` function.
- Add a `get_model_class(name: str) -> type` function.
- Register `"flat_gaussian"` and `"gaussian_mixture"`.

### 3. Update `TieredForest` Serialisation
**File:** `hfn/tiered_forest.py`
- **`_evict_lru`:**
  - Retrieve the model state using `node.prob_model.get_state()`.
  - Construct the `model_state` dictionary: `{"model_type": get_model_name(type(node.prob_model)), "params": node.prob_model.get_state()}`.
  - Serialize `model_state` using `pickle.dumps` and add it to the `np.savez_compressed` call under the key `model_state`.
- **`_load_from_cold`:**
  - Check if `"model_state"` exists in the loaded `.npz` data.
  - If it exists, deserialize it using `pickle.loads`.
  - Look up the model class via the registry using `model_state["model_type"]`.
  - Instantiate the model using `model_cls.from_state(model_state["params"])`.
  - Pass the instantiated model to the `HFN` constructor via the `prob_model` parameter.
  - If `"model_state"` does not exist (legacy data), fall back to creating a `FlatGaussianModel` using the stored `mu`, `sigma`, and `use_diag`.

## Verification
- Write tests in `tests/hfn/test_tiered_forest.py` to:
  - Save and load a standard node (Flat Gaussian).
  - Save and load a node equipped with a `GaussianMixtureModel`.
  - Verify that a legacy `.npz` file (simulated by saving without `model_state`) loads correctly as a Flat Gaussian.
