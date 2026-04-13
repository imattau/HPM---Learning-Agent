
import tempfile
import numpy as np
from pathlib import Path
import pickle
from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest
from hfn.probabilistic_models import FlatGaussianModel, GaussianMixtureModel

def test_flat_gaussian_serialisation():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, max_hot=1)
        mu = np.array([1., 2., 3., 4.])
        sigma = np.array([0.1, 0.2, 0.3, 0.4])
        node = HFN(mu=mu, sigma=sigma, id="n1", use_diag=True)
        
        tf.register(node)
        # Add another node to evict n1 to cold
        tf.register(HFN(mu=np.zeros(4), sigma=np.ones(4), id="n2"))
        
        assert tf.cold_count() == 1
        path = Path(td) / "n1.npz"
        assert path.exists()
        
        # Load back
        loaded = tf.get("n1")
        assert loaded is not None
        assert isinstance(loaded.prob_model, FlatGaussianModel)
        assert np.allclose(loaded.prob_model.mu, mu)
        assert np.allclose(loaded.prob_model.sigma, sigma)

def test_gmm_serialisation():
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td, max_hot=1)
        
        comp1 = FlatGaussianModel(mu=np.array([1., 0., 0., 0.]), sigma=np.ones(4)*0.1)
        comp2 = FlatGaussianModel(mu=np.array([0., 1., 0., 0.]), sigma=np.ones(4)*0.2)
        mixture = GaussianMixtureModel(components=[comp1, comp2], weights=[0.3, 0.7])
        
        node = HFN(mu=np.array([0.5, 0.5, 0., 0.]), sigma=np.ones(4), prob_model=mixture, id="gmm_node")
        
        tf.register(node)
        tf.register(HFN(mu=np.zeros(4), sigma=np.ones(4), id="n2")) # evict
        
        assert tf.cold_count() == 1
        
        # Load back
        loaded = tf.get("gmm_node")
        assert loaded is not None
        assert isinstance(loaded.prob_model, GaussianMixtureModel)
        assert len(loaded.prob_model.components) == 2
        assert np.allclose(loaded.prob_model.weights, [0.3, 0.7])
        assert np.allclose(loaded.prob_model.components[0].mu, [1., 0., 0., 0.])
        assert np.allclose(loaded.prob_model.components[1].sigma, np.ones(4)*0.2)

def test_legacy_load_compatibility():
    """Verify that a node saved WITHOUT model_state loads as FlatGaussianModel."""
    with tempfile.TemporaryDirectory() as td:
        tf = TieredForest(D=4, forest_id="test", cold_dir=td)
        
        # Manually create a legacy .npz file (no model_state)
        path = Path(td) / "legacy.npz"
        mu = np.array([1., 1., 1., 1.])
        sigma = np.array([0.5, 0.5, 0.5, 0.5])
        np.savez_compressed(
            path,
            mu=mu,
            sigma=sigma,
            use_diag=True,
            child_ids_str=np.array("")
        )
        
        # Add to mu_index so tf.get() knows it exists
        tf._mu_index["legacy"] = mu
        if hasattr(tf, "_protected_ids"):
            tf._protected_ids.discard("legacy")
        
        loaded = tf.get("legacy")
        assert loaded is not None
        assert isinstance(loaded.prob_model, FlatGaussianModel)
        assert np.allclose(loaded.mu, mu)
        assert np.allclose(loaded.sigma, sigma)
