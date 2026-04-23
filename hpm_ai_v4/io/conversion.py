import numpy as np
import json
from hpm_ai_v4.pattern import HierarchicalPattern

class PatternSerializer:
    """Save and load patterns to JSON-ready dictionaries."""
    @staticmethod
    def to_dict(pattern):
        data = {
            'id': pattern.id,
            'complexity': pattern.complexity,
            'running_loss': float(pattern.running_loss),
            'weight': float(pattern.weight),
            'latent_dim': pattern.latent_dim,
            'obs_dim': pattern.obs_dim
        }
        
        # All patterns now have B matrix
        data.update({
            'B': pattern.B.tolist()
        })
        
        if pattern.complexity >= 2:
            data.update({
                'A3': pattern.A3.tolist(),
                'A32': pattern.A32.tolist(),
                'A21': pattern.A21.tolist(),
                'pi3': pattern.pi3.tolist()
            })
            
        return data

    @staticmethod
    def from_dict(data):
        p_id = data.get('id', 0)
        complexity = data.get('complexity', 3)
        obs_dim = data.get('obs_dim', 2)
        
        if complexity >= 2:
            p = HierarchicalPattern(p_id, latent_dim=data.get('latent_dim', 2), obs_dim=obs_dim)
            p.A3 = np.array(data['A3'])
            p.A32 = np.array(data['A32'])
            p.A21 = np.array(data['A21'])
            p.pi3 = np.array(data['pi3'])
        else:
            p = HierarchicalPattern.flat(p_id, obs_dim=obs_dim)
            
        p.B = np.array(data['B'])
        p.running_loss = data['running_loss']
        p.weight = data['weight']
        p.complexity = complexity
        return p

class ObservationScaler:
    """Normalize or discretize continuous inputs."""
    @staticmethod
    def standardise(sequence):
        seq = np.array(sequence)
        mean = np.mean(seq)
        std = np.std(seq)
        return (seq - mean) / (std + 1e-12)

    @staticmethod
    def discrete_by_quantiles(sequence, num_bins=10):
        seq = np.array(sequence)
        bins = np.percentile(seq, np.linspace(0, 100, num_bins + 1))
        return (np.digitize(seq, bins) - 1).tolist()
