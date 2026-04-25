# hpm_ai_v4/tools/serializer.py
import pickle
import json
import numpy as np
from hpm_ai_v4.pattern import HierarchicalPattern

FIELDS = ['id', 'A', 'B', 'pi', 'latent_dim', 'obs_dim',
          'running_loss', 'weight', 'creation_step']
OPTIONAL = ['source_corpus', 'density_at_save', 'level']


class PatternSerializer:

    @staticmethod
    def _to_dict(p):
        d = {f: getattr(p, f) for f in FIELDS}
        for f in OPTIONAL:
            if hasattr(p, f):
                d[f] = getattr(p, f)
        return d

    @staticmethod
    def _from_dict(d):
        p = HierarchicalPattern(d['id'], latent_dim=d['latent_dim'], obs_dim=d['obs_dim'])
        p.A = np.array(d['A'], dtype=np.float32)
        p.B = np.array(d['B'], dtype=np.float32)
        p.pi = np.array(d['pi'], dtype=np.float32)
        p.running_loss = d['running_loss']
        p.weight = d['weight']
        p.creation_step = d.get('creation_step', 0)
        for f in OPTIONAL:
            if f in d:
                setattr(p, f, d[f])
        p._refresh_log_cache()
        return p

    @staticmethod
    def save(patterns, path):
        with open(path, 'wb') as f:
            pickle.dump([PatternSerializer._to_dict(p) for p in patterns], f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return [PatternSerializer._from_dict(d) for d in pickle.load(f)]

    @staticmethod
    def save_json(patterns, path):
        def convert(d):
            return {k: (v.tolist() if hasattr(v, 'tolist') else v)
                    for k, v in d.items()}
        with open(path, 'w') as f:
            json.dump([convert(PatternSerializer._to_dict(p)) for p in patterns], f)

    @staticmethod
    def load_json(path):
        with open(path) as f:
            return [PatternSerializer._from_dict(d) for d in json.load(f)]
