"""PhysicsAgent: Physics-specific reasoning and computation agent."""
from __future__ import annotations
import re
import numpy as np
from typing import List, Optional, Any, Tuple
from hfn.hfn import HFN
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.domains.physics_domain import PhysicsDomainConfig

class PhysicsAgent(BaseHFNAgent):
    """
    Agent specializing in physics word problems and scientific laws.
    """
    def __init__(self, config: PhysicsDomainConfig, forest=None, **kwargs) -> None:
        super().__init__(config, forest=forest, **kwargs)
        self._register_physics_primitives()

    def _register_physics_primitives(self) -> None:
        """Register L1 primitives for physics calculations."""
        self.add_strategy("calc_force", self.primitive_calc_force)
        self.add_strategy("calc_vel", self.primitive_calc_velocity)
        self.add_strategy("calc_accel", self.primitive_calc_acceleration)

    # --- Primitives (L1) ---

    def primitive_calc_force(self, inputs: List[Any], outputs: List[Any]) -> Optional[List[HFN]]:
        """Primitive: F = m * a"""
        if len(inputs) < 2: return None
        m, a = inputs[0], inputs[1]
        try:
            m_val = float(m.metadata.get("value", m)) if isinstance(m, HFN) else float(m)
            a_val = float(a.metadata.get("value", a)) if isinstance(a, HFN) else float(a)
            f_val = m_val * a_val
            return [self._create_value_node(f_val, "force", "N")]
        except Exception: return None

    def primitive_calc_velocity(self, inputs: List[Any], outputs: List[Any]) -> Optional[List[HFN]]:
        """Primitive: v = d / t"""
        if len(inputs) < 2: return None
        d, t = inputs[0], inputs[1]
        try:
            d_val = float(d.metadata.get("value", d)) if isinstance(d, HFN) else float(d)
            t_val = float(t.metadata.get("value", t)) if isinstance(t, HFN) else float(t)
            v_val = d_val / t_val
            return [self._create_value_node(v_val, "velocity", "m/s")]
        except Exception: return None

    def primitive_calc_acceleration(self, inputs: List[Any], outputs: List[Any]) -> Optional[List[HFN]]:
        """Primitive: a = F / m"""
        if len(inputs) < 2: return None
        f, m = inputs[0], inputs[1]
        try:
            f_val = float(f.metadata.get("value", f)) if isinstance(f, HFN) else float(f)
            m_val = float(m.metadata.get("value", m)) if isinstance(m, HFN) else float(m)
            a_val = f_val / m_val
            return [self._create_value_node(a_val, "acceleration", "m/s^2")]
        except Exception: return None

    def _create_value_node(self, val: float, type_name: str, unit: str) -> HFN:
        node_id = f"physics_{type_name}_{val}"
        mu = np.zeros(self.m_dim)
        mu[0] = val / 100.0
        node = HFN(mu=mu, sigma=np.ones(self.m_dim)*0.1, id=node_id, use_diag=True)
        node.relation_type = f"physics_{type_name}"
        node.metadata = {"value": val, "unit": unit, "text": f"{val} {unit}"}
        self.observer.register(node)
        return node

    def detect_intent(self, query: str) -> Optional[Tuple[str, Dict[str, float]]]:
        """Detect intent and extract variables (force, mass, accel, etc.)"""
        q = query.lower()
        
        # 1. Newton's 2nd Law (Force)
        if "force" in q and "mass" in q and "acceleration" in q:
            # Simple heuristic extraction
            m = re.search(r"mass\s+of\s+([\d\.]+)", q)
            a = re.search(r"acceleration\s+of\s+([\d\.]+)", q)
            if m and a:
                return "calc_force", {"mass": float(m.group(1)), "accel": float(a.group(1))}
        
        # 2. Acceleration
        if "acceleration" in q and "force" in q and "mass" in q:
            f = re.search(r"force\s+of\s+([\d\.]+)", q)
            m = re.search(r"mass\s+of\s+([\d\.]+)", q)
            if f and m:
                return "calc_accel", {"force": float(f.group(1)), "mass": float(m.group(1))}
                
        return None

    def derive(self, intent: str, params: Dict[str, float]) -> Optional[HFN]:
        """Apply physics primitive to compute result."""
        if intent == "calc_force":
            res = self.primitive_calc_force([params["mass"], params["accel"]], [])
        elif intent == "calc_accel":
            res = self.primitive_calc_acceleration([params["force"], params["mass"]], [])
        else:
            return None
        return res[0] if res else None
