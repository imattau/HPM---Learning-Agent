from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Optional, Any

@dataclass
class Distribution:
    """Base class for probability distributions."""
    def log_prob(self, x: np.ndarray) -> float:
        raise NotImplementedError
    
    def entropy(self) -> float:
        raise NotImplementedError
        
    def hellinger_distance(self, other: Distribution) -> float:
        raise NotImplementedError

@dataclass
class GaussianDistribution(Distribution):
    """Multivariate Gaussian for continuous data."""
    mu: np.ndarray
    sigma: np.ndarray  # Covariance matrix
    
    def log_prob(self, x: np.ndarray) -> float:
        # Standard multivariate normal log-likelihood
        k = len(self.mu)
        sign, logdet = np.linalg.slogdet(self.sigma)
        if sign <= 0:
            return -1e10 # Numerical stability
        
        diff = x - self.mu
        inv_sigma = np.linalg.inv(self.sigma)
        exponent = -0.5 * diff.T @ inv_sigma @ diff
        normalization = -0.5 * (k * np.log(2 * np.pi) + logdet)
        return exponent + normalization
    
    def entropy(self) -> float:
        k = len(self.mu)
        _, logdet = np.linalg.slogdet(self.sigma)
        return 0.5 * (k * (1 + np.log(2 * np.pi)) + logdet)
        
    def hellinger_distance(self, other: GaussianDistribution) -> float:
        """
        H^2(P, Q) = 1 - (det(sigma_p)^1/4 * det(sigma_q)^1/4 / det((sigma_p+sigma_q)/2)^1/2) * 
                    exp(-1/8 * (mu_p - mu_q)^T * ((sigma_p+sigma_q)/2)^-1 * (mu_p - mu_q))
        """
        sigma_avg = (self.sigma + other.sigma) / 2
        inv_sigma_avg = np.linalg.inv(sigma_avg)
        diff = self.mu - other.mu
        
        _, logdet_p = np.linalg.slogdet(self.sigma)
        _, logdet_q = np.linalg.slogdet(other.sigma)
        _, logdet_avg = np.linalg.slogdet(sigma_avg)
        
        # Using logs for numerical stability
        log_num = 0.25 * logdet_p + 0.25 * logdet_q
        log_den = 0.5 * logdet_avg
        
        exponent = -0.125 * diff.T @ inv_sigma_avg @ diff
        bc = np.exp(log_num - log_den + exponent) # Bhattacharyya coefficient
        
        return np.sqrt(max(0, 1 - bc))

@dataclass
class CategoricalDistribution(Distribution):
    """Discrete categorical distribution."""
    probs: np.ndarray
    
    def __post_init__(self):
        # Normalize probs
        total = np.sum(self.probs)
        if total > 0:
            self.probs = self.probs / total
        else:
            self.probs = np.ones_like(self.probs) / len(self.probs)
    
    def log_prob(self, x: int) -> float:
        return np.log(self.probs[x] + 1e-12)
    
    def entropy(self) -> float:
        return -np.sum(self.probs * np.log(self.probs + 1e-12))
        
    def hellinger_distance(self, other: CategoricalDistribution) -> float:
        """
        H(P, Q) = 1/sqrt(2) * || sqrt(P) - sqrt(Q) ||_2
        """
        return (1 / np.sqrt(2)) * np.linalg.norm(np.sqrt(self.probs) - np.sqrt(other.probs))
