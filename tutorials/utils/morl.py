"""
Multi-Objective RL utilities for gradient conflict resolution and multi-objective optimization.
Implements CAGrad, PCGrad, MGDA, and other MORL algorithms.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional, Callable, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten_gradients(parameters) -> torch.Tensor:
    """
    Flatten all parameter gradients into a single vector.
    
    Parameters
    ----------
    parameters : iterable of nn.Parameter
        Model parameters with gradients
    
    Returns
    -------
    torch.Tensor
        Flattened gradient vector
    """
    grads = []
    for p in parameters:
        if p.grad is None:
            grads.append(torch.zeros_like(p).view(-1))
        else:
            grads.append(p.grad.view(-1))
    return torch.cat(grads)


def unflatten_gradients(parameters, flat_grad: torch.Tensor) -> None:
    """
    Assign flat gradient vector back to parameter gradients.
    
    Parameters
    ----------
    parameters : iterable of nn.Parameter
        Model parameters to assign gradients to
    flat_grad : torch.Tensor
        Flattened gradient vector
    """
    offset = 0
    for p in parameters:
        numel = p.numel()
        grad_slice = flat_grad[offset:offset + numel].view_as(p)
        if p.grad is None:
            p.grad = grad_slice.clone()
        else:
            p.grad.copy_(grad_slice)
        offset += numel


class GradientConflictResolver:
    """Collection of gradient conflict resolution algorithms for MORL."""
    
    @staticmethod
    def cagrad(
        gradients: List[torch.Tensor],
        c: float = 0.5,
        return_coefficients: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, np.ndarray]:
        """
        Conflict-Averse Gradient (CAGrad) algorithm.
        
        Optimizes: g* = g0 + λ * gw
        where g0 is average gradient, gw is weighted combination,
        and λ balances between average direction and conflict resolution.
        
        Parameters
        ----------
        gradients : List[torch.Tensor]
            List of gradient vectors (one per objective)
        c : float
            Trade-off parameter controlling conservativeness (0 to 1)
        return_coefficients : bool
            If True, also return the weighting coefficients
        
        Returns
        -------
        torch.Tensor or (torch.Tensor, np.ndarray)
            Combined gradient, optionally with coefficients
        """
        n = len(gradients)
        if n == 1:
            return gradients[0] if not return_coefficients else (gradients[0], np.array([1.0]))
        
        # Average gradient
        g0 = sum(gradients) / n
        
        # Compute gradient inner products matrix
        GG = torch.zeros(n, n, device=gradients[0].device)
        for i in range(n):
            for j in range(n):
                GG[i, j] = torch.dot(gradients[i], gradients[j])
        
        g0_norm = torch.norm(g0)
        
        # Optimization for two objectives (analytical solution)
        if n == 2:
            g11, g12, g22 = GG[0, 0], GG[0, 1], GG[1, 1]
            
            # Grid search for optimal x in [0, 1]
            best_x = 0.5
            best_obj = float('inf')
            
            for x in np.linspace(0, 1, 21):
                x = float(x)
                gw_norm_sq = x**2 * g11 + (1-x)**2 * g22 + 2*x*(1-x)*g12
                obj = c * math.sqrt(max(gw_norm_sq.item(), 1e-8)) + \
                      0.5 * x * (g11 + g22 - 2*g12).item() + \
                      (0.5 + x) * (g12 - g22).item() + g22.item()
                
                if obj < best_obj:
                    best_obj = obj
                    best_x = x
            
            # Compute final gradient
            gw = best_x * gradients[0] + (1 - best_x) * gradients[1]
            gw_norm = torch.norm(gw)
            
            if gw_norm > 1e-8:
                lam = c * g0_norm / gw_norm
            else:
                lam = 0.0
            
            g_star = g0 + lam * gw
            g_final = g_star / (1.0 + c)
            
            if return_coefficients:
                return g_final, np.array([best_x, 1 - best_x])
            return g_final
        
        # Multi-objective case (n > 2): use Frank-Wolfe algorithm
        else:
            alpha = np.ones(n) / n  # Initialize with uniform weights
            
            for _ in range(50):  # Frank-Wolfe iterations
                # Compute gradient w.r.t. alpha
                Gg = GG @ torch.tensor(alpha, device=GG.device)
                grad_alpha = Gg.cpu().numpy()
                
                # Find descent direction (vertex of simplex)
                idx_min = np.argmin(grad_alpha)
                s = np.zeros(n)
                s[idx_min] = 1.0
                
                # Line search
                d = s - alpha
                step_size = 2.0 / (_ + 2)  # Standard Frank-Wolfe step size
                alpha = alpha + step_size * d
            
            # Compute weighted gradient
            gw = sum(a * g for a, g in zip(alpha, gradients))
            gw_norm = torch.norm(gw)
            
            if gw_norm > 1e-8:
                lam = c * g0_norm / gw_norm
            else:
                lam = 0.0
            
            g_star = g0 + lam * gw
            g_final = g_star / (1.0 + c)
            
            if return_coefficients:
                return g_final, alpha
            return g_final
    
    @staticmethod
    def pcgrad(gradients: List[torch.Tensor]) -> torch.Tensor:
        """
        Projecting Conflicting Gradients (PCGrad) algorithm.
        
        Projects conflicting gradients to reduce interference.
        When two gradients have negative cosine similarity,
        projects one onto the normal plane of the other.
        
        Parameters
        ----------
        gradients : List[torch.Tensor]
            List of gradient vectors (one per objective)
        
        Returns
        -------
        torch.Tensor
            Combined gradient with conflicts resolved
        """
        n = len(gradients)
        if n == 1:
            return gradients[0]
        
        # Random order for symmetry
        indices = torch.randperm(n)
        grad_sum = gradients[indices[0]].clone()
        
        for i in range(1, n):
            g_i = gradients[indices[i]]
            
            # Check for conflict (negative cosine similarity)
            dot_prod = torch.dot(grad_sum, g_i)
            
            if dot_prod < 0:
                # Project g_i onto the normal plane of grad_sum
                grad_sum_norm_sq = torch.dot(grad_sum, grad_sum)
                if grad_sum_norm_sq > 1e-8:
                    proj = dot_prod / grad_sum_norm_sq
                    g_i = g_i - proj * grad_sum
            
            grad_sum = grad_sum + g_i
        
        return grad_sum / n
    
    @staticmethod
    def mgda(
        gradients: List[torch.Tensor],
        normalize: bool = True
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Multiple Gradient Descent Algorithm (MGDA).
        
        Finds the minimum-norm point in the convex hull of gradients.
        This provides a Pareto-optimal descent direction.
        
        Parameters
        ----------
        gradients : List[torch.Tensor]
            List of gradient vectors (one per objective)
        normalize : bool
            Whether to normalize gradients before combination
        
        Returns
        -------
        torch.Tensor, np.ndarray
            Combined gradient and weight coefficients
        """
        n = len(gradients)
        if n == 1:
            return gradients[0], np.array([1.0])
        
        # Optionally normalize gradients
        if normalize:
            grads_normalized = []
            for g in gradients:
                g_norm = torch.norm(g)
                if g_norm > 1e-8:
                    grads_normalized.append(g / g_norm)
                else:
                    grads_normalized.append(g)
            gradients = grads_normalized
        
        # Compute gradient matrix G where G[i,j] = <g_i, g_j>
        G = torch.zeros(n, n, device=gradients[0].device)
        for i in range(n):
            for j in range(n):
                G[i, j] = torch.dot(gradients[i], gradients[j])
        
        # Use Frank-Wolfe to solve the QP: min_α ||Σ α_i g_i||^2
        alpha = np.ones(n) / n
        
        for iteration in range(100):
            # Compute gradient: ∇f(α) = 2Gα
            grad = 2 * (G @ torch.tensor(alpha, device=G.device)).cpu().numpy()
            
            # Find Frank-Wolfe vertex (minimum gradient component)
            idx_min = np.argmin(grad)
            
            # Optimal step size via line search
            delta = np.zeros(n)
            delta[idx_min] = 1.0
            delta = delta - alpha
            
            # Compute optimal step size analytically
            numerator = -np.dot(grad, delta)
            denominator = 2 * np.dot(delta, G.cpu().numpy() @ delta)
            
            if denominator > 1e-8:
                step_size = np.clip(numerator / denominator, 0, 1)
            else:
                step_size = 0
            
            # Update alpha
            alpha_new = alpha + step_size * delta
            
            # Check convergence
            if np.linalg.norm(alpha_new - alpha) < 1e-6:
                break
            
            alpha = alpha_new
        
        # Compute final weighted gradient
        g_mgda = sum(a * g for a, g in zip(alpha, gradients))
        
        return g_mgda, alpha
    
    @staticmethod
    def imtl_g(gradients: List[torch.Tensor]) -> torch.Tensor:
        """
        Impartial Multi-Task Learning (IMTL-G) gradient balancing.
        
        Scales gradients to have equal magnitudes before averaging,
        ensuring no single objective dominates due to gradient scale.
        
        Parameters
        ----------
        gradients : List[torch.Tensor]
            List of gradient vectors (one per objective)
        
        Returns
        -------
        torch.Tensor
            Balanced gradient
        """
        n = len(gradients)
        if n == 1:
            return gradients[0]
        
        # Compute gradient norms
        norms = [torch.norm(g) for g in gradients]
        mean_norm = sum(norms) / n
        
        # Scale gradients to have mean norm
        balanced_grads = []
        for g, norm in zip(gradients, norms):
            if norm > 1e-8:
                balanced_grads.append(g * (mean_norm / norm))
            else:
                balanced_grads.append(g)
        
        return sum(balanced_grads) / n


def compute_surrogate_gradients(
    actor_critic: nn.Module,
    batch_data: Dict[str, torch.Tensor],
    objective: str = 'reward',
    clip_epsilon: float = 0.2,
    compute_gradient: bool = True
) -> torch.Tensor | torch.Tensor:
    """
    Compute policy gradient for a specific objective.
    
    Parameters
    ----------
    actor_critic : nn.Module
        Actor-critic model
    batch_data : Dict[str, torch.Tensor]
        Batch data containing observations, actions, advantages, etc.
    objective : str
        Which objective to optimize ('reward', 'cost', 'entropy')
    clip_epsilon : float
        PPO clipping parameter
    compute_gradient : bool
        If True, compute actual gradient; if False, return loss
    
    Returns
    -------
    torch.Tensor
        Gradient vector or loss value
    """
    obs = batch_data['obs']
    actions = batch_data['actions']
    old_logprobs = batch_data['logprobs']
    
    # Select advantages based on objective
    if objective == 'reward':
        advantages = batch_data['advantages']
    elif objective == 'cost':
        advantages = -batch_data['cost_advantages']  # Minimize cost
    elif objective == 'entropy':
        # For entropy objective, we want to maximize entropy
        if hasattr(actor_critic, 'evaluate_actions'):
            _, entropy, _, _ = actor_critic.evaluate_actions(obs, actions)
        else:
            dist, _, _ = actor_critic(obs)
            entropy = dist.entropy().sum(-1)
        return entropy.mean() if not compute_gradient else torch.autograd.grad(entropy.mean(), actor_critic.parameters())
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Compute log probabilities for current policy
    if hasattr(actor_critic, 'evaluate_actions'):
        log_probs, _, _, _ = actor_critic.evaluate_actions(obs, actions)
    else:
        dist, _, _ = actor_critic(obs)
        log_probs = dist.log_prob(actions).sum(-1)
    
    # PPO surrogate loss
    ratio = torch.exp(log_probs - old_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    loss = -torch.min(surr1, surr2).mean()
    
    if compute_gradient:
        # Compute gradient
        actor_critic.zero_grad()
        loss.backward(retain_graph=True)
        grad = flatten_gradients(actor_critic.parameters()).detach()
        actor_critic.zero_grad()
        return grad
    else:
        return loss


class MultiObjectiveWrapper:
    """
    Wrapper to handle multiple objectives in RL training.
    Tracks per-objective performance and provides utilities for MO optimization.
    """
    
    def __init__(
        self,
        objectives: List[str],
        weights: Optional[List[float]] = None,
        combination_method: str = 'weighted_sum',
        conflict_resolver: Optional[str] = None
    ):
        """
        Parameters
        ----------
        objectives : List[str]
            List of objective names (e.g., ['reward', 'cost', 'fairness'])
        weights : Optional[List[float]]
            Initial weights for objectives (defaults to uniform)
        combination_method : str
            How to combine objectives ('weighted_sum', 'chebyshev', 'priority')
        conflict_resolver : Optional[str]
            Gradient conflict resolution method ('cagrad', 'pcgrad', 'mgda', None)
        """
        self.objectives = objectives
        self.n_objectives = len(objectives)
        
        if weights is None:
            self.weights = np.ones(self.n_objectives) / self.n_objectives
        else:
            self.weights = np.array(weights) / np.sum(weights)
        
        self.combination_method = combination_method
        self.conflict_resolver = conflict_resolver
        
        # Performance tracking
        self.objective_history = {obj: [] for obj in objectives}
        self.weight_history = []
    
    def combine_losses(
        self,
        losses: Dict[str, torch.Tensor],
        return_weighted: bool = True
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Combine multiple objective losses.
        
        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            Dictionary of losses for each objective
        return_weighted : bool
            If True, return single combined loss; else return weighted dict
        
        Returns
        -------
        torch.Tensor or Dict[str, torch.Tensor]
            Combined loss or weighted losses
        """
        if self.combination_method == 'weighted_sum':
            if return_weighted:
                total_loss = sum(
                    self.weights[i] * losses[obj]
                    for i, obj in enumerate(self.objectives)
                    if obj in losses
                )
                return total_loss
            else:
                return {
                    obj: self.weights[i] * losses[obj]
                    for i, obj in enumerate(self.objectives)
                    if obj in losses
                }
        
        elif self.combination_method == 'chebyshev':
            # Chebyshev scalarization: min max_i w_i * |f_i - z_i*|
            # Requires reference point z* (best value for each objective)
            weighted_diffs = [
                self.weights[i] * torch.abs(losses[obj])
                for i, obj in enumerate(self.objectives)
                if obj in losses
            ]
            return torch.max(torch.stack(weighted_diffs))
        
        elif self.combination_method == 'priority':
            # Lexicographic priority: optimize objectives in order of importance
            # For training, use weighted sum with exponentially decreasing weights
            priority_weights = np.array([10 ** (-i) for i in range(self.n_objectives)])
            priority_weights /= priority_weights.sum()
            
            total_loss = sum(
                priority_weights[i] * losses[obj]
                for i, obj in enumerate(self.objectives)
                if obj in losses
            )
            return total_loss
        
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
    
    def resolve_gradient_conflicts(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Resolve conflicts between objective gradients.
        
        Parameters
        ----------
        gradients : Dict[str, torch.Tensor]
            Dictionary of gradients for each objective
        
        Returns
        -------
        torch.Tensor
            Combined gradient with conflicts resolved
        """
        grad_list = [gradients[obj] for obj in self.objectives if obj in gradients]
        
        if self.conflict_resolver == 'cagrad':
            return GradientConflictResolver.cagrad(grad_list)
        elif self.conflict_resolver == 'pcgrad':
            return GradientConflictResolver.pcgrad(grad_list)
        elif self.conflict_resolver == 'mgda':
            combined, _ = GradientConflictResolver.mgda(grad_list)
            return combined
        elif self.conflict_resolver == 'imtl_g':
            return GradientConflictResolver.imtl_g(grad_list)
        else:
            # No conflict resolution, use weighted sum
            return sum(
                self.weights[i] * grad
                for i, grad in enumerate(grad_list)
            )
    
    def update_weights(
        self,
        performance: Dict[str, float],
        method: str = 'performance_based'
    ):
        """
        Dynamically update objective weights based on performance.
        
        Parameters
        ----------
        performance : Dict[str, float]
            Current performance for each objective
        method : str
            Weight update method ('performance_based', 'random', 'cyclic')
        """
        if method == 'performance_based':
            # Increase weight for underperforming objectives
            # Assumes higher values are better
            perfs = np.array([performance.get(obj, 0) for obj in self.objectives])
            
            # Normalize performances
            if perfs.std() > 1e-8:
                perfs_normalized = (perfs - perfs.mean()) / perfs.std()
            else:
                perfs_normalized = perfs
            
            # Update weights inversely proportional to performance
            new_weights = np.exp(-perfs_normalized)
            self.weights = new_weights / new_weights.sum()
        
        elif method == 'random':
            # Random weight vector from Dirichlet distribution
            self.weights = np.random.dirichlet(np.ones(self.n_objectives))
        
        elif method == 'cyclic':
            # Cycle through objectives
            current_max_idx = np.argmax(self.weights)
            next_idx = (current_max_idx + 1) % self.n_objectives
            self.weights = np.zeros(self.n_objectives)
            self.weights[next_idx] = 1.0
        
        self.weight_history.append(self.weights.copy())
    
    def log_performance(self, performance: Dict[str, float]):
        """Log performance for each objective."""
        for obj in self.objectives:
            if obj in performance:
                self.objective_history[obj].append(performance[obj])
