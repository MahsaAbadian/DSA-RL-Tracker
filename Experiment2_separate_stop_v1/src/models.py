#!/usr/bin/env python3
"""
Shared model architectures for DSA RL Experiment 2.
- SharedBackboneActorCritic: Full model with shared backbone + separate actor head (8 actions) + stop head (binary)
- ActorOnlyWithStop: Lightweight model for inference (Actor + Stop heads, no Critic)
"""
import torch
import torch.nn as nn

# ---------- HELPERS ----------
def gn(c): 
    """GroupNorm helper"""
    return nn.GroupNorm(4, c)

# ---------- FULL MODEL (TRAINING) ----------
class SharedBackboneActorCritic(nn.Module):
    """
    Full Actor-Critic model with shared backbone and separate stop head.
    
    Architecture:
    - Shared backbone: CNN + LSTM → 128 features
    - Actor head: 128 → 8 logits (movement actions only, 0-7)
    - Stop head: 128 → 1 logit (binary: stop or continue)
    - Critic head: Separate CNN + LSTM → 128 features → 1 value
    """
    def __init__(self, n_movement_actions=8, K=8):
        super().__init__()
        self.n_movement_actions = n_movement_actions
        self.K = K
        
        # Shared backbone: CNN + LSTM
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.shared_lstm = nn.LSTM(input_size=n_movement_actions, hidden_size=64, batch_first=True)
        
        # Actor head: Movement actions only (0-7)
        self.actor_head = nn.Sequential(
            nn.Linear(128, 128), 
            nn.PReLU(), 
            nn.Linear(128, n_movement_actions)
        )
        
        # Stop head: Binary classification (stop or continue)
        self.stop_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.Linear(128, 1)  # Single logit for binary classification
        )

        # Critic CNN (Privileged info - uses GT map, separate from shared backbone)
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.critic_lstm = nn.LSTM(input_size=n_movement_actions, hidden_size=64, batch_first=True)
        self.critic_head = nn.Sequential(nn.Linear(128, 128), nn.PReLU(), nn.Linear(128, 1))
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=torch.sqrt(torch.tensor(2.0)))
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)

    def forward(self, actor_obs, critic_gt, ahist_onehot, hc_shared=None, hc_critic=None):
        """
        Forward pass for training.
        
        Args:
            actor_obs: Actor observation (B, 4, 33, 33)
            critic_gt: Ground truth map for critic (B, 1, 33, 33)
            ahist_onehot: Action history one-hot (B, K, n_movement_actions) - only movement actions!
            hc_shared: Shared LSTM hidden state (optional)
            hc_critic: Critic LSTM hidden state (optional)
        
        Returns:
            movement_logits: Movement action logits (B, n_movement_actions) - actions 0-7
            stop_logit: Stop logit (B,) - single value, use sigmoid for probability
            value: State value estimate (B,)
            hc_shared: Updated shared LSTM hidden state
            hc_critic: Updated critic LSTM hidden state
        """
        # Shared backbone
        feat_shared = self.shared_cnn(actor_obs).flatten(1)  # (B, 64)
        lstm_shared, hc_shared = self.shared_lstm(ahist_onehot, hc_shared)  # (B, K, 64)
        joint_features = torch.cat([feat_shared, lstm_shared[:, -1, :]], dim=1)  # (B, 128)
        
        # Actor head: Movement actions only
        movement_logits = self.actor_head(joint_features)  # (B, 8)
        
        # Stop head: Binary classification
        stop_logit = self.stop_head(joint_features)  # (B, 1)

        # Critic (separate from shared backbone)
        critic_input = torch.cat([actor_obs, critic_gt], dim=1)
        feat_c = self.critic_cnn(critic_input).flatten(1)
        lstm_c, hc_critic = self.critic_lstm(ahist_onehot, hc_critic)
        joint_c = torch.cat([feat_c, lstm_c[:, -1, :]], dim=1)
        value = self.critic_head(joint_c).squeeze(-1)
        
        return movement_logits, stop_logit, value, hc_shared, hc_critic

# ---------- ACTOR+STOP MODEL (INFERENCE) ----------
class ActorOnlyWithStop(nn.Module):
    """
    Lightweight Actor+Stop model for inference.
    Only contains the policy network (movement + stop), no critic.
    """
    def __init__(self, n_movement_actions=8, K=8):
        super().__init__()
        self.n_movement_actions = n_movement_actions
        self.K = K
        
        # Shared backbone (same as training model)
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.shared_lstm = nn.LSTM(input_size=n_movement_actions, hidden_size=64, batch_first=True)
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(128, 128), 
            nn.PReLU(), 
            nn.Linear(128, n_movement_actions)
        )
        
        # Stop head
        self.stop_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, actor_obs, ahist_onehot, hc_shared=None):
        """
        Forward pass for inference.
        
        Args:
            actor_obs: Actor observation (B, 4, 33, 33)
            ahist_onehot: Action history one-hot (B, K, n_movement_actions)
            hc_shared: Shared LSTM hidden state (optional)
        
        Returns:
            movement_logits: Movement action logits (B, n_movement_actions)
            stop_logit: Stop logit (B,)
            hc_shared: Updated shared LSTM hidden state
        """
        # Shared backbone
        feat_shared = self.shared_cnn(actor_obs).flatten(1)
        lstm_shared, hc_shared = self.shared_lstm(ahist_onehot, hc_shared)
        joint_features = torch.cat([feat_shared, lstm_shared[:, -1, :]], dim=1)
        
        # Actor head
        movement_logits = self.actor_head(joint_features)
        
        # Stop head
        stop_logit = self.stop_head(joint_features)  # (B, 1)
        
        return movement_logits, stop_logit, hc_shared

