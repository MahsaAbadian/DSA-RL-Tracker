#!/usr/bin/env python3
"""
Model architectures for DSA RL Experiment 4 (v2).
- DecoupledStopBackboneActorCritic: Full model with SEPARATE backbones for Actor and Stop.
- DecoupledStopActorOnly: Inference model with separate backbones.
"""
import torch
import torch.nn as nn

# ---------- HELPERS ----------
def gn(c): 
    """GroupNorm helper"""
    return nn.GroupNorm(4, c)

# ---------- FULL MODEL (TRAINING) ----------
class DecoupledStopBackboneActorCritic(nn.Module):
    """
    Actor-Critic model where the Stop Head has its own dedicated vision backbone.
    This decouples the "where to go" (Actor) from the "is this the end" (Stop) tasks.
    """
    def __init__(self, n_movement_actions=8, K=8):
        super().__init__()
        self.n_movement_actions = n_movement_actions
        self.K = K
        
        # 1. Actor Backbone (Movement): CNN + LSTM
        # Takes 4 channels: [current_crop, prev1, prev2, path_mask]
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.actor_lstm = nn.LSTM(input_size=n_movement_actions, hidden_size=64, batch_first=True)
        self.actor_head = nn.Sequential(
            nn.Linear(128, 128), nn.PReLU(), 
            nn.Linear(128, n_movement_actions)
        )
        
        # 2. Stop Backbone (Dedicated Vision): Deepened with Dilation
        # Takes 2 channels: [current_crop, path_mask]
        self.stop_cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.stop_head = nn.Sequential(
            nn.Linear(128, 64), nn.PReLU(), # 64 (CNN) + 64 (LSTM) = 128
            nn.Linear(64, 1)
        )

        # 3. Critic CNN (Privileged info - uses GT map)
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

    def forward(self, actor_obs, critic_gt, ahist_onehot, hc_actor=None, hc_critic=None):
        # Actor/Movement Path
        feat_a = self.actor_cnn(actor_obs).flatten(1)
        lstm_a, hc_actor = self.actor_lstm(ahist_onehot, hc_actor)
        joint_a = torch.cat([feat_a, lstm_a[:, -1, :]], dim=1)
        movement_logits = self.actor_head(joint_a)
        
        # Stop Path (Improved)
        # Use Channel 0 (image) and Channel 3 (path mask)
        stop_input = torch.cat([actor_obs[:, 0:1, :, :], actor_obs[:, 3:4, :, :]], dim=1)
        feat_stop = self.stop_cnn(stop_input).flatten(1)
        # Combine vision with movement history
        joint_stop = torch.cat([feat_stop, lstm_a[:, -1, :]], dim=1)
        stop_logit = self.stop_head(joint_stop).squeeze(-1)

        # Critic Path
        critic_input = torch.cat([actor_obs, critic_gt], dim=1)
        feat_c = self.critic_cnn(critic_input).flatten(1)
        lstm_c, hc_critic = self.critic_lstm(ahist_onehot, hc_critic)
        joint_c = torch.cat([feat_c, lstm_c[:, -1, :]], dim=1)
        value = self.critic_head(joint_c).squeeze(-1)
        
        return movement_logits, stop_logit, value, hc_actor, hc_critic

# ---------- INFERENCE MODEL ----------
class DecoupledStopActorOnly(nn.Module):
    def __init__(self, n_movement_actions=8, K=8):
        super().__init__()
        self.n_movement_actions = n_movement_actions
        self.K = K
        
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.actor_lstm = nn.LSTM(input_size=n_movement_actions, hidden_size=64, batch_first=True)
        self.actor_head = nn.Sequential(
            nn.Linear(128, 128), nn.PReLU(), 
            nn.Linear(128, n_movement_actions)
        )
        
        self.stop_cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.stop_head = nn.Sequential(
            nn.Linear(128, 64), nn.PReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, actor_obs, ahist_onehot, hc_actor=None):
        feat_a = self.actor_cnn(actor_obs).flatten(1)
        lstm_a, hc_actor = self.actor_lstm(ahist_onehot, hc_actor)
        joint_a = torch.cat([feat_a, lstm_a[:, -1, :]], dim=1)
        movement_logits = self.actor_head(joint_a)
        
        # Use Channel 0 (image) and Channel 3 (path mask)
        stop_input = torch.cat([actor_obs[:, 0:1, :, :], actor_obs[:, 3:4, :, :]], dim=1)
        feat_stop = self.stop_cnn(stop_input).flatten(1)
        joint_stop = torch.cat([feat_stop, lstm_a[:, -1, :]], dim=1)
        stop_logit = self.stop_head(joint_stop).squeeze(-1)
        
        return movement_logits, stop_logit, hc_actor
