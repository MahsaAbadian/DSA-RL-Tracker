#!/usr/bin/env python3
"""
Shared model architectures for DSA RL Experiment.
- AsymmetricActorCritic: Full model for training (Actor + Critic)
- ActorOnly: Lightweight model for inference (Actor only, no Critic)
"""
import torch
import torch.nn as nn

# ---------- HELPERS ----------
def gn(c): 
    """GroupNorm helper"""
    return nn.GroupNorm(4, c)

# ---------- FULL MODEL (TRAINING) ----------
class AsymmetricActorCritic(nn.Module):
    """
    Full Actor-Critic model for training.
    Uses both actor (policy) and critic (value function) for PPO.
    """
    def __init__(self, n_actions=9, K=8):
        super().__init__()
        # Actor CNN
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # Actor LSTM (Stateful memory)
        self.actor_lstm = nn.LSTM(input_size=n_actions, hidden_size=64, batch_first=True)
        self.actor_head = nn.Sequential(nn.Linear(128, 128), nn.PReLU(), nn.Linear(128, n_actions))

        # Critic CNN (Privileged info - uses GT map)
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.critic_lstm = nn.LSTM(input_size=n_actions, hidden_size=64, batch_first=True)
        self.critic_head = nn.Sequential(nn.Linear(128, 128), nn.PReLU(), nn.Linear(128, 1))
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=torch.sqrt(torch.tensor(2.0)))
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)

    def forward(self, actor_obs, critic_gt, ahist_onehot, hc_actor=None, hc_critic=None):
        """
        Forward pass for training.
        
        Args:
            actor_obs: Actor observation (B, 4, 33, 33)
            critic_gt: Ground truth map for critic (B, 1, 33, 33)
            ahist_onehot: Action history one-hot (B, K, n_actions)
            hc_actor: Actor LSTM hidden state (optional)
            hc_critic: Critic LSTM hidden state (optional)
        
        Returns:
            logits: Action logits (B, n_actions)
            value: State value estimate (B,)
            hc_actor: Updated actor hidden state
            hc_critic: Updated critic hidden state
        """
        # Actor
        feat_a = self.actor_cnn(actor_obs).flatten(1)
        lstm_a, hc_actor = self.actor_lstm(ahist_onehot, hc_actor)
        joint_a = torch.cat([feat_a, lstm_a[:, -1, :]], dim=1)
        logits = self.actor_head(joint_a)

        # Critic
        critic_input = torch.cat([actor_obs, critic_gt], dim=1)
        feat_c = self.critic_cnn(critic_input).flatten(1)
        lstm_c, hc_critic = self.critic_lstm(ahist_onehot, hc_critic)
        joint_c = torch.cat([feat_c, lstm_c[:, -1, :]], dim=1)
        value = self.critic_head(joint_c).squeeze(-1)
        
        return logits, value, hc_actor, hc_critic

# ---------- ACTOR-ONLY MODEL (INFERENCE) ----------
class ActorOnly(nn.Module):
    """
    Lightweight Actor-only model for inference.
    Only contains the policy network, no critic.
    ~50% smaller than full model.
    """
    def __init__(self, n_actions=9, K=8):
        super().__init__()
        # Actor CNN (same as AsymmetricActorCritic)
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=3, dilation=3), gn(64), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # Actor LSTM
        self.actor_lstm = nn.LSTM(input_size=n_actions, hidden_size=64, batch_first=True)
        self.actor_head = nn.Sequential(nn.Linear(128, 128), nn.PReLU(), nn.Linear(128, n_actions))

    def forward(self, actor_obs, ahist_onehot, hc_actor=None):
        """
        Forward pass for inference.
        
        Args:
            actor_obs: Actor observation (B, 4, 33, 33)
            ahist_onehot: Action history one-hot (B, K, n_actions)
            hc_actor: Actor LSTM hidden state (optional)
        
        Returns:
            logits: Action logits (B, n_actions)
            hc_actor: Updated actor hidden state
        """
        feat_a = self.actor_cnn(actor_obs).flatten(1)
        lstm_a, hc_actor = self.actor_lstm(ahist_onehot, hc_actor)
        joint_a = torch.cat([feat_a, lstm_a[:, -1, :]], dim=1)
        logits = self.actor_head(joint_a)
        
        return logits, hc_actor

