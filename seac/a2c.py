import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import gym
from model import Policy, FCNetwork
from gym.spaces.utils import flatdim
from storage import RolloutStorage
from sacred import Ingredient

algorithm = Ingredient("algorithm")


@algorithm.config
def config():
    lr = 3e-4
    adam_eps = 0.001
    gamma = 0.99
    use_gae = False
    gae_lambda = 0.95
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5

    use_proper_time_limits = True
    recurrent_policy = False
    use_linear_lr_decay = False

    seac_coef = 1.0

    num_processes = 4
    num_steps = 5

    device = "cpu"


class A2C:
    @algorithm.capture()
    def __init__(
        self,
        agent_id,
        obs_space,
        action_space,
        lr,
        adam_eps,
        recurrent_policy,
        num_steps,
        num_processes,
        device,
    ):
        self.agent_id = agent_id
        self.obs_size = flatdim(obs_space)
        self.action_size = flatdim(action_space)
        self.obs_space = obs_space
        self.action_space = action_space

        self.model = Policy(
            obs_space, action_space, base_kwargs={"recurrent": recurrent_policy},
        )

        self.storage = RolloutStorage(
            obs_space,
            action_space,
            self.model.recurrent_hidden_state_size,
            num_steps,
            num_processes,
        )

        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr, eps=adam_eps)

        # self.intr_stats = RunningStats()
        self.saveables = {
            "model": self.model,
            "optimizer": self.optimizer,
        }

    def save(self, path):
        torch.save(self.saveables, os.path.join(path, "models.pt"))

    def restore(self, path):
        checkpoint = torch.load(os.path.join(path, "models.pt"))
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @algorithm.capture
    def compute_returns(self, use_gae, gamma, gae_lambda, use_proper_time_limits):
        with torch.no_grad():
            next_value = self.model.get_value(
                self.storage.obs[-1],
                self.storage.recurrent_hidden_states[-1],
                self.storage.masks[-1],
            ).detach()

        self.storage.compute_returns(
            next_value, use_gae, gamma, gae_lambda, use_proper_time_limits,
        )

    @algorithm.capture
    def update(
        self,
        storages,
        value_loss_coef,
        entropy_coef,
        seac_coef,
        max_grad_norm,
        device,
    ):

        obs_shape = self.storage.obs.size()[2:]
        action_shape = self.storage.actions.size()[-1]
        num_steps, num_processes, _ = self.storage.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.model.evaluate_actions(
            self.storage.obs[:-1].view(-1, *obs_shape),
            self.storage.recurrent_hidden_states[0].view(
                -1, self.model.recurrent_hidden_state_size
            ),
            self.storage.masks[:-1].view(-1, 1),
            self.storage.actions.view(-1, action_shape),
        )

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = self.storage.returns[:-1] - values

        policy_loss = -(advantages.detach() * action_log_probs).mean()
        value_loss = advantages.pow(2).mean()


        # calculate prediction loss for the OTHER actor
        other_agent_ids = [x for x in range(len(storages)) if x != self.agent_id]
        seac_policy_loss = 0
        seac_value_loss = 0
        for oid in other_agent_ids:

            other_values, logp, _, _ = self.model.evaluate_actions(
                storages[oid].obs[:-1].view(-1, *obs_shape),
                storages[oid]
                .recurrent_hidden_states[0]
                .view(-1, self.model.recurrent_hidden_state_size),
                storages[oid].masks[:-1].view(-1, 1),
                storages[oid].actions.view(-1, action_shape),
            )
            other_values = other_values.view(num_steps, num_processes, 1)
            logp = logp.view(num_steps, num_processes, 1)
            other_advantage = (
                storages[oid].returns[:-1] - other_values
            )  # or storages[oid].rewards

            importance_sampling = (
                logp.exp() / (storages[oid].action_log_probs.exp() + 1e-7)
            ).detach()
            # importance_sampling = 1.0
            seac_value_loss += (
                importance_sampling * other_advantage.pow(2)
            ).mean()
            seac_policy_loss += (
                -importance_sampling * logp * other_advantage.detach()
            ).mean()

        self.optimizer.zero_grad()
        (
            policy_loss
            + value_loss_coef * value_loss
            - entropy_coef * dist_entropy
            + seac_coef * seac_policy_loss
            + seac_coef * value_loss_coef * seac_value_loss
        ).backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss_coef * value_loss.item(),
            "dist_entropy": entropy_coef * dist_entropy.item(),
            "importance_sampling": importance_sampling.mean().item(),
            "seac_policy_loss": seac_coef * seac_policy_loss.item(),
            "seac_value_loss": seac_coef
            * value_loss_coef
            * seac_value_loss.item(),
        }
