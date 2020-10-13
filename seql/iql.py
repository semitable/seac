import os
import time

import numpy as np

import torch

from marl_algorithm import MarlAlgorithm
from marl_utils import soft_update
from agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else device)

MSELoss = torch.nn.MSELoss()


class IQL(MarlAlgorithm):
    """
    (Deep) Independent Q-Learning (IQL) class

    Original IQL paper:
    Tan, M. (1993).
    Multi-agent reinforcement learning: Independent vs. cooperative agents.
    In Proceedings of the tenth international conference on machine learning (pp. 330-337).

    Link: http://web.mit.edu/16.412j/www/html/Advanced%20lectures/2004/Multi-AgentReinforcementLearningIndependentVersusCooperativeAgents.pdf

    Deep Q-Learning (DQN) paper:
    Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Petersen, S. (2015).
    Human-level control through deep reinforcement learning.
    Nature, 518(7540), 529.

    Link: https://www.nature.com/articles/nature14236?wm=book_wap_0005
    """

    def __init__(self, n_agents, observation_sizes, action_sizes, params):
        """
        Initialise parameters for IQL training
        :param n_agents: number of agents
        :param observation_sizes: dimension of observation for each agent
        :param action_sizes: dimension of action for each agent
        :param params: parsed arglist parameter list
        """
        super(IQL, self).__init__(
            n_agents, observation_sizes, action_sizes, params
        )

        self.shared_experience = params.shared_experience
        self.shared_lambda = params.shared_lambda
        self.targets_type = params.targets
        self.model_dev = device  # device for model
        self.trgt_model_dev = device  # device for target model

        self.agents = [
            Agent(observation_sizes[i], action_sizes[i], params)
            for i in range(n_agents)
        ]

    def reset(self, episode):
        """
        Reset algorithm for new episode
        :param episode: new episode number
        """
        self.prep_rollouts(device=device)

    def prep_rollouts(self, device=device):
        """
        Prepare networks for rollout steps and use given device
        :param device: device to cast networks to
        """
        for a in self.agents:
            a.model.eval()
        if device == "gpu":
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.model_dev == device:
            for a in self.agents:
                a.model = fn(a.model)
            self.model_dev = device

    def step(self, observations, explore=False, available_actions=None):
        """
        Take a step forward in environment with all agents
        :param observations: list of observations for each agent
        :param explore: flag whether or not to add exploration noise
        :param available_actions: binary vector (n_agents, n_actions) where each list contains
                                  binary values indicating whether action is applicable
        :return: list of actions for each agent
        """
        if available_actions is None:
            return [a.step(obs, explore)[0] for a, obs in zip(self.agents, observations)]
        else:
            return [
                a.step(obs, explore, available_actions[i])[0]
                for i, (a, obs) in enumerate(zip(self.agents, observations))
            ]
        self.t_steps += 1

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_model, a.model, self.params.tau)

    def prep_training(self, device="gpu"):
        """
        Prepare networks for training and use given device
        :param device: device to cast networks to
        """
        for a in self.agents:
            a.model.train()
            a.target_model.train()
        if device == "gpu":
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.model_dev == device:
            for a in self.agents:
                a.model = fn(a.model)
            self.model_dev = device
        if not self.trgt_model_dev == device:
            for a in self.agents:
                a.target_model = fn(a.target_model)
            self.trgt_model_dev = device

    def update_agent(self, sample, agent_i, use_cuda):
        """
        Update parameters of agent model based on sample from replay buffer
        :param sample: tuple of (observations, actions, rewards, next
                        observations, and episode end masks) sampled randomly from
                        the replay buffer
        :param agent_i: index of agent to update
        :param use_cuda: flag if cuda/ gpus should be used
        :return: q loss
        """
        # timer = time.process_time()
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        curr_agent.optimizer.zero_grad()

        if self.targets_type == "simple":
            q_next_states = curr_agent.target_model(next_obs)
            target_next_states = q_next_states.max(-1)[0]
        elif self.targets_type == "double":
            q_tp1_values = curr_agent.model(next_obs).detach()
            _, a_prime = q_tp1_values.max(1)
            q_next_states = curr_agent.target_model(next_obs)
            target_next_states = q_next_states.gather(1, a_prime.unsqueeze(1))
        elif self.targets_type == "our-double":
            # this does not use target network but instead uses the network of another agent
            other_agent = self.agents[int(not agent_i)] # or sample any other agent except agent_i (if agents>2)
            q_tp1_values = curr_agent.model(next_obs).detach()
            _, a_prime = q_tp1_values.max(1)
            q_next_states = other_agent.model(next_obs).detach()
            target_next_states = q_next_states.gather(1, a_prime.unsqueeze(1))
        elif self.targets_type == "our-clipped":
            # uses TD3's clipped q networks by taking the min of all agents models
            target_next_states = torch.cat([a.model(next_obs).detach().max(dim=1)[0].unsqueeze(1) for a in self.agents], dim=1).min(dim=1)[0]


        # compute Q-targets for current states
        target_states = (
            rews.view(-1, 1)
            + self.gamma * target_next_states.view(-1, 1)# * (1 - dones.view(-1, 1))
        )

        # target_timer = time.process_time() - timer
        # print(f"\t\tTarget computation time: {target_timer}")
        # timer = time.process_time()


        # local Q-values
        all_q_states = curr_agent.model(obs)
        q_states = torch.sum(all_q_states * acs, dim=1).view(-1, 1)

        # q_timer = time.process_time() - timer
        # print(f"\t\tQ-values computation time: {q_timer}")
        # timer = time.process_time()

        if self.shared_experience:
            batch_size_agent = self.batch_size // self.n_agents
            agent_mask = np.arange(batch_size_agent * agent_i, batch_size_agent * (agent_i + 1))
            other_agents_mask = np.concatenate([np.arange(0, batch_size_agent * agent_i), np.arange(batch_size_agent * (agent_i + 1), self.batch_size)])
            qloss = MSELoss(q_states[agent_mask], target_states[agent_mask].detach())
            qloss += self.shared_lambda * MSELoss(q_states[other_agents_mask], target_states[other_agents_mask].detach())
        else:
            qloss = MSELoss(q_states, target_states.detach())

        # loss_timer = time.process_time() - timer
        # print(f"\t\tLoss computation time: {loss_timer}")
        qloss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.model.parameters(), 0.5)
        curr_agent.optimizer.step()

        return qloss

    def update(self, memory, use_cuda=False):
        """
        Train agent models based on memory samples
        :param memory: replay buffer memory to sample experience from
        :param use_cuda: flag if cuda/ gpus should be used
        :return: qnetwork losses
        """
        q_losses = []
        if use_cuda:
            self.prep_training(device="gpu")
        else:
            self.prep_training(device=device)
        if self.shared_experience:
            samples = memory.sample_shared(self.params.batch_size)
        for a_i in range(self.n_agents):
            # print(f"\tUpdate agent {a_i}:")
            # timer = time.process_time()
            if not self.shared_experience:
                samples = memory.sample(self.params.batch_size, a_i)
            # sample_time = time.process_time() - timer
            # print(f"\t\tSample time from memory: {sample_time}")
            q_loss = self.update_agent(samples, a_i, use_cuda=False)
            q_losses.append(q_loss)
        self.update_all_targets()
        self.prep_rollouts(device=device)

        return q_losses

    def load_model_networks(self, directory, extension="_final"):
        """
        Load model networks of all agents
        :param directory: path to directory where to load models from
        """
        for i, agent in enumerate(self.agents):
            name = "iql_agent%d_params" % i
            name += extension
            agent.model.load_state_dict(
                torch.load(os.path.join(directory, name), map_location=device)
            )
