import argparse
import time
import random

import numpy as np
import torch
from torch.autograd import Variable

from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

from iql import IQL
from baseline_buffer import MARLReplayBuffer

from utilities.model_saver import ModelSaver
from utilities.logger import Logger

USE_CUDA = False #torch.cuda.is_available()
TARGET_TYPES = ["simple", "double", "our-double", "our-clipped"]


class Train:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            "Reinforcement Learning experiments for multiagent environments"
        )
        self.parse_args()
        self.arglist = self.parser.parse_args()

    def parse_default_args(self):
        """
        Parse default arguments for MARL training script
        """
        # algorithm
        self.parser.add_argument("--hidden_dim", default=128, type=int)
        self.parser.add_argument("--shared_experience", action="store_true", default=False)
        self.parser.add_argument("--shared_lambda", default=1.0, type=float)
        self.parser.add_argument(
            "--targets", type=str, default="simple", help="target computation used for DQN"
        )

        # training length
        self.parser.add_argument(
            "--num_episodes", type=int, default=120000, help="number of episodes"
        )
        self.parser.add_argument(
            "--max_episode_len", type=int, default=25, help="maximum episode length"
        )

        # core training parameters
        self.parser.add_argument(
            "--n_training_threads", default=1, type=int, help="number of training threads"
        )
        self.parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
        self.parser.add_argument(
            "--tau", type=float, default=0.05, help="tau as stepsize for target network updates"
        )
        self.parser.add_argument(
            "--lr", type=float, default=0.0001, help="learning rate for Adam optimizer" #use 5e-5 for RWARE
        )
        self.parser.add_argument(
            "--seed", type=int, default=None, help="random seed used throughout training"
        )
        self.parser.add_argument(
            "--steps_per_update", type=int, default=1, help="number of steps before updates"
        )

        self.parser.add_argument(
            "--buffer_capacity", type=int, default=int(1e6), help="Replay buffer capacity"
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=128,
            help="number of episodes to optimize at the same time",
        )
        self.parser.add_argument(
            "--epsilon", type=float, default=1.0, help="epsilon value"
        )
        self.parser.add_argument(
            "--goal_epsilon", type=float, default=0.01, help="epsilon target value"
        )
        self.parser.add_argument(
            "--epsilon_decay", type=float, default=10, help="epsilon decay value"
        )
        self.parser.add_argument(
            "--epsilon_anneal_slow", action="store_true", default=False, help="anneal epsilon slowly"
        )

        # visualisation
        self.parser.add_argument("--render", action="store_true", default=False)
        self.parser.add_argument(
            "--eval_frequency", default=50, type=int, help="frequency of evaluation episodes"
        )
        self.parser.add_argument(
            "--eval_episodes", default=5, type=int, help="number of evaluation episodes"
        )
        self.parser.add_argument(
            "--run", type=str, default="default", help="run name for stored paths"
        )
        self.parser.add_argument("--save_interval", default=100, type=int)
        self.parser.add_argument("--training_returns_freq", default=100, type=int)

    def parse_args(self):
        """
        parse own arguments
        """
        self.parse_default_args()

    def extract_sizes(self, spaces):
        """
        Extract space dimensions
        :param spaces: list of Gym spaces
        :return: list of ints with sizes for each agent
        """
        sizes = []
        for space in spaces:
            if isinstance(space, Box):
                size = sum(space.shape)
            elif isinstance(space, Dict):
                size = sum(self.extract_sizes(space.values()))
            elif isinstance(space, Discrete) or isinstance(space, MultiBinary):
                size = space.n
            elif isinstance(space, MultiDiscrete):
                size = sum(space.nvec)
            else:
                raise ValueError("Unknown class of space: ", type(space))
            sizes.append(size)
        return sizes

    def create_environment(self):
        """
        Create environment instance
        :return: environment (gym interface), env_name, task_name, n_agents, observation_sizes,
                 action_sizes, discrete_actions
        """
        raise NotImplementedError()

    def reset_environment(self):
        """
        Reset environment for new episode
        :return: observation (as torch tensor)
        """
        raise NotImplementedError

    def select_actions(self, obs, explore=True):
        """
        Select actions for agents
        :param obs: joint observation
        :param explore: flag if exploration should be used
        :return: action_tensor, action_list
        """
        raise NotImplementedError()

    def environment_step(self, actions):
        """
        Take step in the environment
        :param actions: actions to apply for each agent
        :return: reward, done, next_obs (as Pytorch tensors)
        """
        raise NotImplementedError()

    def environment_render(self):
        """
        Render visualisation of environment
        """
        raise NotImplementedError()

    def fill_buffer(self, timesteps):
        """
        Randomly sample actions and store experience in buffer
        :param timesteps: number of timesteps
        """
        t = 0
        while t < timesteps:
            done = False
            obs = self.reset_environment()
            while not done and t < timesteps:
                actions = [space.sample() for space in self.action_spaces]
                rewards, dones, next_obs, _ = self.environment_step(actions)
                onehot_actions = np.zeros((len(actions), self.action_sizes[0]))
                onehot_actions[np.arange(len(actions)), actions] = 1
                self.memory.add(obs, onehot_actions, rewards, next_obs, dones)
                obs = next_obs
                t += 1
                done = all(dones)

    def eval(self, ep, n_agents):
        """
        Execute evaluation episode without exploration
        :param ep: episode number
        :param n_agents: number of agents in task
        :return: returns, episode_length, done
        """
        obs = self.reset_environment()
        self.alg.reset(ep)

        episode_returns = np.array([0.0] * n_agents)
        episode_length = 0
        done = False

        while not done and episode_length < self.arglist.max_episode_len:
            torch_obs = [
                Variable(torch.Tensor(obs[i]), requires_grad=False) for i in range(n_agents)
            ]

            actions, _ = self.select_actions(torch_obs, False)
            rewards, dones, next_obs, _ = self.environment_step(actions)

            episode_returns += rewards

            obs = next_obs
            episode_length += 1
            done = all(dones)

        return episode_returns, episode_length, done

    def set_seeds(self, seed):
        """
        Set random seeds before model creation
        :param seed (int): seed to use
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False


    def train(self):
        """
        Abstract training flow
        """
        # set random seeds before model creation
        self.set_seeds(self.arglist.seed)

        # use number of threads if no GPUs are available
        if not USE_CUDA:
            torch.set_num_threads(self.arglist.n_training_threads)

        env, env_name, task_name, n_agents, observation_spaces, action_spaces, observation_sizes, action_sizes = (
            self.create_environment()
        )
        self.env = env
        self.n_agents = n_agents
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.observation_sizes = observation_sizes
        self.action_sizes = action_sizes

        if self.arglist.max_episode_len == 25:
            steps = self.arglist.num_episodes * 20 #self.arglist.max_episode_len
        else:
            steps = self.arglist.num_episodes * self.arglist.max_episode_len
        # steps-th root of goal epsilon
        if self.arglist.epsilon_anneal_slow:
            decay_factor = self.arglist.epsilon_decay ** (1 / float(steps))
            self.arglist.decay_factor = decay_factor
            print(
                f"Epsilon is decaying with (({self.arglist.epsilon_decay} - {decay_factor}**t) / {self.arglist.epsilon_decay}) to {self.arglist.goal_epsilon} over {steps} steps."
            )
        else:
            decay_epsilon = self.arglist.goal_epsilon ** (1 / float(steps))
            self.arglist.decay_factor = decay_epsilon
            print(
                "Epsilon is decaying with factor %.7f to %.3f over %d steps."
                % (decay_epsilon, self.arglist.goal_epsilon, steps)
            )

        print("Observation sizes: ", observation_sizes)
        print("Action sizes: ", action_sizes)

        target_type = self.arglist.targets
        if not target_type in TARGET_TYPES:
            print(f"Invalid target type {target_type}!")
            return
        else:
            if target_type == "simple":
                print("Simple target computation used")
            elif target_type == "double":
                print("Double target computation used")
            elif target_type == "our-double":
                print("Agent-double target computation used")
            elif target_type == "our-clipped":
                print("Agent-clipped target computation used")
        
        # create algorithm trainer
        self.alg = IQL(
            n_agents, observation_sizes, action_sizes, self.arglist
        )

        obs_size = observation_sizes[0]
        for o_size in observation_sizes[1:]:
            assert obs_size == o_size
        act_size = action_sizes[0]
        for a_size in action_sizes[1:]:
            assert act_size == a_size

        self.memory = MARLReplayBuffer(
            self.arglist.buffer_capacity,
            n_agents,
        )

        # set random seeds past model creation
        self.set_seeds(self.arglist.seed)

        self.model_saver = ModelSaver("models", self.arglist.run)
        self.logger = Logger(
            n_agents,
            task_name,
            self.arglist.run,
        )

        self.fill_buffer(5000)

        print("Starting iterations...")
        start_time = time.process_time()
        # timer = time.process_time()
        # env_time = 0
        # step_time = 0
        # update_time = 0
        # after_ep_time = 0

        t = 0
        training_returns_saved = 0

        episode_returns = []
        episode_agent_returns = []
        for ep in range(self.arglist.num_episodes):
            obs = self.reset_environment()
            self.alg.reset(ep)

            # episode_returns = np.array([0.0] * n_agents)
            episode_length = 0
            done = False

            while not done and episode_length < self.arglist.max_episode_len:
                torch_obs = [
                    Variable(torch.Tensor(obs[i]), requires_grad=False) for i in range(n_agents)
                ]

                # env_time += time.process_time() - timer
                # timer = time.process_time()
                actions, onehot_actions = self.select_actions(torch_obs)
                # step_time += time.process_time() - timer
                # timer = time.process_time()
                rewards, dones, next_obs, info = self.environment_step(actions)

                # episode_returns += rewards

                self.memory.add(obs, onehot_actions, rewards, next_obs, dones)

                t += 1

                # env_time += time.process_time() - timer
                # timer = time.process_time()
                if (
                    len(self.memory) >= self.arglist.batch_size
                    and (t % self.arglist.steps_per_update) == 0
                ):
                    losses = self.alg.update(self.memory, USE_CUDA)
                    self.logger.log_losses(ep, losses)
                    #self.logger.dump_losses(1)

                # update_time += time.process_time() - timer
                # timer = time.process_time()
                # for displaying learned policies
                if self.arglist.render:
                    self.environment_render()

                obs = next_obs
                episode_length += 1
                done = all(dones)

                if done or episode_length == self.arglist.max_episode_len:
                    episode_returns.append(info["episode_reward"])
                    agent_returns = []
                    for i in range(n_agents):
                        agent_returns.append(info[f"agent{i}/episode_reward"])
                    episode_agent_returns.append(agent_returns)


            # env_time += time.process_time() - timer
            # timer = time.process_time()
            if  (training_returns_saved + 1) * t >= self.arglist.training_returns_freq:
                training_returns_saved += 1
                returns = np.array(episode_returns[-10:])
                mean_return = returns.mean()
                agent_returns = np.array(episode_agent_returns[-10:])
                mean_agent_return = agent_returns.mean(axis=0)

                self.logger.log_training_returns(t, mean_return, mean_agent_return)

            if ep % self.arglist.eval_frequency == 0:
                eval_returns = np.zeros((self.arglist.eval_episodes, n_agents))
                for i in range(self.arglist.eval_episodes):
                    ep_returns, _, _ = self.eval(ep, n_agents)
                    eval_returns[i, :] = ep_returns
                self.logger.log_episode(
                    ep, eval_returns.mean(0), eval_returns.var(0), self.alg.agents[0].epsilon
                )
                self.logger.dump_episodes(1)
            if ep % 100 == 0 and ep > 0:
                duration = time.process_time() - start_time
                self.logger.dump_train_progress(ep, self.arglist.num_episodes, duration)

            if ep % self.arglist.save_interval == 0 and ep > 0:
                # save models
                print("Remove previous models")
                self.model_saver.clear_models()
                print("Saving intermediate models")
                self.model_saver.save_models(self.alg, str(ep))
                # save logs
                print("Remove previous logs")
                self.logger.clear_logs()
                print("Saving intermediate logs")
                self.logger.save_training_returns(extension=str(ep))
                self.logger.save_episodes(extension=str(ep))
                self.logger.save_losses(extension=str(ep))
                # save parameter log
                self.logger.save_parameters(
                    env_name,
                    task_name,
                    n_agents,
                    observation_sizes,
                    action_sizes,
                    self.arglist,
                )

            # after_ep_time += time.process_time() - timer
            # timer = time.process_time()
            # print(f"Episode {ep} times:")
            # print(f"\tEnv time: {env_time}s")
            # print(f"\tStep time: {step_time}s")
            # print(f"\tUpdate time: {update_time}s")
            # print(f"\tAfter Ep time: {after_ep_time}s")
            # env_time = 0
            # step_time = 0
            # update_time = 0
            # after_ep_time = 0

        duration = time.process_time() - start_time
        print("Overall duration: %.2fs" % duration)

        # save models
        print("Remove previous models")
        self.model_saver.clear_models()
        print("Saving final models")
        self.model_saver.save_models(self.alg, "final")

        # save logs
        print("Remove previous logs")
        self.logger.clear_logs()
        print("Saving final logs")
        self.logger.save_episodes(extension="final")
        self.logger.save_losses(extension="final")
        self.logger.save_duration_cuda(duration, torch.cuda.is_available())

        # save parameter log
        self.logger.save_parameters(
            env_name,
            task_name,
            n_agents,
            observation_sizes,
            action_sizes,
            self.arglist,
        )

        env.close()

    if __name__ == "__main__":
        train = Train()
        train.train()
