import random
import numpy as np
import torch

class MarlAlgorithm:
    """
    abstract class for MARL algorithm
    """

    def __init__(self, n_agents, observation_sizes, action_sizes, params):
        """
        Initialise parameters for MARL training
        :param n_agents: number of agents
        :param observation_sizes: dimension of observation for each agent
        :param action_sizes: dimension of action for each agent
        :param params: parsed arglist parameter list
        """
        self.n_agents = n_agents
        self.observation_sizes = observation_sizes
        self.action_sizes = action_sizes
        self.params = params
        self.batch_size = params.batch_size
        self.gamma = params.gamma
        self.tau = params.tau
        self.learning_rate = params.lr
        self.epsilon = params.epsilon
        self.decay_factor = params.decay_factor
        self.seed = params.seed

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        self.t_steps = 0

    def reset(self, episode):
        """
        Reset algorithm for new episode
        :param episode: new episode number
        """
        raise NotImplementedError

    def step(self, observations, explore=False, available_actions=None):
        """
        Take a step forward in environment with all agents
        :param observations: list of observations for each agent
        :param explore: flag whether or not to add exploration noise
        :param available_actions: binary vector (n_agents, n_actions) where each list contains
                                  binary values indicating whether action is applicable
        :return: list of actions for each agent
        """
        raise NotImplementedError

    def update(self, memory, use_cuda=False):
        """
        Train agent models based on memory samples
        :param memory: replay buffer memory to sample experience from
        :param use_cuda: flag if cuda/ gpus should be used
        :return: tuple of loss lists
        """
        raise NotImplementedError

    def load_model_networks(self, directory, extension="_final"):
        """
        Load model networks of all agents
        :param directory: path to directory where to load models from
        """
        raise NotImplementedError
