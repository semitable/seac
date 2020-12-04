import time

import numpy as np

import gym

import lbforaging

from wrappers import RecordEpisodeStatistics

from train import Train


class LBFTrain(Train):
    """
    Training environment for the level-based foraging environment (LBF)
    """

    def __init__(self):
        """
        Create LBF Train instance
        """
        super(LBFTrain, self).__init__()

    def parse_args(self):
        """
        parse own arguments including default args and rware specific args
        """
        self.parse_default_args()
        self.parser.add_argument(
            "--env", type=str, default=None, help="name of the lbf environment"
        )

    def create_environment(self):
        """
        Create environment instance
        :return: environment (gym interface), env_name, task_name, n_agents, observation_sizes,
                 action_sizes
        """
        # load scenario from script
        env = gym.make(self.arglist.env)
        env = RecordEpisodeStatistics(env, deque_size=10)

        task_name = self.arglist.env

        n_agents = env.n_agents

        print("Observation spaces: ", [env.observation_space[i] for i in range(n_agents)])
        print("Action spaces: ", [env.action_space[i] for i in range(n_agents)])
        observation_sizes = self.extract_sizes(env.observation_space)
        action_sizes = self.extract_sizes(env.action_space)

        return (
            env,
            "lbf",
            task_name,
            n_agents,
            env.observation_space,
            env.action_space,
            observation_sizes,
            action_sizes,
        )

    def reset_environment(self):
        """
        Reset environment for new episode
        :return: observation (as torch tensor)
        """
        obs = self.env.reset()
        obs = [np.expand_dims(o, axis=0) for o in obs]
        return obs

    def select_actions(self, obs, explore=True):
        """
        Select actions for agents
        :param obs: joint observations for agents
        :return: actions, onehot_actions
        """
        # get actions as torch Variables
        torch_agent_actions = self.alg.step(obs, explore)
        # convert actions to numpy arrays
        onehot_actions = [ac.data.numpy() for ac in torch_agent_actions]
        # convert onehot to ints
        actions = np.argmax(onehot_actions, axis=-1)

        return actions, onehot_actions

    def environment_step(self, actions):
        """
        Take step in the environment
        :param actions: actions to apply for each agent
        :return: reward, done, next_obs (as Pytorch tensors), info
        """
        # environment step
        next_obs, reward, done, info = self.env.step(actions)
        next_obs = [np.expand_dims(o, axis=0) for o in next_obs]
        return reward, done, next_obs, info

    def environment_render(self):
        """
        Render visualisation of environment
        """
        self.env.render()
        time.sleep(0.1)


if __name__ == "__main__":
    train = LBFTrain()
    train.train()
