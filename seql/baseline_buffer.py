import numpy as np
import random
import torch

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class MARLReplayBuffer(object):
    def __init__(self, size, num_agents):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        num_agents: int
            Number of agents
        """
        self.size = size
        self.num_agents = num_agents
        self.buffers = [ReplayBuffer(size) for _ in range(num_agents)]

    def __len__(self):
        return len(self.buffers[0])

    def add(self, observations, actions, rewards, next_observations, dones):
        for i, (o, a, r, no, d) in enumerate(zip(observations, actions, rewards, next_observations, dones)):
            self.buffers[i].add(o, a, r, no, d)

    def sample(self, batch_size, agent_i):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        agent_i: int
            Index of agent to sample for
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        cast = lambda x: torch.from_numpy(x).float()
        obs, act, rew, next_obs, done = self.buffers[agent_i].sample(batch_size)
        obs = cast(obs).squeeze()
        act = cast(act)
        rew = cast(rew)
        next_obs = cast(next_obs).squeeze()
        done = cast(done)
        return obs, act, rew, next_obs, done

    def sample_shared(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        batch_size_each = batch_size // self.num_agents
        obs = []
        act = []
        rew = []
        next_obs = []
        done = []
        for agent_i in range(self.num_agents):
            o, a, r, no, d = self.buffers[agent_i].sample(batch_size_each)
            obs.append(o)
            act.append(a)
            rew.append(r)
            next_obs.append(no)
            done.append(d)
        cast = lambda x: torch.from_numpy(x).float()
        obs = cast(np.vstack(obs)).squeeze()
        act = cast(np.vstack(act))
        rew = cast(np.vstack(rew))
        next_obs = cast(np.vstack(next_obs)).squeeze()
        done = cast(np.vstack(done))
        return obs, act, rew, next_obs, done
