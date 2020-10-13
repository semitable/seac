import torch
import robotic_warehouse
import lbforaging
import gym

from a2c import A2C
from wrappers import RecordEpisodeStatistics, TimeLimit

path = "pretrained/rware-small-4ag"
env_name = "rware-small-4ag-v1"
time_limit = 500 # 25 for LBF

RUN_STEPS = 1500

env = gym.make(env_name)
env = TimeLimit(env, time_limit)
env = RecordEpisodeStatistics(env)

agents = [
    A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
    for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
]

for agent in agents:
    agent.restore(path + f"/agent{agent.agent_id}")

obs = env.reset()

for i in range(RUN_STEPS):
    obs = [torch.from_numpy(o) for o in obs]
    _, actions, _ , _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
    actions = [a.item() for a in actions]
    env.render()
    obs, _, done, info = env.step(actions)
    if all(done):
        obs = env.reset()
        print("--- Episode Finished ---")
        print(f"Episode rewards: {sum(info['episode_reward'])}")
        print(info)
        print(" --- ")


