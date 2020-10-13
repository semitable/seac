# Independent Q-Learning (IQL) with DQNs for robotic warehouse environment

## Implementation
This implementation is based on Pytorch.

## References
Original IQL paper: Tan, M. (1993). Multi-agent reinforcement learning: Independent vs. cooperative agents. In Proceedings of the tenth international conference on machine learning, pages 330–337. ([IQL paper PDF](http://web.mit.edu/16.412j/www/html/Advanced%20lectures/2004/Multi-AgentReinforcementLearningIndependentVersusCooperativeAgents.pdf))

Deep Q-Learning (DQL) paper: Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., and Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540):529–533. ([DQL paper PDF](https://daiwk.github.io/assets/dqn.pdf))

## Environment

Build for robotic warehouse task. Can be extended with new training script for any environment, but assumes same action and observation space for all agents (for effiency).

## Shared experience

Shared experience training is implemented to use experience of all agents' buffers with equal shares (flag `--shared_experience`). Shared experience MSE loss is weighted by `--shared_lambda` which is by default set to 1.0.
