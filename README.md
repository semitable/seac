# Shared Experience Actor Critic

This repository is the official implementation of [Shared Experience Actor Critic](https://arxiv.org/abs/2006.07169). 

## Requirements

For the experiments in LBF and RWARE, please install from:
- [Level Based Foraging Official Repo](https://github.com/uoe-agents/lb-foraging)
- [Multi-Robot Warehouse Official Repo](https://github.com/uoe-agents/robotic-warehouse)

Also requires, PyTorch 1.6+

## Training - SEAC
To train the agents in the paper, navigate to the seac directory:
```
cd seac
```

And run:

```train
python train.py with <env config>
```

Valid environment configs are: 
- `env_name=Foraging-15x15-3p-4f-v0 time_limit=25`
- ...
- `env_name=Foraging-12x12-2p-1f-v0 time_limit=25` or any other foraging environment size/configuration.
- `env_name=rware-tiny-2ag-v1 time_limit=500` 
- `env_name=rware-tiny-4ag-v1 time_limit=500` 
- ...
- `env_name=rware-tiny-2ag-hard-v1 time_limit=500` or any other rware environment size/configuration.
## Training - SEQL

To train the agents in the paper, navigate to the seac directory:
```
cd seql
```

And run the training script. Possible options are: 
- `python lbf_train.py --env Foraging-12x12-2p-1f-v0` 
- ...
- `python lbf_train.py --env Foraging-15x15-3p-4f-v0` or any other foraging environment size/configuration.
- `python rware_train.py --env "rware-tiny-2ag-v1"`
- ...
- `python rware_train.py --env "rware-tiny-4ag-v1"`or any other rware environment size/configuration.

## Evaluation/Visualization - SEAC

To load and render the pretrained models in SEAC, run in the seac directory

```eval
python evaluate.py
```

## Citation
```
@inproceedings{christianos2020shared,
  title={Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning},
  author={Christianos, Filippos and Sch{\"a}fer, Lukas and Albrecht, Stefano V},
  booktitle = {Advances in Neural Information Processing Systems},
  year={2020}
}
```
