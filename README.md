# RM-GECO

This repo contains the official implementation for the paper "[Enhancing Transfer of Reinforcement Learning Agents with Abstract Contextual Embeddings](https://openreview.net/forum?id=b57S-VN84b0)", accepted at the NeurIPS'22 nCSI workshop.

Please use the below citation if you find this work useful in your research:
```
@inproceedings{azran2022enhancing,
    title={Enhancing Transfer of Reinforcement Learning Agents with Abstract Contextual Embeddings},
    author={Guy Azran and Mohamad Hosein Danesh and Stefano V Albrecht and Sarah Keren},
    booktitle={NeurIPS 2022 Workshop on Neuro Causal and Symbolic AI (nCSI)},
    year={2022},
    url={https://openreview.net/forum?id=b57S-VN84b0}
}
```

## Installation

Create and activate anaconda environment:
```shell
conda env create -f environment.yml
conda activate rmrl
```
Install our modified version of stable baselines by:
```shell
cd stable-baselines3
pip install -e .
```

## Usage

In order to run experiments and replicate our results, you may run the following command:
```shell
python -m rmrl [args]
```
In the `__main__.py`, the arguments to setup the agent, the environment, and the experiment settings are initialized. In order to change them, take a look at [this part](https://github.com/sarah-keren/RM-RL/blob/ncsi/rmrl/__main__.py#L202-L476) of code, or run the following:
```shell
python -m rmrl --help
```

This script first trains the agent for `timesteps`, in the meantime stores checkpoints of the agent every `chkp_freq` steps, and finally run experiments on a variety of tasks and settings iteratively.

## Acknowledgements

Some parts of this code relies on the [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). 
