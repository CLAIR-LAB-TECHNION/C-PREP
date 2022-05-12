# RMs for RL

## installation:

- Create and activate anaconda environment:
```bash
conda env create -f environment.yml
conda activate rmrl
```

### mujoco setup
- download mujoco binaries for [Linux](https://www.roboti.us/download/mjpro150_linux.zip) or
[macOS](https://www.roboti.us/download/mjpro150_osx.zip)
- unzip the contents into "~/.mujoco/mjpro150"
- get a license key [here](https://www.roboti.us/license.html)
- place the license key file at "~/.mujoco/mjkey.txt"
- run `python3 -c 'import mujoco_py'` to check installation
- if all else fails, follow the instructions
[here](https://neptune.ai/blog/installing-mujoco-to-work-with-openai-gym-environments)

## TODO's

### Coding
- [ ] Domains
    - [x] HalfCheetah changing velocity environment
    - [ ] Grid world
    - [ ] Multi-taxi (with single taxi)
    - [ ] More...?
- [ ] Reward machine entity
    - [x] base class as graph
    - [x] reward shaping
    - [ ] counterfactual RM observations
    - [ ] Hierarchical RM?
- [ ] potential functions
    - [x] value iteration
    - [ ] distance from goal
- [ ] stable baselines integration
    - [ ] graph inputs in experience replay/rollout buffer  !!!known issue!!!
    - [x] reward shaping
    - [x] custom feature extractor with graph support

### known issues:
- [ ] batching issue for graph inputs
- [ ] log original rewards in RM env for true performance
