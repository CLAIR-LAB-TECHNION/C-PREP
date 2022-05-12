# RMs for RL

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
