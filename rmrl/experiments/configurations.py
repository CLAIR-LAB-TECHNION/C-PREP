from enum import Enum
from pathlib import Path
from typing import Iterable, Union

from rmrl.envs.grid.reward_machines.single_taxi import TaxiEnvRM
from rmrl.envs.grid.single_taxi import fixed_entities_env, changing_map_env
from rmrl.nn.models import cur_state_embedding
from rmrl.reward_machines.potential_functions import ValueIteration

# data constants and options
NUM_CONTEXT_PAIR_SAMPLES = 100  # 500
OVERSAMPLE_FACTOR = 5
SRC_SET_FRAC = 0.8
SAMPLE_SEED = 24

LEARNING_RATES = [1e-4]  # [1 / (10 ** i) for i in range(1, 6)]
BATCH_SIZES = [32]  # [2 ** i for i in range(3, 10)]
TOTAL_TIMESTEPS = 5e5
DQN_EXPLORATION_FRACTIONS = [0.3]  # np.linspace(0, 1, 10).tolist()
OUT_DIMS = [32]  # [16, 32, 64, 128, 256]
HIDDEN_DIMS = [[32, 32]]  # list(set(tuple(sorted(hd)) for hd in powerset(OUT_DIMS, max_subset_len=2)))
NODE_AGGS = [cur_state_embedding]  # [ignore_state_mean, cur_state_embedding]
GOAL_STATE_REWARDS = [1]
GRID_RESOLUTIONS = [None, (2, 2), (1, 1)]
FUEL_RESOLUTIONS = [None]  # TODO update fuel resolutions
NUM_SEEDS = 10
BASE_SEED = 42
SEEDS = [BASE_SEED * i for i in range(1, NUM_SEEDS + 1)]
ON_POLICY_N_STEPS = [1024]
OFF_POLICY_LEARNING_STARTS = [0]
OFF_POLICY_TRAIN_FREQ = [1]
OFF_POLICY_GRADIENT_STEPS = [1]

LOG_INTERVAL = 1
N_EVAL_EPISODES = 100
EVAL_FREQ = 1000
MAX_NO_IMPROVEMENT_EVALS = 10
MIN_EVALS = 50

# defaults for reward shaping
DEFAULT_RS_GAMMA = 0.9
DEFAULT_POT_FN = ValueIteration()

# default directories
EXPERIMENTS_DUMPS_DIR = Path('experiment_dumps/')
CONTEXTS_DIR = Path('sampled_contexts/')
PRETRAINED_GNN_DIR = Path('grpt_model/')


# important dictionary keys for RMENV_Dict
ENV_KEY = 'env'
RM_KEY = 'rm'
ENV_KWARGS_KEY = 'env_kwargs'
RM_KWARGS_KEY = 'rm_kwargs'
CONTEXT_SPACES_KEY = 'context_spaces'


class SupportedExperiments(Enum):
    NO_TRANSFER = 'NoTransferExperiment'
    WITH_TRANSFER = 'WithTransferExperiment'


class SupportedEnvironments(Enum):
    SMALL = 'small'
    GRID_NAVIGATION = 'grid_nav'


class ContextSpaces(Enum):
    FIXED_ENTITIES = 'fixed_entities'
    CHANGING_MAP = 'changing_map'


RMENV_DICT = {
    SupportedEnvironments.SMALL: {
        ENV_KWARGS_KEY: {
            'num_passengers': 1,
            'pickup_only': True,
            'max_steps': 25,
            'domain_map': [
                "+-------+",
                "| : | : |",
                "| : : : |",
                "| : | | |",
                "| | | : |",
                "+-------+"
            ]
        },
        CONTEXT_SPACES_KEY: {
            ContextSpaces.FIXED_ENTITIES: {
                ENV_KEY: fixed_entities_env,
                RM_KEY: TaxiEnvRM,
            },
            ContextSpaces.CHANGING_MAP: {
                ENV_KEY: changing_map_env,
                RM_KEY: TaxiEnvRM,
            }
        }
    },
    SupportedEnvironments.GRID_NAVIGATION: {
        ENV_KWARGS_KEY: {
            'num_passengers': 1,
            'pickup_only': True,
            'max_steps': 200
        },
        CONTEXT_SPACES_KEY: {
            ContextSpaces.FIXED_ENTITIES: {
                ENV_KEY: fixed_entities_env,
                RM_KEY: TaxiEnvRM,
            },
            ContextSpaces.CHANGING_MAP: {
                ENV_KEY: changing_map_env,
                RM_KEY: TaxiEnvRM,
            }
        }
    }
}


class Mods(Enum):
    AS = 'AS'
    RS = 'RS'
    GECO = 'GECO'
    GECOUPT = 'GECOUPT'
    # VIN = 'VIN'  # NOT IMPLEMENTED


class Algos(Enum):
    DQN = 'DQN'
    # A2C = 'A2C'
    # DDPG = 'DDPG'
    PPO = 'PPO'
    # SAC = 'SAC'


OFF_POLICY_ALGOS = [Algos.DQN]
ON_POLICY_ALGOS = [Algos.PPO]


class ExperimentConfiguration:
    def __init__(self,
                 env: SupportedEnvironments,
                 cspace: ContextSpaces,
                 alg: Algos,
                 mods: Iterable[Mods],
                 rm_kwargs: dict,
                 model_kwargs: dict,
                 alg_kwargs: dict,
                 seed: int):
        self.env = env
        self.cspace = cspace
        self.alg = alg
        self.mods = list(mods)
        self.rm_kwargs = rm_kwargs
        self.model_kwargs = model_kwargs
        self.alg_kwargs = alg_kwargs
        self.seed = seed

        self.env_kwargs = RMENV_DICT[env][ENV_KWARGS_KEY]

    @property
    def env_name(self):
        return self.env.value

    @property
    def cspace_name(self):
        return self.cspace.value

    @property
    def all_kwargs(self):
        return dict(**self.rm_kwargs, **self.model_kwargs, **self.alg_kwargs)

    def __contains__(self, item: Union[Mods, Algos]):
        return (item in self.mods or  # item is a contained modification
                item in [self.env, self.cspace, self.alg],  # item is one of the single values
                item in self.all_kwargs.keys())

    def __str__(self):
        return f'underlying RL algorithm: {self.alg.value}\n' \
               f'modifications: {", ".join(map(lambda m: m.value, self.mods))}\n' \
               f'random seed: {self.seed}'

    def __repr__(self):
        return '/'.join(self.__repr_value(n, v) for n, v in [('env', self.env),
                                                             ('cspace', self.cspace),
                                                             ('alg', self.alg),
                                                             ('mods', self.mods),
                                                             ('rm_kwargs', self.rm_kwargs),
                                                             ('alg_kwargs', self.alg_kwargs),
                                                             ('model_kwargs', self.model_kwargs),
                                                             ('seed', self.seed)])

    def __repr_value(self, name, value):
        rv = f'{name}-'
        if isinstance(value, Enum):
            return self.__repr_value(name, value.value)
        elif callable(value):
            if hasattr(value, '__name__'):
                return rv + value.__name__
            else:
                return rv.__class__.__name__
        elif isinstance(value, dict):
            return rv + '(' + ','.join(f'({self.__repr_value(k, v)})' for k, v in value.items()) + ')'
        elif isinstance(value, list) or isinstance(value, tuple) or isinstance(value, set):
            return rv + '(' + ','.join(self.__repr_value(str(i), v).split("-", 1)[-1]
                                       for i, v in enumerate(value)) + ')'
        else:
            return rv + str(value)
