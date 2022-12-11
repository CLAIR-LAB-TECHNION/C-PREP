import os
import pprint
from enum import Enum
from pathlib import Path
from typing import Iterable
from torch_geometric.nn import GATConv, GINEConv

from rmrl.envs.grid.reward_machines.single_taxi import TaxiEnvRM
from rmrl.envs.grid.single_taxi import fixed_entities_env, changing_map_env
from rmrl.nn.models import cur_state_embedding, ignore_state_mean
from rmrl.reward_machines.potential_functions import ValueIteration
from rmrl.utils.lr_schedulers import *

from multi_taxi import Event

# data constants and options
NUM_SRC_SAMPLES = [10]
NUM_TGT_SAMPLES = [1]
OVERSAMPLE_FACTOR = 5

MODS = [[]]

LEARNING_RATES = [1e-4]  # [1 / (10 ** i) for i in range(1, 6)]
BATCH_SIZES = [32]  # [2 ** i for i in range(3, 10)]
GAMMA = [0.99]
TOTAL_TIMESTEPS = 5e5
TARGET_TIMESTEPS = None
TARGET_EVAL_FREQ = None
DQN_EXPLORATION_TIMESTEPS = [5e4]  # np.linspace(0, 1, 10).tolist()
DQN_TARGET_UPDATE_INTERVAL = [500]
OUT_DIMS = [32]  # [16, 32, 64, 128, 256]
HIDDEN_DIMS = [[32, 32]]  # list(set(tuple(sorted(hd)) for hd in powerset(OUT_DIMS, max_subset_len=2)))
GNN_TYPE = [GATConv]
NODE_AGGS = [cur_state_embedding]  # [ignore_state_mean, cur_state_embedding]
GOAL_STATE_REWARDS = [1.]
GRID_RESOLUTIONS = [None]
FUEL_RESOLUTIONS = [None]
NUM_SEEDS = 30
SEED = [42]
ON_POLICY_N_STEPS = [1024]
ON_POLICY_ENT_COEF = 0.01
PPO_N_EPOCHS = [10]
OFF_POLICY_LEARNING_STARTS = [0]  # no waiting for training by default
OFF_POLICY_TRAIN_FREQ = [4]  # align with DQN default
OFF_POLICY_GRADIENT_STEPS = [1]  # align with DQN default

LOG_INTERVAL = 1
N_EVAL_EPISODES = 100
EVAL_FREQ = 1000
MAX_NO_IMPROVEMENT_EVALS = 10
MIN_TIMESTEPS = 50_000

# defaults for reward shaping
DEFAULT_RS_GAMMA = 0.9
DEFAULT_POT_FN = ValueIteration()

# CV experiment defaults
DEFAULT_NUM_TGT_SAMPLES_FOR_CV = 1

# default directories
EXPERIMENTS_DUMPS_DIR = Path('experiment_dumps/')
PRETRAINED_GNN_DIR = Path('grpt_model/')

MODELS_DIR = 'models'
LOGS_DIR = 'logs'
TB_LOG_DIR = 'tensorboard'
EVAL_LOG_DIR = 'eval'
SAVED_CONTEXTS_DIR = 'sampled_contexts'

# important dictionary keys for RMENV_Dict
ENV_KEY = 'env'
RM_KEY = 'rm'
ENV_KWARGS_KEY = 'env_kwargs'
RM_KWARGS_KEY = 'rm_kwargs'
CONTEXT_SPACES_KEY = 'context_spaces'

# saved models
BEST_MODEL_NAME = 'best_model'
FINAL_MODEL_NAME = 'final_model'
CHKP_MODEL_NAME_PREFIX = 'chkp'


class SupportedExperiments(Enum):
    NO_TRANSFER = 'NoTransferExperiment'
    WITH_TRANSFER = 'WithTransferExperiment'
    CV_TRANSFER = 'CVTransferExperiment'


class SupportedEnvironments(Enum):
    SMALL_GN = 'small_gn'
    GN = 'gn'
    APPLE_PICKING_2 = 'ap_2'
    APPLE_PICKING_3 = 'ap_3'
    APPLE_PICKING_4 = 'ap_4'
    APPLE_PICKING_5 = 'ap_5'

    SMALL_PD = 'small_pd'
    PD_1 = 'pd_1'
    PD_2 = 'pd_2'
    PD_3 = 'pd_3'
    PD_4 = 'pd_4'
    PD_5 = 'pd_5'


class ContextSpaces(Enum):
    FIXED_ENTITIES = 'fixed_entities'
    CHANGING_MAP = 'changing_map'


RMENV_DICT = {
    SupportedEnvironments.SMALL_GN: {
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
            ],
            'reward_table': {
                Event.STEP: 0,
                Event.PICKUP: 0,
                Event.DEAD: 0,
                Event.OBJECTIVE: 1
            },
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
    SupportedEnvironments.GN: {
        ENV_KWARGS_KEY: {
            'num_passengers': 1,
            'pickup_only': True,
            'max_steps': 100,
            'reward_table': {
                Event.STEP: 0,
                Event.PICKUP: 0,
                Event.DEAD: 0,
                Event.OBJECTIVE: 1
            },
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
    SupportedEnvironments.APPLE_PICKING_2: {
        ENV_KWARGS_KEY: {
            'num_passengers': 2,
            'pickup_only': True,
            'max_steps': 110,
            'reward_table': {
                Event.STEP: 0,
                Event.PICKUP: 0,
                Event.DEAD: 0,
                Event.OBJECTIVE: 1
            },
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
    SupportedEnvironments.APPLE_PICKING_3: {
        ENV_KWARGS_KEY: {
            'num_passengers': 3,
            'pickup_only': True,
            'max_steps': 120,
            'reward_table': {
                Event.STEP: 0,
                Event.PICKUP: 0,
                Event.DEAD: 0,
                Event.OBJECTIVE: 1
            },
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
    SupportedEnvironments.APPLE_PICKING_4: {
        ENV_KWARGS_KEY: {
            'num_passengers': 4,
            'pickup_only': True,
            'max_steps': 130,
            'reward_table': {
                Event.STEP: 0,
                Event.PICKUP: 0,
                Event.DEAD: 0,
                Event.OBJECTIVE: 1
            },
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
    SupportedEnvironments.APPLE_PICKING_5: {
        ENV_KWARGS_KEY: {
            'num_passengers': 5,
            'pickup_only': True,
            'max_steps': 140,
            'reward_table': {
                Event.STEP: 0,
                Event.PICKUP: 0,
                Event.DEAD: 0,
                Event.OBJECTIVE: 1
            },
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
    SupportedEnvironments.SMALL_PD: {
        ENV_KWARGS_KEY: {
            'num_passengers': 1,
            'no_intermediate_dropoff': True,
            'max_steps': 50,
            'domain_map': [
                "+-------+",
                "| : | : |",
                "| : : : |",
                "| : | | |",
                "| | | : |",
                "+-------+"
            ],
            'reward_table': {
                Event.STEP: 0,
                Event.PICKUP: 0,
                Event.FINAL_DROPOFF: 0,
                Event.INTERMEDIATE_DROPOFF: 0,
                Event.DEAD: 0,
                Event.OBJECTIVE: 1
            },
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
    SupportedEnvironments.PD_1: {
        ENV_KWARGS_KEY: {
            'num_passengers': 1,
            'no_intermediate_dropoff': True,
            'max_steps': 180,
            'reward_table': {
                Event.STEP: 0,
                Event.PICKUP: 0,
                Event.FINAL_DROPOFF: 0,
                Event.INTERMEDIATE_DROPOFF: 0,
                Event.DEAD: 0,
                Event.OBJECTIVE: 1
            },
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
    SupportedEnvironments.PD_2: {
        ENV_KWARGS_KEY: {
            'num_passengers': 2,
            'no_intermediate_dropoff': True,
            'max_steps': 200,
            'reward_table': {
                Event.STEP: 0,
                Event.PICKUP: 0,
                Event.FINAL_DROPOFF: 0,
                Event.INTERMEDIATE_DROPOFF: 0,
                Event.DEAD: 0,
                Event.OBJECTIVE: 1
            },
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
    SupportedEnvironments.PD_3: {
        ENV_KWARGS_KEY: {
            'num_passengers': 3,
            'no_intermediate_dropoff': True,
            'max_steps': 220,
            'reward_table': {
                Event.STEP: 0,
                Event.PICKUP: 0,
                Event.FINAL_DROPOFF: 0,
                Event.INTERMEDIATE_DROPOFF: 0,
                Event.DEAD: 0,
                Event.OBJECTIVE: 1
            },
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
    SupportedEnvironments.PD_4: {
        ENV_KWARGS_KEY: {
            'num_passengers': 4,
            'no_intermediate_dropoff': True,
            'max_steps': 240,
            'reward_table': {
                Event.STEP: 0,
                Event.PICKUP: 0,
                Event.FINAL_DROPOFF: 0,
                Event.INTERMEDIATE_DROPOFF: 0,
                Event.DEAD: 0,
                Event.OBJECTIVE: 1
            },
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
    SupportedEnvironments.PD_5: {
        ENV_KWARGS_KEY: {
            'num_passengers': 5,
            'no_intermediate_dropoff': True,
            'max_steps': 260,
            'reward_table': {
                Event.STEP: 0,
                Event.PICKUP: 0,
                Event.FINAL_DROPOFF: 0,
                Event.INTERMEDIATE_DROPOFF: 0,
                Event.DEAD: 0,
                Event.OBJECTIVE: 1
            },
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
}


class Mods(Enum):
    # context representations
    OHE = 'OHE'
    HCV = 'HCV'

    # RM modifications
    AS = 'AS'
    NDS = 'NDS'
    RS = 'RS'
    GECO = 'GECO'
    GECOUPT = 'GECOUPT'


class Algos(Enum):
    DQN = 'DQN'
    A2C = 'A2C'
    # DDPG = 'DDPG'
    PPO = 'PPO'
    # SAC = 'SAC'


OFF_POLICY_ALGOS = [Algos.DQN]
ON_POLICY_ALGOS = [Algos.PPO]

ALL_ENUM_CFGS = [
    SupportedExperiments,
    SupportedEnvironments,
    ContextSpaces,
    Mods,
    Algos
]

NAME_VALUE_SEP = '-'
MULTI_VAL_SEP = ','
CFG_VALS_SEP = os.path.sep


class ExperimentConfiguration:
    def __init__(self, env: SupportedEnvironments, cspace: ContextSpaces, alg: Algos, mods: Iterable[Mods],
                 exp_kwargs: dict, rm_kwargs: dict, model_kwargs: dict, alg_kwargs: dict, num_src_samples: int,
                 num_tgt_samples: int, max_timesteps: int, eval_freq: int, n_eval_episodes: int,
                 max_no_improvement_evals: int, min_timesteps: int, seed: int):
        self.env = env
        self.cspace = cspace
        self.alg = alg
        self.mods = list(mods)
        self.exp_kwargs = exp_kwargs
        self.rm_kwargs = rm_kwargs
        self.model_kwargs = model_kwargs
        self.alg_kwargs = alg_kwargs
        self.num_src_samples = num_src_samples
        self.num_tgt_samples = num_tgt_samples
        self.max_timesteps = max_timesteps
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_timesteps = min_timesteps
        self.seed = seed

    @property
    def env_name(self):
        return self.env.value

    @property
    def cspace_name(self):
        return self.cspace.value

    @property
    def all_kwargs(self):
        return dict(**self.rm_kwargs, **self.model_kwargs, **self.alg_kwargs)

    def __contains__(self, item):
        return (item in self.mods or  # item is a contained modification
                item in [self.env, self.cspace, self.alg] or  # item is one of the single values
                item in self.all_kwargs.items())

    def __str__(self):
        return pprint.pformat(vars(self), indent=2)

    def __repr__(self):
        return CFG_VALS_SEP.join(self.__repr_value(n, v) for n, v in vars(self).items())

    def __repr_value(self, name, value):
        rv = f'{name}{NAME_VALUE_SEP}'
        if isinstance(value, Enum):
            return self.__repr_value(name, value.value)
        elif callable(value):
            if hasattr(value, '__name__'):
                return rv + value.__name__
            else:
                return rv + repr(value)
        elif isinstance(value, dict):
            return rv + '(' + MULTI_VAL_SEP.join(f'({self.__repr_value(k, v)})' for k, v in value.items()) + ')'
        elif isinstance(value, list) or isinstance(value, tuple) or isinstance(value, set):
            return rv + '(' + MULTI_VAL_SEP.join(self.__repr_value(str(i), v).split("-", 1)[-1]
                                                 for i, v in enumerate(value)) + ')'
        else:
            return rv + str(value)

    @classmethod
    def from_repr_value(cls, repr_val):
        kwargs = {}
        for v in repr_val.split(CFG_VALS_SEP):
            cls.__get_repr_val_inner_value(v, kwargs)

        return cls(**kwargs)

    @classmethod
    def load_all_configurations_in_path(cls, path):
        cfg_reprs = []
        for root, dirs, files in os.walk(path):
            if LOGS_DIR in dirs:
                cfg_reprs.append(root.replace(str(path) + '/', '', 1))
            if MODELS_DIR in root or LOGS_DIR in root:
                continue
        return [cls.from_repr_value(repr_val) for repr_val in cfg_reprs]

    @classmethod
    def __get_repr_val_inner_value(cls, repr_val, kwargs):
        name, value = repr_val.split(NAME_VALUE_SEP, 1)

        if value == '()':
            kwargs[name] = {}
        elif value.startswith('(('):
            inner_kwargs = {}
            dict_vals = value[2:-2]
            for v in dict_vals.split(f'){MULTI_VAL_SEP}('):
                cls.__get_repr_val_inner_value(v, inner_kwargs)
            kwargs[name] = inner_kwargs
        elif value.startswith('('):
            inner_list = []
            list_vals = value[1:-1]
            for v in list_vals.split(MULTI_VAL_SEP):
                inner_list.append(cls.__get_repr_concrete_value(v, ))
            kwargs[name] = inner_list
        else:
            kwargs[name] = cls.__get_repr_concrete_value(value)

    @staticmethod
    def __get_repr_concrete_value(repr_val):
        try:
            return eval(repr_val)
        except:
            pass

        for enum_cfg in ALL_ENUM_CFGS:
            try:
                return enum_cfg(repr_val)
            except ValueError:
                pass

        return repr_val  # return as string
