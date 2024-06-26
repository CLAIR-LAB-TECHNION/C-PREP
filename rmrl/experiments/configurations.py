import os
import pprint
from enum import Enum
from pathlib import Path
from typing import Iterable
from torch_geometric.nn import GATConv, GINEConv

from rmrl.envs.grid.reward_machines.single_taxi import TaxiEnvRM
from rmrl.envs.grid.single_taxi import fixed_entities_env, changing_map_env, pickup_order_env
from rmrl.envs.mujoco.reward_machines.half_cheetah import LocationRM, LapsRM
from rmrl.envs.mujoco.half_cheetah import location_env, lap_runner_env

from rmrl.nn.models import cur_state_embedding, ignore_state_mean
from rmrl.reward_machines.potential_functions import ValueIteration, DistFromGoal
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
DQN_TARGET_UPDATE_INTERVAL = [10_000]  # align with sb3 default
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
OFF_POLICY_GRADIENT_STEPS = [4]  # default is 1 grad step per env step

LOG_INTERVAL = 1
N_EVAL_EPISODES = 100
EVAL_FREQ = 1000
MAX_NO_IMPROVEMENT_EVALS = 10
MIN_TIMESTEPS = 50_000

# defaults for reward shaping
DEFAULT_RS_GAMMA = 0.9
DEFAULT_POT_FN = DistFromGoal()
DEFAULT_POT_FN_CHEETAH = ValueIteration(neg_results=True)

# CV experiment defaults
DEFAULT_NUM_TGT_SAMPLES_FOR_CV = 1

# default directories
EXPERIMENTS_DUMPS_DIR = Path('experiment_dumps/')
PRETRAINED_GNN_DIR = Path('grpt_model/')

RUNS_DIR = 'runs'
MODELS_DIR = 'models'
LOGS_DIR = 'logs'
TB_LOG_DIR = 'tensorboard'
EVAL_LOG_DIR = 'eval'
SAVED_CONTEXTS_DIR = 'sampled_contexts'
GENERATED_RMS_DIR = 'generated_rms'

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

# saved buffers
BEST_BUFFER_NAME = 'best_buffer'
FINAL_BUFFER_NAME = 'final_buffer'


class SupportedExperiments(Enum):
    NO_TRANSFER = 'NoTransferExperiment'
    WITH_TRANSFER = 'WithTransferExperiment'
    CV_TRANSFER = 'CVTransferExperiment'


class SupportedEnvironments(Enum):
    GN_4X4_1PAS = 'gn_4x4_1pas'
    GN_4X4_2PAS = 'gn_4x4_2pas'
    GN_4X4_3PAS = 'gn_4x4_3pas'

    GN_6X6_1PAS = 'gn_6x6_1pas'
    GN_6X6_2PAS = 'gn_6x6_2pas'
    GN_6X6_3PAS = 'gn_6x6_3pas'
    GN_6X6_4PAS = 'gn_6x6_4pas'

    GN_DEFAULT_1PAS = 'gn_default_1pas'
    GN_DEFAULT_2PAS = 'gn_default_2pas'
    GN_DEFAULT_3PAS = 'gn_default_3pas'
    GN_DEFAULT_4PAS = 'gn_default_4pas'
    GN_DEFAULT_5PAS = 'gn_default_5pas'

    PD_4X4_1PAS = 'pd_4x4_1pas'
    PD_4X4_2PAS = 'pd_4x4_2pas'
    PD_4X4_3PAS = 'pd_4x4_3pas'

    PD_6X6_1PAS = 'pd_6x6_1pas'
    PD_6X6_2PAS = 'pd_6x6_2pas'
    PD_6X6_3PAS = 'pd_6x6_3pas'
    PD_6X6_4PAS = 'pd_6x6_4pas'

    PD_DEFAULT_1PAS = 'pd_default_1pas'
    PD_DEFAULT_2PAS = 'pd_default_2pas'
    PD_DEFAULT_3PAS = 'pd_default_3pas'
    PD_DEFAULT_4PAS = 'pd_default_4pas'
    PD_DEFAULT_5PAS = 'pd_default_5pas'

    PD_12X12_1PAS = 'pd_12x12_1pas'
    PD_12X12_2PAS = 'pd_12x12_2pas'
    PD_12X12_3PAS = 'pd_12x12_3pas'
    PD_12X12_4PAS = 'pd_12x12_4pas'
    PD_12X12_5PAS = 'pd_12x12_5pas'

    PD_6X6_1PAS_STOCH = 'pd_6x6_1pas_stoch'
    PD_6X6_2PAS_STOCH = 'pd_6x6_2pas_stoch'
    PD_8X8_1PAS_STOCH = 'pd_8x8_1pas_stoch'
    PD_8X8_2PAS_STOCH = 'pd_8x8_2pas_stoch'
    PD_12X12_1PAS_STOCH = 'pd_12x12_1pas_stoch'

    PO_4X4_5PAS = 'po_4x4_5pas'
    PO_6X6_5PAS = 'po_6x6_5pas'
    PO_DEFAULT_10PAS = 'po_default_10pas'

    CHEETAH_LOC = 'cheetah_loc'
    CHEETAH_LAP = 'cheetah_lap'


class ContextSpaces(Enum):
    FIXED_ENTITIES = 'fixed_entities'
    CHANGING_MAP = 'changing_map'
    PICKUP_ORDER = 'pickup_order'

    CHANGING_LOC = 'changing_loc'


RMENV_DICT = {
    # pickup only environments
    SupportedEnvironments.GN_4X4_1PAS: {
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
            },
        }
    },
    SupportedEnvironments.GN_4X4_2PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 2,
            'pickup_only': True,
            'max_steps': 30,
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
    SupportedEnvironments.GN_4X4_3PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 3,
            'pickup_only': True,
            'max_steps': 35,
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
    SupportedEnvironments.GN_6X6_1PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 1,
            'pickup_only': True,
            'max_steps': 35,
            'domain_map': [
                "+-----------+",
                "| | | : | : |",
                "| | : : : : |",
                "| | | : | : |",
                "| : : : | : |",
                "| : : | : | |",
                "| | : : : | |",
                "+-----------+"
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
    SupportedEnvironments.GN_6X6_2PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 2,
            'pickup_only': True,
            'max_steps': 40,
            'domain_map': [
                "+-----------+",
                "| | | : | : |",
                "| | : : : : |",
                "| | | : | : |",
                "| : : : | : |",
                "| : : | : | |",
                "| | : : : | |",
                "+-----------+"
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
    SupportedEnvironments.GN_6X6_3PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 3,
            'pickup_only': True,
            'max_steps': 45,
            'domain_map': [
                "+-----------+",
                "| | | : | : |",
                "| | : : : : |",
                "| | | : | : |",
                "| : : : | : |",
                "| : : | : | |",
                "| | : : : | |",
                "+-----------+"
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
    SupportedEnvironments.GN_6X6_4PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 4,
            'pickup_only': True,
            'max_steps': 50,
            'domain_map': [
                "+-----------+",
                "| | | : | : |",
                "| | : : : : |",
                "| | | : | : |",
                "| : : : | : |",
                "| : : | : | |",
                "| | : : : | |",
                "+-----------+"
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
    SupportedEnvironments.GN_DEFAULT_1PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 1,
            'pickup_only': True,
            'max_steps': 50,
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
    SupportedEnvironments.GN_DEFAULT_2PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 2,
            'pickup_only': True,
            'max_steps': 55,
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
    SupportedEnvironments.GN_DEFAULT_3PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 3,
            'pickup_only': True,
            'max_steps': 60,
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
    SupportedEnvironments.GN_DEFAULT_4PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 4,
            'pickup_only': True,
            'max_steps': 65,
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
    SupportedEnvironments.GN_DEFAULT_5PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 5,
            'pickup_only': True,
            'max_steps': 70,
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

    # pickup and dropoff environments
    SupportedEnvironments.PD_4X4_1PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 1,
            'no_intermediate_dropoff': True,
            'max_steps': 35,
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
    SupportedEnvironments.PD_4X4_2PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 2,
            'no_intermediate_dropoff': True,
            'max_steps': 40,
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
    SupportedEnvironments.PD_4X4_3PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 3,
            'no_intermediate_dropoff': True,
            'max_steps': 45,
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
    SupportedEnvironments.PD_6X6_1PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 1,
            'no_intermediate_dropoff': True,
            'max_steps': 70,
            'domain_map': [
                "+-----------+",
                "| | | : | : |",
                "| | : : : : |",
                "| | | : | : |",
                "| : : : | : |",
                "| : : | : | |",
                "| | : : : | |",
                "+-----------+"
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
    SupportedEnvironments.PD_6X6_2PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 2,
            'no_intermediate_dropoff': True,
            'max_steps': 75,
            'domain_map': [
                "+-----------+",
                "| | | : | : |",
                "| | : : : : |",
                "| | | : | : |",
                "| : : : | : |",
                "| : : | : | |",
                "| | : : : | |",
                "+-----------+"
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
    SupportedEnvironments.PD_6X6_3PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 3,
            'no_intermediate_dropoff': True,
            'max_steps': 80,
            'domain_map': [
                "+-----------+",
                "| | | : | : |",
                "| | : : : : |",
                "| | | : | : |",
                "| : : : | : |",
                "| : : | : | |",
                "| | : : : | |",
                "+-----------+"
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
    SupportedEnvironments.PD_6X6_4PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 4,
            'no_intermediate_dropoff': True,
            'max_steps': 85,
            'domain_map': [
                "+-----------+",
                "| | | : | : |",
                "| | : : : : |",
                "| | | : | : |",
                "| : : : | : |",
                "| : : | : | |",
                "| | : : : | |",
                "+-----------+"
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
    SupportedEnvironments.PD_DEFAULT_1PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 1,
            'no_intermediate_dropoff': True,
            'max_steps': 170,
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
    SupportedEnvironments.PD_DEFAULT_2PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 2,
            'no_intermediate_dropoff': True,
            'max_steps': 175,
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
    SupportedEnvironments.PD_DEFAULT_3PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 3,
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
    SupportedEnvironments.PD_DEFAULT_4PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 4,
            'no_intermediate_dropoff': True,
            'max_steps': 185,
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
    SupportedEnvironments.PD_DEFAULT_5PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 5,
            'no_intermediate_dropoff': True,
            'max_steps': 190,
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
    SupportedEnvironments.PD_12X12_1PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 1,
            'no_intermediate_dropoff': True,
            'max_steps': 500,
            'domain_map': [
                "+-----------------------+",
                "| : | : | : : : : : | : |",
                "| | : : : : : : : | | : |",
                "| | | : | : : : : | : : |",
                "| : : : | : : | : | : | |",
                "| : | : : | : : : | : | |",
                "| : | : : | : : : : | : |",
                "| : | | : : : | : : | : |",
                "| : : | : | : | : : | : |",
                "| : : : : | : : : : | : |",
                "| : | : : : : | : : | | |",
                "| : | : : | : | : : : | |",
                "| : | : : | : | : : : | |",
                "+-----------------------+"
            ],
            'reward_table': {
                e: 1 if e == Event.OBJECTIVE else 0
                for e in Event
            }
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
    SupportedEnvironments.PD_12X12_1PAS_STOCH: {
        ENV_KWARGS_KEY: {
            'num_passengers': 1,
            'no_intermediate_dropoff': True,
            'max_steps': 500,
            'domain_map': [
                "+-----------------------+",
                "| : | : | : : : : : | : |",
                "| | : : : : : : : | | : |",
                "| | | : | : : : : | : : |",
                "| : : : | : : | : | : | |",
                "| : | : : | : : : | : | |",
                "| : | : : | : : : : | : |",
                "| : | | : : : | : : | : |",
                "| : : | : | : | : : | : |",
                "| : : : : | : : : : | : |",
                "| : | : : : : | : : | | |",
                "| : | : : | : | : : : | |",
                "| : | : : | : | : : : | |",
                "+-----------------------+"
            ],
            'reward_table': {
                e: 1 if e == Event.OBJECTIVE else 0
                for e in Event
            },
            'stochastic_actions': {
                'north': {
                    'north': 0.9,
                    'east': 0.05,
                    'west': 0.05
                },
                'south': {
                    'south': 0.9,
                    'east': 0.05,
                    'west': 0.05
                },
                'east': {
                    'east': 0.9,
                    'north': 0.05,
                    'south': 0.05
                },
                'west': {
                    'west': 0.9,
                    'north': 0.05,
                    'south': 0.05
                },
            }
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
    SupportedEnvironments.PD_8X8_1PAS_STOCH: {
        ENV_KWARGS_KEY: {
            'num_passengers': 1,
            'no_intermediate_dropoff': True,
            'max_steps': 500,
            'domain_map': [
                "+---------------+",
                "| : | : | : : : |",
                "| | : : : : : : |",
                "| | | : | : : : |",
                "| : : : | : : | |",
                "| : | : : | : : |",
                "| : | : : | : : |",
                "| : | | : : : | |",
                "| : : | : | : | |",
                "+---------------+"
            ],
            'reward_table': {
                e: 1 if e == Event.OBJECTIVE else 0
                for e in Event
            },
            'stochastic_actions': {
                'north': {
                    'north': 0.9,
                    'east': 0.05,
                    'west': 0.05
                },
                'south': {
                    'south': 0.9,
                    'east': 0.05,
                    'west': 0.05
                },
                'east': {
                    'east': 0.9,
                    'north': 0.05,
                    'south': 0.05
                },
                'west': {
                    'west': 0.9,
                    'north': 0.05,
                    'south': 0.05
                },
            }
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
    SupportedEnvironments.PD_8X8_2PAS_STOCH: {
        ENV_KWARGS_KEY: {
            'num_passengers': 2,
            'no_intermediate_dropoff': True,
            'max_steps': 500,
            'domain_map': [
                "+---------------+",
                "| : | : | : : : |",
                "| | : : : : : : |",
                "| | | : | : : : |",
                "| : : : | : : | |",
                "| : | : : | : : |",
                "| : | : : | : : |",
                "| : | | : : : | |",
                "| : : | : | : | |",
                "+---------------+"
            ],
            'reward_table': {
                e: 1 if e == Event.OBJECTIVE else 0
                for e in Event
            },
            'stochastic_actions': {
                'north': {
                    'north': 0.9,
                    'east': 0.05,
                    'west': 0.05
                },
                'south': {
                    'south': 0.9,
                    'east': 0.05,
                    'west': 0.05
                },
                'east': {
                    'east': 0.9,
                    'north': 0.05,
                    'south': 0.05
                },
                'west': {
                    'west': 0.9,
                    'north': 0.05,
                    'south': 0.05
                },
            }
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
    SupportedEnvironments.PD_6X6_1PAS_STOCH: {
        ENV_KWARGS_KEY: {
            'num_passengers': 1,
            'no_intermediate_dropoff': True,
            'max_steps': 1000,
            'domain_map': [
                "+-----------+",
                "| | | : | : |",
                "| | : : : : |",
                "| | | : | : |",
                "| : : : | : |",
                "| : : | : | |",
                "| | : : : | |",
                "+-----------+"
            ],
            'reward_table': {
                e: 1 if e == Event.OBJECTIVE else 0
                for e in Event
            },
            'stochastic_actions': {
                'north': {
                    'north': 0.9,
                    'east': 0.05,
                    'west': 0.05
                },
                'south': {
                    'south': 0.9,
                    'east': 0.05,
                    'west': 0.05
                },
                'east': {
                    'east': 0.9,
                    'north': 0.05,
                    'south': 0.05
                },
                'west': {
                    'west': 0.9,
                    'north': 0.05,
                    'south': 0.05
                },
            }
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
    SupportedEnvironments.PD_6X6_2PAS_STOCH: {
        ENV_KWARGS_KEY: {
            'num_passengers': 2,
            'no_intermediate_dropoff': True,
            'max_steps': 1000,
            'domain_map': [
                "+-----------+",
                "| | | : | : |",
                "| | : : : : |",
                "| | | : | : |",
                "| : : : | : |",
                "| : : | : | |",
                "| | : : : | |",
                "+-----------+"
            ],
            'reward_table': {
                e: 1 if e == Event.OBJECTIVE else 0
                for e in Event
            },
            'stochastic_actions': {
                'north': {
                    'north': 0.9,
                    'east': 0.05,
                    'west': 0.05
                },
                'south': {
                    'south': 0.9,
                    'east': 0.05,
                    'west': 0.05
                },
                'east': {
                    'east': 0.9,
                    'north': 0.05,
                    'south': 0.05
                },
                'west': {
                    'west': 0.9,
                    'north': 0.05,
                    'south': 0.05
                },
            }
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

    # pickup order environments
    SupportedEnvironments.PO_4X4_5PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 5,
            'pickup_only': True,
            'pickup_order': [0, 1, 2, 3, 4],
            'max_steps': 100,
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
            ContextSpaces.PICKUP_ORDER: {
                ENV_KEY: pickup_order_env,
                RM_KEY: TaxiEnvRM,
            },
        }
    },
    SupportedEnvironments.PO_6X6_5PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 5,
            'pickup_only': True,
            'pickup_order': [0, 1, 2, 3, 4],
            'max_steps': 225,
            'domain_map': [
                "+-----------+",
                "| | | : | : |",
                "| | : : : : |",
                "| | | : | : |",
                "| : : : | : |",
                "| : : | : | |",
                "| | : : : | |",
                "+-----------+"
            ],
            'reward_table': {
                Event.STEP: 0,
                Event.PICKUP: 0,
                Event.DEAD: 0,
                Event.OBJECTIVE: 1
            },
        },
        CONTEXT_SPACES_KEY: {
            ContextSpaces.PICKUP_ORDER: {
                ENV_KEY: pickup_order_env,
                RM_KEY: TaxiEnvRM,
            },
        }
    },
    SupportedEnvironments.PO_DEFAULT_10PAS: {
        ENV_KWARGS_KEY: {
            'num_passengers': 10,
            'pickup_only': True,
            'pickup_order': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'max_steps': 700,
            'reward_table': {
                e: 1 if e == Event.OBJECTIVE else 0
                for e in Event
            },
        },
        CONTEXT_SPACES_KEY: {
            ContextSpaces.PICKUP_ORDER: {
                ENV_KEY: pickup_order_env,
                RM_KEY: TaxiEnvRM,
            },
        }
    },

    # cheetah
    SupportedEnvironments.CHEETAH_LOC: {
        ENV_KWARGS_KEY: {},
        CONTEXT_SPACES_KEY: {
            ContextSpaces.CHANGING_LOC: {
                ENV_KEY: location_env,
                RM_KEY: LocationRM,
            },
        }
    },
    SupportedEnvironments.CHEETAH_LAP: {
        ENV_KWARGS_KEY: {},
        CONTEXT_SPACES_KEY: {
            ContextSpaces.CHANGING_LOC: {
                ENV_KEY: lap_runner_env,
                RM_KEY: LapsRM,
            },
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

    @classmethod
    def has_rm_mod(cls, mods):
        return any(m in [cls.AS, cls.NDS, cls.RS, cls.GECO, cls.GECOUPT] for m in mods)


class Algos(Enum):
    DQN = 'DQN'
    A2C = 'A2C'
    DDPG = 'DDPG'
    PPO = 'PPO'
    SAC = 'SAC'


OFF_POLICY_ALGOS = [Algos.DQN, Algos.SAC, Algos.DDPG]
ON_POLICY_ALGOS = [Algos.PPO, Algos.A2C]

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
        self.mods = sorted(mods, key=lambda m: m.value)  # sorted for identical experiment consistency
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
        return CFG_VALS_SEP.join(self.repr_value(n, v) for n, v in vars(self).items())

    def repr_value(self, name, value):
        rv = f'{name}{NAME_VALUE_SEP}'
        if isinstance(value, Enum):
            return self.repr_value(name, value.value)
        elif callable(value):
            if hasattr(value, '__name__'):
                return rv + value.__name__
            else:
                return rv + repr(value)
        elif isinstance(value, dict):
            return rv + '(' + MULTI_VAL_SEP.join(f'({self.repr_value(k, v)})' for k, v in value.items()) + ')'
        elif isinstance(value, list) or isinstance(value, tuple) or isinstance(value, set):
            return rv + '(' + MULTI_VAL_SEP.join(self.repr_value(str(i), v).split("-", 1)[-1]
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


class TransferConfiguration(ExperimentConfiguration):
    def __init__(self, env: SupportedEnvironments, cspace: ContextSpaces, alg: Algos, mods: Iterable[Mods],
                 exp_kwargs: dict, rm_kwargs: dict, model_kwargs: dict, alg_kwargs: dict, num_src_samples: int,
                 num_tgt_samples: int, max_timesteps: int, eval_freq: int, n_eval_episodes: int,
                 max_no_improvement_evals: int, min_timesteps: int, seed: int, tsf_kwargs: dict):
        super().__init__(env, cspace, alg, mods, exp_kwargs, rm_kwargs, model_kwargs, alg_kwargs, num_src_samples,
                         num_tgt_samples, max_timesteps, eval_freq, n_eval_episodes, max_no_improvement_evals,
                         min_timesteps, seed)
        self.tsf_kwargs = tsf_kwargs

    def get_src_config(self):
        kwargs = vars(self).copy()
        kwargs.pop('tsf_kwargs')
        return ExperimentConfiguration(**kwargs)
