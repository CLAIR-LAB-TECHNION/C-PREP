import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Type

import stable_baselines3 as sb3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from rmrl.nn.models import RMFeatureExtractorSB
from rmrl.reward_machines.potential_functions import ValueIteration
from rmrl.reward_machines.rm_env import RMEnvWrapper
from rmrl.utils.callbacks import TrueRewardRMEnvCallback
from .configurations import *

DEFAULT_RS_GAMMA = 0.9
DEFAULT_POT_FN = ValueIteration()
DEFAULT_TOTAL_TIMESTEPS = 2e6

MODELS_DIR = 'models'
LOGS_DIR = 'logs'
TB_LOG_DIR = 'tensorboard'
EVAL_LOG_DIR = 'eval'


class Experiment(ABC):
    def __init__(self, cfg: ExperimentConfiguration, log_interval=4, n_eval_episodes=5, eval_freq=1000, dump_dir=None,
                 verbose=0):
        self.cfg = cfg
        self.log_interval = log_interval
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.dump_dir = Path(dump_dir or '.')
        self.verbose = verbose

        # save cfg as experiment name
        self.exp_name = f'{self.__class__.__name__}/{repr(cfg)}'

        # extract special kwargs
        self.rs_gamma = cfg.rm_kwargs.pop('rs_gamma', DEFAULT_RS_GAMMA)
        self.pot_fn = cfg.rm_kwargs.pop('pot_fn', DEFAULT_POT_FN)
        self.total_timesteps = cfg.alg_kwargs.pop('total_timesteps', DEFAULT_TOTAL_TIMESTEPS)

        # get algorithm class
        self.alg_class: Type[BaseAlgorithm] = getattr(sb3, cfg.alg.value)

        # get env and RM functions
        fns_dict = RMENV_DICT[self.cfg.env][CONTEXT_SPACES_KEY][self.cfg.cspace]
        self.env_fn = fns_dict[ENV_KEY]
        self.rm_fn = fns_dict[RM_KEY]

    def run(self, *contexts):
        envs = []
        eval_envs = []
        for context in contexts:
            # create two identical envs for training and eval
            env = self.get_env_for_context(context)
            eval_env = self.get_env_for_context(context)

            # convert env to RM env and save
            rm_env = self.env_to_rm_env(env)
            envs.append(rm_env)

            # convert evaluation env to RM env and save
            # eval_env = Monitor(eval_env)  # eval env not automatically wrapped with monitor
            rm_eval_env = self.env_to_rm_env(eval_env, is_eval=True)
            eval_envs.append(rm_eval_env)

        start = time.time()
        self._run(envs, eval_envs)
        end = time.time()
        print(f'execution time: {end - start}; experiment: {self.exp_name}')

    @abstractmethod
    def _run(self, envs: List[RMEnvWrapper], eval_envs: List[RMEnvWrapper]):
        pass

    def get_experiment_env(self):
        return self.env_fn(**self.cfg.env_kwargs)

    def get_env_for_context(self, context):
        env = self.get_experiment_env()
        env.reset()  # make sure env is fully initialized
        env.set_task(context)

        return env

    def env_to_rm_env(self, env, is_eval=False):
        # create RM and reshape rewards
        rm = self.rm_fn(env, **self.cfg.rm_kwargs)
        pots = self.pot_fn(rm, self.rs_gamma)
        rm.reshape_rewards(pots, self.rs_gamma)

        # init env with RM support
        rm_env = RMEnvWrapper(env=env,
                              rm=rm,
                              rm_observations=Mods.GECO in self.cfg,
                              use_rm_reward=Mods.RS in self.cfg and not is_eval)  # use orig rewards for eval

        return rm_env

    def get_agent_for_env(self, env, eval_env=None):
        try:
            agent = self.load_agent_for_env(env)
        except FileNotFoundError:
            agent = self.train_agent_for_env(env, eval_env)

        return agent

    def load_agent_for_env(self, env):
        return BaseAlgorithm.load(self.models_dir / str(env.task))

    def train_agent_for_env(self, env, eval_env):
        task_name = str(env.task)

        policy_kwargs = dict(
            features_extractor_class=RMFeatureExtractorSB,
            features_extractor_kwargs=dict(embed_cur_state=Mods.AS in self.cfg, **self.cfg.model_kwargs)
        )

        agent = self.alg_class(
            env=env,
            policy='MultiInputPolicy',
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.tb_log_dir / task_name,
            verbose=self.verbose,
            seed=self.cfg.seed,
            **self.cfg.alg_kwargs
        )

        return self.train_agent(agent, eval_env, task_name=task_name)

    def train_agent(self, agent, eval_env, task_name):
        # init callbacks for learning
        true_reward_callback = TrueRewardRMEnvCallback()  # log the original reward (not RM reward)
        # eval_callback = EvalCallback(eval_env,
        #                              n_eval_episodes=self.n_eval_episodes,
        #                              eval_freq=self.eval_freq,
        #                              log_path=self.eval_log_dir / task_name,
        #                              best_model_save_path=self.models_dir / task_name,
        #                              verbose=self.verbose)
        #
        # cb = CallbackList([
        #     true_reward_callback,
        #     eval_callback
        # ])

        # train agent
        return agent.learn(
            # total_timesteps=self.total_timesteps,
            #
            # callback=cb,
            # log_interval=self.log_interval,
            # tb_log_name=task_name,

            total_timesteps=self.total_timesteps,
            callback=true_reward_callback,
            log_interval=self.log_interval,
            eval_env=eval_env,
            eval_freq=self.eval_freq,
            n_eval_episodes=self.n_eval_episodes,
            tb_log_name=task_name,
            eval_log_path=self.eval_log_dir / task_name,
        )

    @property
    def exp_dump_dir(self):
        return self.dump_dir / self.exp_name

    @property
    def models_dir(self):
        return self.exp_dump_dir / MODELS_DIR

    @property
    def logs_dir(self):
        return self.exp_dump_dir / LOGS_DIR

    @property
    def tb_log_dir(self):
        return self.logs_dir / TB_LOG_DIR

    @property
    def eval_log_dir(self):
        return self.logs_dir / EVAL_LOG_DIR
