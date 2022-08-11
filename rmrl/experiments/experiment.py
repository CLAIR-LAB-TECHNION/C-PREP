import pickle
import time
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Type

import stable_baselines3 as sb3
from rmrl.nn.models import RMFeatureExtractorSB
from rmrl.reward_machines.rm_env import RMEnvWrapper
from rmrl.utils.callbacks import TrueRewardRMEnvCallback, ProgressBarCallback
from rmrl.utils.misc import sha3_hash
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from .configurations import *


class Experiment(ABC):
    def __init__(self, cfg: ExperimentConfiguration, total_timesteps=5e5,
                 log_interval=1, n_eval_episodes=100, eval_freq=1000, max_no_improvement_evals=10, min_evals=50,
                 dump_dir=None, verbose=0, ):
        self.cfg = cfg
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.dump_dir = Path(dump_dir or '.')
        self.verbose = verbose

        # save cfg as experiment name
        self.exp_name = f'{self.__class__.__name__}/{repr(cfg)}'

        # extract special kwargs
        self.rs_gamma = cfg.rm_kwargs.pop('rs_gamma', DEFAULT_RS_GAMMA)
        self.pot_fn = cfg.rm_kwargs.pop('pot_fn', DEFAULT_POT_FN)

        # get algorithm class
        self.alg_class: Union[Type[OnPolicyAlgorithm], Type[OffPolicyAlgorithm]] = getattr(sb3, cfg.alg.value)

        # get env and RM functions
        fns_dict = RMENV_DICT[self.cfg.env][CONTEXT_SPACES_KEY][self.cfg.cspace]
        self.env_fn = fns_dict[ENV_KEY]
        self.rm_fn = fns_dict[RM_KEY]

        self.env_kwargs = RMENV_DICT[self.cfg.env][ENV_KWARGS_KEY]

    def run(self, *contexts):
        train_envs = []
        eval_envs = []
        for context_set in contexts:
            # create two identical envs for training and eval
            envs = self.get_envs_per_context_in_set(context_set)

            # convert env to RM env
            rm_envs = [self.env_to_rm_env(env) for env in envs]

            # convert to vec env for parallel training
            rm_vec_env = DummyVecEnv([partial(lambda env: env, env) for env in rm_envs])

            # save training env
            train_envs.append(rm_vec_env)

            # create env for evaluation (no parallel)
            eval_env = self.get_single_env_for_context_set(context_set)
            rm_eval_env = self.env_to_rm_env(eval_env, is_eval=True)
            # rm_eval_env = Monitor(rm_eval_env)  # eval env not automatically wrapped with monitor
            eval_envs.append(rm_eval_env)

        start = time.time()
        self._run(train_envs, eval_envs)
        end = time.time()
        print(f'execution time: {end - start}; experiment: {self.exp_name}')

    @abstractmethod
    def _run(self, envs: List[DummyVecEnv], eval_envs: List[RMEnvWrapper]):
        pass

    def get_experiment_env(self):
        return self.env_fn(**self.env_kwargs)

    def get_env_for_context(self, context):
        env = self.get_experiment_env()
        env.task = context
        env.reset()  # make sure env is fully initialized

        return env

    def get_envs_per_context_in_set(self, context_set):
        return [self.get_env_for_context(c) for c in context_set]

    def get_single_env_for_context_set(self, context_set):
        env = self.env_fn(**self.env_kwargs, change_task_on_reset=True)
        env.set_fixed_contexts(context_set)
        env.reset()

        return env

    def env_to_rm_env(self, env, is_eval=False):
        # create RM and reshape rewards
        rm = self.rm_fn(env, **self.cfg.rm_kwargs)
        pots = self.pot_fn(rm, self.rs_gamma)
        rm.reshape_rewards(pots, self.rs_gamma)

        # init env with RM support
        rm_env = RMEnvWrapper(env=env,
                              rm_fn=lambda e: self.rm_fn(e, **self.cfg.rm_kwargs),
                              rm_observations=Mods.GECO in self.cfg,
                              use_rm_reward=Mods.RS in self.cfg and not is_eval)  # use orig rewards for eval

        rm_env.reset()

        return rm_env

    def new_agent_for_env(self, env):
        num_props = env.envs[0].rm.num_propositions  # all rms should have the same rm
        policy_kwargs = dict(
            features_extractor_class=RMFeatureExtractorSB,
            features_extractor_kwargs=dict(embed_cur_state=Mods.AS in self.cfg,
                                           # TODO more generic call to grpt
                                           pretrained_gnn_path=f'grpt_model/{num_props}_props/gnn'
                                           if Mods.GECOUPT in self.cfg
                                           else None,
                                           **self.cfg.model_kwargs)
        )

        return self.alg_class(
            env=env,
            policy='MultiInputPolicy',
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(self.tb_log_dir),
            verbose=self.verbose,
            seed=self.cfg.seed,
            **self.cfg.alg_kwargs
        )

    def get_agent_for_env(self, env, eval_env):
        try:
            agent = self.load_agent_for_env(env)
        except FileNotFoundError:
            agent = self.train_agent_for_env(env, eval_env)

        return agent

    def load_agent_for_env(self, env):
        task_name = self.get_env_task_name(env)
        return self.load_agent_for_task(task_name, init_env=env)

    def load_agent_for_task(self, task_name, init_env=None):
        loaded_agent = self.alg_class.load(self.models_dir / task_name / 'best_model', init_env)
        print(f'loaded agent for task {task_name}')
        return loaded_agent

    def get_env_task(self, env):
        if isinstance(env, DummyVecEnv):
            return [e.task for e in env.envs]
        else:
            return env.fixed_contexts

    def get_env_task_name(self, env):
        task = tuple(self.get_env_task(env))
        return sha3_hash(task)

    def train_agent_for_env(self, env, eval_env):
        agent = self.new_agent_for_env(env)

        task_name = self.get_env_task_name(env)
        return self.train_agent(agent, eval_env, task_name=task_name)

    def train_agent(self, agent, eval_env, task_name):
        # init callbacks for learning
        true_reward_callback = TrueRewardRMEnvCallback()  # log the original reward (not RM reward)
        pb_callback = ProgressBarCallback()
        early_stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=self.max_no_improvement_evals,
                                                               min_evals=self.min_evals,
                                                               verbose=False)
        eval_callback = EvalCallback(eval_env=Monitor(eval_env),
                                     callback_after_eval=early_stop_callback,
                                     n_eval_episodes=self.n_eval_episodes,
                                     eval_freq=self.eval_freq,
                                     log_path=self.eval_log_dir / task_name,
                                     best_model_save_path=self.models_dir / task_name,
                                     verbose=self.verbose)
        checkpoint_callback = CheckpointCallback(save_freq=self.eval_freq,
                                                 save_path=self.models_dir / task_name / 'checkpoints',
                                                 name_prefix='chkp',
                                                 verbose=self.verbose)

        # train agent
        print(f'training agent for task {task_name}')
        return agent.learn(
            total_timesteps=self.total_timesteps,
            callback=[true_reward_callback, pb_callback, eval_callback, checkpoint_callback],
            log_interval=self.log_interval,
            tb_log_name=task_name,
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

    @property
    def saved_contexts_dir(self):
        return self.dump_dir / SAVED_CONTEXTS_DIR / self.cfg.env_name / self.cfg.cspace_name

    @classmethod
    def load_all_experiments_in_path(cls, path=None):
        if path is None:
            path = EXPERIMENTS_DUMPS_DIR
        path = Path(path)

        cfgs = ExperimentConfiguration.load_all_configurations_in_path(path / cls.__name__)
        exps = [cls(cfg, dump_dir=path) for cfg in cfgs]

        return exps

    @property
    def label(self):
        return SupportedExperiments(self.__class__.__name__)

    def load_or_sample_contexts(self):
        num_src_samples = self.cfg.num_src_samples
        num_tgt_samples = self.cfg.num_tgt_samples
        contexts_file = (self.saved_contexts_dir /
                         f'src_samples={num_src_samples}__'
                         f'tgt_samples={num_tgt_samples}__'
                         f'seed={self.cfg.seed}')
        try:
            with open(contexts_file, 'rb') as f:
                src_contexts, tgt_contexts = pickle.load(f)
        except (FileNotFoundError, EOFError):
            # create env for sampling
            env = self.get_experiment_env()

            # set seed for constant sampling
            env.seed(self.cfg.seed)

            # sample contexts
            num_samples = num_src_samples + num_tgt_samples
            contexts = env.sample_task(num_samples * OVERSAMPLE_FACTOR)  # oversample
            contexts = list(set(contexts))  # remove duplicates
            contexts = contexts[:num_samples]  # reduce to desired number of

            # check enough contexts
            if len(contexts) < num_samples:
                warnings.warn(f'wanted {num_samples} contexts for env {self.cfg.env_name} in context. '
                              f'sampled {len(contexts)}')

            src_contexts, tgt_contexts = contexts[:num_src_samples], contexts[num_src_samples:]

            # save contexts
            contexts_file.parent.mkdir(exist_ok=True, parents=True)
            with open(contexts_file, 'wb') as f:
                pickle.dump((src_contexts, tgt_contexts), f)

        return src_contexts, tgt_contexts
