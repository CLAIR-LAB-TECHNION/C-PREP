import math
import pickle
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from multiprocessing import Lock
from typing import List

from tqdm.auto import tqdm

from .configurations import *
from .cv_transfer import CVTransferExperiment
from .experiment import Experiment
from .no_transfer import NoTransferExperiment
from .with_transfer import WithTransferExperiment

TIMESTAMP_FORMAT = '%Y-%m-%d-%H_%M_%S.%f'

EXPERIMENTS_DUMPS_DIR = Path('experiment_dumps')
CONTEXTS_DIR_NAME = Path('sampled_contexts')

EXP_TO_FNS = {
    SupportedExperiments.NO_TRANSFER: NoTransferExperiment,
    SupportedExperiments.WITH_TRANSFER: WithTransferExperiment,
    SupportedExperiments.CV_TRANSFER: CVTransferExperiment
}

pbar_lock = Lock()


class ExperimentsRunner:
    def __init__(self, experiments: List[SupportedExperiments], cfgs: List[ExperimentConfiguration], total_timesteps,
                 log_interval, n_eval_episodes, eval_freq, max_no_improvement_evals, min_evals,
                 sample_seed, num_src_samples, num_tgt_samples, num_workers, verbose):
        self.experiments = experiments
        self.cfgs = cfgs
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.sample_seed = sample_seed
        self.num_src_samples = num_src_samples
        self.num_tgt_samples = num_tgt_samples
        self.num_workers = num_workers
        self.verbose = int(verbose)  # assert int input for sb3

    def run(self):
        start = time.time()

        exp_classes = [EXP_TO_FNS[exp_label] for exp_label in self.experiments]
        exp_objects = [exp_class(cfg, self.total_timesteps, self.log_interval, self.n_eval_episodes, self.eval_freq,
                                 self.max_no_improvement_evals, self.min_evals, dump_dir=EXPERIMENTS_DUMPS_DIR,
                                 verbose=self.verbose)
                       for exp_class in exp_classes
                       for cfg in self.cfgs]

        exp_args, c_src_args, c_tgt_args = self.__get_map_args(exp_objects)

        if self.num_workers > 1:
            self._run_multiprocess(exp_args, c_src_args, c_tgt_args)
        else:
            self._run(exp_args, c_src_args, c_tgt_args)

        end = time.time()
        print(f'time to run all experiments: {end - start}')

    def _run_multiprocess(self, exp_args, c_src_args, c_tgt_args):
        with ThreadPoolExecutor(self.num_workers) as executor:
            list(tqdm(iterable=executor.map(self.run_exp_with_args, exp_args, c_src_args, c_tgt_args),
                      total=self.num_runs,
                      desc='all experiments'))

    def _run(self, exp_args, c_src_args, c_tgt_args):
        for run_args in tqdm(zip(exp_args, c_src_args, c_tgt_args),
                                      total=self.num_runs,
                                      desc='all experiments'):
            self.run_exp_with_args(*run_args)

    @staticmethod
    def run_exp_with_args(exp, c_src, c_tgt):
        if exp.label == SupportedExperiments.NO_TRANSFER:
            exp.run(c_tgt)
        elif exp.label == SupportedExperiments.WITH_TRANSFER:
            exp.run(c_src, c_tgt)
        elif exp.label == SupportedExperiments.CV_TRANSFER:
            exp.run(c_src + c_tgt)
        else:
            raise NotImplementedError(f'unsupported experiment label {exp.label.value}')

    @staticmethod
    def load_or_sample_contexts(exp: Experiment, num_src_samples, num_tgt_samples, sample_seed: int):
        contexts_file = (CONTEXTS_DIR_NAME / exp.cfg.env_name / exp.cfg.cspace_name /
                         f'src_samples={num_src_samples}__tgt_samples={num_tgt_samples}__seed={sample_seed}')
        try:
            with open(contexts_file, 'rb') as f:
                src_contexts, tgt_contexts = pickle.load(f)
        except (FileNotFoundError, EOFError):
            # create env for sampling
            env = exp.get_experiment_env()

            # set seed for constant sampling
            env.seed(sample_seed)

            # sample contexts
            num_samples = num_src_samples + num_tgt_samples
            contexts = env.sample_task(num_samples * OVERSAMPLE_FACTOR)  # oversample
            contexts = list(set(contexts))  # remove duplicates
            contexts = contexts[:num_samples]  # reduce to desired number of

            # check enough contexts
            if len(contexts) < num_samples:
                warnings.warn(f'wanted {num_samples} contexts for env {exp.cfg.env_name} in context. '
                              f'sampled {len(contexts)}')

            src_contexts, tgt_contexts = contexts[:num_src_samples], contexts[num_src_samples:]

            # save contexts
            contexts_file.parent.mkdir(exist_ok=True, parents=True)
            with open(contexts_file, 'wb') as f:
                pickle.dump((src_contexts, tgt_contexts), f)

        return src_contexts, tgt_contexts

    @property
    def num_runs(self):
        return math.prod(map(len, [self.experiments, self.cfgs, self.sample_seed, self.num_src_samples,
                                   self.num_tgt_samples]))

    def __get_map_args(self, experiments):
        return list(zip(*[(exp, *self.load_or_sample_contexts(exp, src_samples, tgt_samples, seed))
                          for exp, src_samples, tgt_samples, seed in product(experiments,
                                                                             self.num_src_samples,
                                                                             self.num_tgt_samples,
                                                                             self.sample_seed)]))
