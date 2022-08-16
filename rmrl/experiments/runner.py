import math
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Lock
from typing import List

from tqdm.auto import tqdm

from .configurations import *
from .cv_transfer import CVTransferExperiment
from .no_transfer import NoTransferExperiment
from .with_transfer import WithTransferExperiment

TIMESTAMP_FORMAT = '%Y-%m-%d-%H_%M_%S.%f'

EXP_TO_FNS = {
    SupportedExperiments.NO_TRANSFER: NoTransferExperiment,
    SupportedExperiments.WITH_TRANSFER: WithTransferExperiment,
    SupportedExperiments.CV_TRANSFER: CVTransferExperiment
}

pbar_lock = Lock()


class ExperimentsRunner:
    def __init__(self, experiments: List[SupportedExperiments], cfgs: List[ExperimentConfiguration], total_timesteps,
                 log_interval, n_eval_episodes, eval_freq, max_no_improvement_evals, min_evals, chkp_freq, num_workers,
                 verbose, force_retrain):
        self.experiments = experiments
        self.cfgs = cfgs
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.chkp_freq = chkp_freq
        self.num_workers = num_workers
        self.verbose = int(verbose)  # assert int input for sb3
        self.force_retrain = force_retrain

    def run(self):
        start = time.time()

        exp_classes = [EXP_TO_FNS[exp_label] for exp_label in self.experiments]
        exp_objects = [exp_class(cfg, self.total_timesteps, self.log_interval, self.n_eval_episodes, self.eval_freq,
                                 self.max_no_improvement_evals, self.min_evals, chkp_freq=self.chkp_freq,
                                 dump_dir=EXPERIMENTS_DUMPS_DIR, verbose=self.verbose,
                                 force_retrain=self.force_retrain)
                       for exp_class in exp_classes
                       for cfg in self.cfgs]

        if self.num_workers > 1:
            self._run_multiprocess(exp_objects)
        else:
            self._run(exp_objects)

        end = time.time()
        print(f'time to run all experiments: {end - start}')

    def _run_multiprocess(self, exp_objects):
        with ThreadPoolExecutor(self.num_workers) as executor:
            list(tqdm(iterable=executor.map(self.run_exp_with_args, exp_objects),
                      total=self.num_runs,
                      desc='all experiments'))

    def _run(self, exp_objects):
        for exp in tqdm(exp_objects, total=self.num_runs, desc='all experiments'):
            self.run_exp_with_args(exp)

    @staticmethod
    def run_exp_with_args(exp):
        try:
            c_src, c_tgt = exp.load_or_sample_contexts()
            if exp.label == SupportedExperiments.NO_TRANSFER:
                exp.run(c_tgt)
            elif exp.label == SupportedExperiments.WITH_TRANSFER:
                exp.run(c_src, c_tgt)
            elif exp.label == SupportedExperiments.CV_TRANSFER:
                exp.run(c_src + c_tgt)
            else:
                raise NotImplementedError(f'unsupported experiment label {exp.label.value}')
        except:
            open(exp.exp_dump_dir / 'FAIL', 'w').close()
            raise

    @property
    def num_runs(self):
        return math.prod(map(len, [self.experiments, self.cfgs]))
