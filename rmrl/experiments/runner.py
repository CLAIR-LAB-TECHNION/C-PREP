import pickle
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from tqdm.auto import tqdm

from .configurations import *
from .experiment import Experiment
from .no_transfer import NoTransferExperiment
from .with_transfer import WithTransferExperiment
from ..utils.misc import split_pairs

TIMESTAMP_FORMAT = '%Y-%m-%d-%H_%M_%S.%f'

EXPERIMENTS_DUMPS_DIR = Path('experiment_dumps')
CONTEXTS_DIR_NAME = Path('sampled_contexts')

EXP_TO_FNS = {
    SupportedExperiments.NO_TRANSFER: NoTransferExperiment,
    SupportedExperiments.WITH_TRANSFER: WithTransferExperiment
}


def single_thread_executor(fn, *args):
    return fn(*args)


class ExperimentsRunner:
    def __init__(self, experiments: List[SupportedExperiments], cfgs: List[ExperimentConfiguration], sample_seed,
                 num_workers, verbose):
        self.experiments = experiments
        self.cfgs = cfgs
        self.sample_seed = sample_seed
        self.num_workers = num_workers
        self.verbose = int(verbose)  # assert int input for sb3

    def run(self):
        if self.num_workers > 1:
            self._run_multithread()
        else:
            self._run()

    def _run_multithread(self):
        with ThreadPoolExecutor(self.num_workers) as executor:
            self._run(executor=executor.submit)

    def _run(self, executor=single_thread_executor):
        start = time.time()

        # add progressbar
        all_pbar = tqdm(total=len(self.experiments) * len(self.cfgs), desc='all experiments')

        # dump_dir_with_ts = EXPERIMENTS_DUMPS_DIR / datetime.now().strftime(TIMESTAMP_FORMAT)
        for exp_label in self.experiments:  # iterate experiments
            exp_class = EXP_TO_FNS[exp_label]
            for cfg in self.cfgs:  # iterate configurations
                exp = exp_class(cfg, dump_dir=EXPERIMENTS_DUMPS_DIR, verbose=self.verbose)

                contexts = self.load_or_sample_contexts(exp, self.sample_seed)

                # check experiment type to get correct input contexts
                for c_src, c_tgt in tqdm(contexts, desc=f'cur exp'):
                    if exp_label == SupportedExperiments.NO_TRANSFER:
                        executor(exp.run, c_src)
                        executor(exp.run, c_tgt)
                    elif exp_label == SupportedExperiments.WITH_TRANSFER:
                        executor(exp.run, c_src, c_tgt)
                    else:
                        raise NotImplementedError(f'unsupported experiment label {exp_label.value}')
                all_pbar.update()
        all_pbar.close()

        end = time.time()
        print(f'time to run all experiments: {end - start}')

    def load_or_sample_contexts(self, exp: Experiment, sample_seed: int):
        contexts_file = CONTEXTS_DIR_NAME / exp.cfg.env_name / exp.cfg.cspace_name / str(sample_seed)
        try:
            with open(contexts_file, 'rb') as f:
                contexts = pickle.load(f)
        except (FileNotFoundError, EOFError):
            # create env for sampling
            env = exp.get_experiment_env()

            # set seed for constant sampling
            env.seed(sample_seed)

            # sample contexts
            contexts = env.sample_task(NUM_CONTEXT_PAIR_SAMPLES * 2 * OVERSAMPLE_FACTOR)  # oversample
            contexts = [pair for pair in split_pairs(contexts) if pair[0] != pair[1]]  # only keep unique src and tgt
            contexts = list(set(contexts))  # remove duplicates

            # contexts = list(set(contexts))  # remove duplicates
            contexts = contexts[:NUM_CONTEXT_PAIR_SAMPLES]  # reduce to desired number of

            # check enough contexts
            if len(contexts) < NUM_CONTEXT_PAIR_SAMPLES:
                warnings.warn(f'wanted {NUM_CONTEXT_PAIR_SAMPLES} contexts for env {exp.cfg.env_name} in context. '
                              f'sampled {len(contexts)}')

            # save contexts
            contexts_file.parent.mkdir(exist_ok=True, parents=True)
            with open(contexts_file, 'wb') as f:
                pickle.dump(contexts, f)

        return contexts
