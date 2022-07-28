import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List

from sklearn.model_selection import train_test_split

from .configurations import *
from .experiment import Experiment
from .no_transfer import NoTransferExperiment
from .with_transfer import WithTransferExperiment

TIMESTAMP_FORMAT = '%Y-%m-%d-%H_%M_%S.%f'

EXPERIMENTS_DUMPS_DIR = Path('experiment_dumps')
CONTEXTS_DIR_NAME = Path('sampled_contexts')

EXP_TO_FNS = {
    SupportedExperiments.NO_TRANSFER: NoTransferExperiment,
    SupportedExperiments.WITH_TRANSFER: WithTransferExperiment
}


class ExperimentsRunner:
    def __init__(self, experiments: List[SupportedExperiments], cfgs: List[ExperimentConfiguration], sample_seeds,
                 num_workers):
        self.experiments = experiments
        self.cfgs = cfgs
        self.sample_seeds = sample_seeds
        self.num_workers = num_workers

    def run(self):
        dump_dir_with_ts = EXPERIMENTS_DUMPS_DIR / datetime.now().strftime(TIMESTAMP_FORMAT)
        with ThreadPoolExecutor(self.num_workers) as executor:
            for exp_label in self.experiments:  # iterate experiments
                exp_class = EXP_TO_FNS[exp_label]
                for cfg in self.cfgs:  # iterate configurations
                    exp = exp_class(cfg, dump_dir=dump_dir_with_ts, verbose=1)
                    for sample_seed in self.sample_seeds:  # iterate possible samples
                        src_contexts, tgt_contexts = self.load_or_sample_contexts(exp, sample_seed)

                        # check experiment type to get correct input contexts
                        if exp_label == SupportedExperiments.NO_TRANSFER:
                            for c in src_contexts + tgt_contexts:  # do all contexts separately
                                # executor.submit(exp.run, c)
                                exp.run(c)
                        elif exp_label == SupportedExperiments.WITH_TRANSFER:
                            for c_src, c_tgt in product(src_contexts, tgt_contexts):
                                executor.submit(exp.run, c_src, c_tgt)  # transfer from source to target
                                executor.submit(exp.run, c_tgt, c_src)  # transfer from target to source
                        else:
                            raise NotImplementedError(f'unsupported experiment label {exp_label.value}')

    def load_or_sample_contexts(self, exp: Experiment, sample_seed: int):
        contexts_file = CONTEXTS_DIR_NAME / exp.cfg.env_name / exp.cfg.cspace_name / str(sample_seed)
        try:
            with open(contexts_file, 'rb') as f:
                src_contexts, tgt_contexts = pickle.load(f)
        except (FileNotFoundError, EOFError):
            # create env for sampling
            env = exp.get_experiment_env()

            # set seed for constant sampling
            env.seed(sample_seed)

            # sample contexts
            contexts = env.sample_task(NUM_CONTEXT_SAMPLES * OVERSAMPLE_FACTOR)  # oversample
            contexts = list(set(contexts))  # remove duplicates
            contexts = contexts[:NUM_CONTEXT_SAMPLES]  # reduce to desired number of

            # check enough contexts
            if len(contexts) < NUM_CONTEXT_SAMPLES:
                warnings.warn(f'wanted {NUM_CONTEXT_SAMPLES} contexts for env {exp.cfg.env_name} in context. '
                              f'got {len(contexts)}')

            # split to source and target (with seed!)
            src_contexts, tgt_contexts = train_test_split(contexts, train_size=SRC_SET_FRAC, random_state=sample_seed)

            # save contexts
            contexts_file.parent.mkdir(exist_ok=True, parents=True)
            with open(contexts_file, 'wb') as f:
                pickle.dump((src_contexts, tgt_contexts), f)

        return src_contexts, tgt_contexts
