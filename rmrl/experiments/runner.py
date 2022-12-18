import math
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Lock
import sys
from typing import List
import traceback

from tqdm.auto import tqdm

from .configurations import *
from .experiment import Experiment

DONE_FILE = 'DONE'
FAIL_FILE = 'FAIL'


TIMESTAMP_FORMAT = '%Y-%m-%d-%H_%M_%S.%f'

pbar_lock = Lock()


class ExperimentsRunner:
    def __init__(self, cfgs: List[ExperimentConfiguration], log_interval,
                 chkp_freq, num_workers, verbose, force_retrain):
        self.cfgs = cfgs
        self.log_interval = log_interval
        self.chkp_freq = chkp_freq
        self.num_workers = num_workers
        self.verbose = int(verbose)  # assert int input for sb3
        self.force_retrain = force_retrain

    def run(self):
        start = time.time()

        exp_objects = [Experiment(cfg, self.log_interval, chkp_freq=self.chkp_freq,
                                 dump_dir=EXPERIMENTS_DUMPS_DIR, verbose=self.verbose,
                                 force_retrain=self.force_retrain)
                       for cfg in self.cfgs]

        if self.num_workers > 1:
            self._run_multiprocess(exp_objects)
        else:
            self._run(exp_objects)

        end = time.time()
        print(f'time to run all experiments: {end - start}')

    def _run_multiprocess(self, exp_objects):
        with ThreadPoolExecutor(self.num_workers) as executor:
            list(tqdm(iterable=executor.map(self._run_exp, exp_objects),
                      total=self.num_runs,
                      desc='all experiments'))

    def _run(self, exp_objects):
        for exp in tqdm(exp_objects, total=self.num_runs, desc='all experiments'):
            self._run_exp(exp)

    @staticmethod
    def _run_exp(exp):
        print(f'running experiment with CFG: {exp.exp_name}')

        if exp.cfg.tsf_kwargs['no_transfer']:
            dump_dir = exp.dump_dir
        else:
            exp._is_tgt = True
            dump_dir = exp.dump_dir
            exp._is_tgt = False

        done_file = dump_dir / DONE_FILE
        fail_file = dump_dir / FAIL_FILE

        if done_file.is_file() and not exp.force_retrain:
            print('experiment already done')
            return
        elif done_file.is_file():  # forced retrainig
            done_file.unlink()
            print('overwriting done experiment')
        elif fail_file.is_file():
            fail_file.unlink()
            print('experiment failed in the passed. retraining all agents')
            exp.force_retrain = True
        elif exp.force_retrain:
            print('redoing experiment')
        else:
            print('finishing incomplete experiment')

        try:
            c_src, c_tgt = exp.load_or_sample_contexts()
            exp.run(c_src, c_tgt)
            open(done_file, 'w').close()  # experiment done indicator
        except:
            # experiment has failed
            # - create path to failure file if not yet created
            # - log traceback to failure file
            # - output traceback to user in stderr
            # - continue to next experiment
            if not exp.dump_dir.exists():
                exp.dump_dir.mkdir(parents=True, exist_ok=True)

            tb = traceback.format_exc()
            with open(fail_file, 'w') as f:
                f.write(tb)

            print(tb, file=sys.stderr)

    @property
    def num_runs(self):
        return len(self.cfgs)
