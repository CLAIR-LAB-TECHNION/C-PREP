import math
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Lock
import sys
from typing import List
import traceback

from tqdm.auto import tqdm

from .configurations import *
from .experiment import Experiment, SRC_TASK_NAME, TGT_TASK_NAME, TSF_TASK_NAME, DONE_FILE, FAIL_FILE


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

        # check if src and tgt experiments are done
        src_done = ExperimentsRunner.check_exp_done(exp, SRC_TASK_NAME)
        tgt_done = ExperimentsRunner.check_exp_done(exp, TGT_TASK_NAME)
        tsf_done = ExperimentsRunner.check_exp_done(exp, TSF_TASK_NAME)

        if src_done and tgt_done and tsf_done:
            print('experiment already completed')
            return
        try:
            c_src, c_tgt = exp.load_or_sample_contexts()
            exp.run(c_src, c_tgt)
        except:
            # experiment has failed
            # - output traceback to user in stderr
            # - continue to next experiment
            tb = traceback.format_exc()
            print(tb, file=sys.stderr)
            raise

    @staticmethod
    def check_exp_done(exp, task_name):

        done = False
        if exp.is_done(task_name) and not exp.force_retrain:
            done = True
            print(f'{exp.exp_name} experiment already done')
        elif exp.is_done(task_name):  # forced retrainig
            exp.done_file(task_name).unlink()
            print(f'overwriting done {exp.exp_name} experiment')
        elif exp.is_fail(task_name):
            exp.fail_file(task_name).unlink()
            print(f'{exp.exp_name} experiment failed in the passed. retraining all agents')
            exp.force_retrain = True
        elif exp.force_retrain:
            print(f'redoing {exp.exp_name} experiment')
        else:
            print(f'finishing incomplete {exp.exp_name} experiment')

        return done

    @property
    def num_runs(self):
        return len(self.cfgs)
