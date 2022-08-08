import matplotlib.pyplot as plt
import numpy as np
import pprint

from rmrl.utils.misc import sha3_hash
from .configurations import *
from .with_transfer import WithTransferExperiment
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from functools import partial


class Plotter:
    def __init__(self, dump_dir, cfg_constraints, u_bound, l_bound, show_scratch=True, show_transfer=True,
                 with_std=False, xlim=None):
        self.constraints = cfg_constraints
        self.u_bound = u_bound
        self.l_bound = l_bound
        self.show_scratch = show_scratch
        self.show_transfer = show_transfer
        self.with_std = with_std
        self.xlim = xlim or tuple()

        experiments = WithTransferExperiment.load_all_experiments_in_path(dump_dir)
        self.experiments = list(filter(self.__check_cfg_constraints, experiments))

        self.pprinter = pprint.PrettyPrinter(indent=2)

    def load_exp_evals(self, src, tgt):
        src_name = sha3_hash(tuple(src))
        tgt_name = sha3_hash(tuple(tgt))

        src_evals, tgt_evals, tsf_evals = [], [], []
        for exp in self.experiments:
            src_evals.append(np.load(exp.eval_log_dir / src_name / 'evaluations.npz'))
            tgt_evals.append(np.load(exp.eval_log_dir / tgt_name / 'evaluations.npz'))
            tsf_evals.append(np.load(exp.eval_log_dir / f'{tgt_name}_transfer_from_{src_name}' / 'evaluations.npz'))

        return src_evals, tgt_evals, tsf_evals

    def plot_compare_evals(self, src_evals, tgt_evals, tsf_evals, axes=None):
        if axes is None:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        else:
            ax1, ax2 = axes
        ax1.set_title(f'Policy performance on SRC context')
        ax1.set_xlabel('timesteps')
        ax1.set_ylabel('mean reward')
        ax1.set_xlim(*self.xlim)
        ax1.axhline(self.u_bound, ls='--')
        ax1.axhline(self.l_bound, ls='--')
        self.plot_evals(src_evals, ax=ax1)
        ax1.legend()

        ax2.set_title(f'Policy performance on TGT context')
        ax2.set_xlabel('timesteps')
        ax2.set_ylabel('mean reward')
        ax2.axhline(self.u_bound, ls='--')
        ax2.axhline(self.l_bound, ls='--')
        if self.show_scratch:
            self.plot_evals(tgt_evals, ax=ax2)
        if self.show_transfer:
            self.plot_evals(tsf_evals, ax=ax2)
        ax2.legend()

    def plot_compare_individual_seed(self, src, tgt, axes=None):
        src_evals, tgt_evals, tsf_evals = self.load_exp_evals(src, tgt)
        self.plot_compare_evals({str(i) + 's': e for i, e in enumerate(src_evals, 1)},
                                {str(i) + 's': e for i, e in enumerate(tgt_evals, 1)},
                                {str(i) + 't': e for i, e in enumerate(tsf_evals, 1)},
                                axes=axes)

        self.pprinter.pprint({
            i: repr(exp.cfg)
            for i, exp in enumerate(self.experiments, 1)
        })

    def plot_compare_agg_seeds(self, src, tgt, axes=None):
        src_evals, tgt_evals, tsf_evals = self.load_exp_evals(src, tgt)

        hash_to_exp = {}
        hash_to_src = {}
        hash_to_tgt = {}
        hash_to_tsf = {}
        for exp1, src1, tgt1, tsf1 in zip(self.experiments, src_evals, tgt_evals, tsf_evals):
            exp1_name = sha3_hash(repr(exp1.cfg).rsplit(CFG_VALS_SEP, 1)[0])

            if exp1_name in hash_to_exp:
                continue

            hash_to_exp.setdefault(exp1_name, []).append(exp1)
            hash_to_src.setdefault(exp1_name, []).append(src1)
            hash_to_tgt.setdefault(exp1_name, []).append(tgt1)
            hash_to_tsf.setdefault(exp1_name, []).append(tsf1)
            for exp2, src2, tgt2, tsf2 in zip(self.experiments, src_evals, tgt_evals, tsf_evals):
                exp2_name = sha3_hash(repr(exp2.cfg).rsplit(CFG_VALS_SEP, 1)[0])

                if exp1_name == exp2_name:
                    hash_to_exp.setdefault(exp1_name, []).append(exp2)
                    hash_to_src.setdefault(exp1_name, []).append(src2)
                    hash_to_tgt.setdefault(exp1_name, []).append(tgt2)
                    hash_to_tsf.setdefault(exp1_name, []).append(tsf2)

        hash_to_src = self.__concat_res_dicts(hash_to_src)
        hash_to_tgt = self.__concat_res_dicts(hash_to_tgt)
        hash_to_tsf = self.__concat_res_dicts(hash_to_tsf)

        self.plot_compare_evals({str(i) + 's': e for i, e in enumerate(hash_to_src.values(), 1)},
                                {str(i) + 's': e for i, e in enumerate(hash_to_tgt.values(), 1)},
                                {str(i) + 't': e for i, e in enumerate(hash_to_tsf.values(), 1)},
                                axes=axes)

        self.pprinter.pprint({
            i: repr(group_exps[0].cfg).rsplit(CFG_VALS_SEP, 1)[0]
            for i, group_exps in enumerate(hash_to_exp.values(), 1)
        })

    def plot_single_eval(self, npz, label, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 7))

        x = npz['timesteps']
        ys = npz['results']
        means = np.array([np.mean(yy) for yy in ys])

        ax.plot(x, means, label=label)
        if self.with_std:
            stds = np.array([np.std(yy) for yy in ys])
            ax.fill_between(x, means - stds, means + stds, alpha=0.2)

    def plot_evals(self, evals, ax=None):
        for k, npz in evals.items():
            self.plot_single_eval(npz, k, ax)

    def __check_cfg_constraints(self, exp):
        if isinstance(self.constraints, list):
            return any(all(getattr(exp.cfg, k) == v for k, v in constraints_dict.items())
                       for constraints_dict in self.constraints)

        return all(getattr(exp.cfg, k) == v for k, v in self.constraints.items())

    def __concat_res_dicts(self, hash_dict):
        new_hash_dict = {}
        for i, (k, evals) in enumerate(hash_dict.items()):
            new_dict = {}
            new_dict['timesteps'] = max([e['timesteps'] for e in evals], key=len)

            results = [0] * len(new_dict['timesteps'])
            for j in range(len(results)):
                res = []
                for e in evals:
                    if j < len(e['results']):
                        res.append(np.mean(e['results'][j]))
                results[j] = res

            new_dict['results'] = results
            new_hash_dict[k] = new_dict

        return new_hash_dict
