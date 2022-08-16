import re
from typing import Type

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .configurations import *
from .cv_transfer import CVTransferExperiment
from .experiment import Experiment
from .with_transfer import TRANSFER_FROM_MIDFIX

EVALUATIONS_FILE = 'evaluations.npz'

TSF_KEY = 'tsf'

TGT_KEY = 'tgt'

SRC_KEY = 'src'

TIMESTEPS_KEY = 'timesteps'
RESULTS_KEY = 'results'

COLORS = list(mpl.colors.BASE_COLORS.keys())


class ResultsHandler:
    def __init__(self, exp_type: Type[Experiment], dump_dir: os.PathLike = None):
        self.exp_type = exp_type
        self.exp_dump_dir = Path(dump_dir or EXPERIMENTS_DUMPS_DIR) / exp_type.__name__

        self.exp_path_dict = dict(enumerate(self.load_all_run_dump_paths(self.exp_dump_dir), 1))
        self.path_to_idx = {v: [k] for k, v in self.exp_path_dict.items()}
        self.exp_obj_dict = {i: [self.make_exp_for_path(p)] for i, p in self.exp_path_dict.items()}

        self.path_to_idx_seed_agg = self.__get_exp_agg(r'/seed-[0-9]+')
        self.exp_path_dict_seed_agg = dict(enumerate(self.path_to_idx_seed_agg.keys(), 1))
        self.exp_obj_dict_seed_agg = {k: [self.exp_obj_dict[i]
                                          for i in self.path_to_idx_seed_agg[p]]
                                      for k, p in self.exp_path_dict_seed_agg.items()}

        if exp_type == CVTransferExperiment:
            self.path_to_idx_fold_agg = self.__get_exp_agg(CFG_VALS_SEP + r'fold-[0-9]+')
            self.exp_path_dict_fold_agg = dict(enumerate(self.path_to_idx_fold_agg.keys(), 1))
            self.exp_obj_dict_fold_agg = {k: [self.exp_obj_dict[i]
                                              for i in self.path_to_idx_fold_agg[p]]
                                          for k, p in self.exp_path_dict_fold_agg.items()}

            self.path_to_idx_fold_and_seed_agg = self.__get_exp_agg(CFG_VALS_SEP + r'seed-[0-9]+/fold-[0-9]+')
            self.exp_path_dict_fold_and_seed_agg = dict(enumerate(self.path_to_idx_fold_and_seed_agg.keys(), 1))
            self.exp_obj_dict_fold_and_seed_agg = {k: [self.exp_obj_dict[i]
                                                       for i in self.path_to_idx_fold_and_seed_agg[p]]
                                                   for k, p in self.exp_path_dict_fold_and_seed_agg.items()}

        else:
            self.path_to_idx_fold_agg = {}
            self.exp_path_dict_fold_agg = {}
            self.exp_obj_dict_fold_agg = {}

            self.path_to_idx_fold_and_seed_agg = {}
            self.exp_path_dict_fold_and_seed_agg = {}
            self.exp_obj_dict_fold_and_seed_agg = {}

    def __get_exp_agg(self, agg_cfg_pattern):
        exp_dict_agg = {}
        for i, exp_path in self.exp_path_dict.items():
            exp_cfg_repr_no_seed = re.sub(agg_cfg_pattern, '', exp_path)

            if exp_cfg_repr_no_seed in exp_dict_agg:
                continue
            exp_dict_agg.setdefault(exp_cfg_repr_no_seed, []).append(i)

            for j, exp_path2 in self.exp_path_dict.items():
                exp2_cfg_repr_no_seed = re.sub(agg_cfg_pattern, '', exp_path2)

                if exp_cfg_repr_no_seed == exp2_cfg_repr_no_seed and i != j:
                    exp_dict_agg.setdefault(exp_cfg_repr_no_seed, []).append(j)

        return exp_dict_agg

    def plot_experiments_eval(self,
                              experiments_idx=None,
                              plot_kwargs_per_idx=None,
                              cfg_constraints=None,
                              exp_agg_type=None,  # None, "seed", "fold", or "fold_and_seed"
                              gamma=1,
                              record_median=False,
                              with_deviation=False,
                              src_xlim=None,
                              tgt_xlim=None,
                              l_bound=None,
                              u_bound=None,
                              show_src_scratch=True,
                              show_tgt_scratch=True,
                              show_tgt_transfer=True,
                              save_path=None):

        # get correct object dict
        if exp_agg_type is None:
            cfg_idx_to_path = self.exp_path_dict
            path_to_idx = self.path_to_idx
        elif exp_agg_type == "seed":
            cfg_idx_to_path = self.exp_path_dict_seed_agg
            path_to_idx = self.path_to_idx_seed_agg
        elif exp_agg_type == "fold":
            cfg_idx_to_path = self.exp_path_dict_fold_agg
            path_to_idx = self.path_to_idx_fold_agg
        elif exp_agg_type == "fold_and_seed":
            cfg_idx_to_path = self.exp_path_dict_fold_and_seed_agg
            path_to_idx = self.path_to_idx_fold_and_seed_agg
        else:
            raise ValueError(f'bad agg type "{exp_agg_type}". can be None, "seed", "fold", or "fold_and_seed"')

        # filter experiments according to given indices
        if experiments_idx:
            toss_idx = set(cfg_idx_to_path.keys()) - set(experiments_idx)
            for i in toss_idx:
                path_to_idx.pop(cfg_idx_to_path[i])

                # filter experiments according to constraints

        path_to_idx = self.__filter_out_unconstrained(path_to_idx, self.exp_obj_dict, cfg_constraints)

        all_res = self.get_path_results(path_to_idx, cfg_idx_to_path, gamma)
        self.plot_compare_evals(
            src_evals={k: v[SRC_KEY] for k, v in all_res.items()},
            tgt_evals={k: v[TGT_KEY] for k, v in all_res.items()},
            tsf_evals={k: v[TSF_KEY] for k, v in all_res.items()},
            l_bound=l_bound,
            u_bound=u_bound,
            show_src_scratch=show_src_scratch,
            show_tgt_scratch=show_tgt_scratch,
            show_tgt_transfer=show_tgt_transfer,
            src_xlim=src_xlim,
            tgt_xlim=tgt_xlim,
            plt_kwargs=plot_kwargs_per_idx or {},
            record_median=record_median,
            with_deviation=with_deviation
        )

        plt.show()

    def plot_compare_evals(self, src_evals, tgt_evals, tsf_evals, l_bound, u_bound, show_src_scratch, show_tgt_scratch,
                           show_tgt_transfer, src_xlim, tgt_xlim, plt_kwargs, record_median=False, with_deviation=False,
                           axes=None):
        if axes is None:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        else:
            ax1, ax2 = axes

        y_label = ('median' if record_median else 'average') + ' return'

        ax1.set_title(f'Policy performance on SRC context')
        ax1.set_xlabel('timesteps')
        ax1.set_ylabel(y_label)
        if u_bound is not None:
            ax1.axhline(u_bound, ls='--')
        if l_bound is not None:
            ax1.axhline(l_bound, ls='--')

        if show_src_scratch:
            self.plot_evals(src_evals, plt_kwargs, record_median=record_median, with_deviation=with_deviation, ax=ax1)
            ax1.set_xlim(src_xlim)
            ax1.legend()

        ax2.set_title(f'Policy performance on TGT context')
        ax2.set_xlabel('timesteps')
        ax2.set_ylabel(y_label)
        if u_bound is not None:
            ax2.axhline(u_bound, ls='--')
        if l_bound is not None:
            ax2.axhline(l_bound, ls='--')
        if show_tgt_scratch:
            self.plot_evals(tgt_evals, plt_kwargs, record_median=record_median, with_deviation=with_deviation, ax=ax2)
        if show_tgt_transfer:
            self.plot_evals(tsf_evals, plt_kwargs, is_transfer=True, record_median=record_median,
                            with_deviation=with_deviation, ax=ax2)

        if show_tgt_scratch or show_tgt_transfer:
            ax2.set_xlim(tgt_xlim)
            ax2.legend()

    def plot_evals(self, evals, plt_kwargs, is_transfer=False, record_median=False,  with_deviation=False, ax=None):
        if isinstance(plt_kwargs, dict):
            plt_kwargs = [plt_kwargs.copy() for _ in range(len(evals))]

        assert len(evals) == len(plt_kwargs)
        for (k, npz), kwargs in zip(evals.items(), plt_kwargs):
            if 'color' not in kwargs:
                kwargs['color'] = COLORS[k % len(COLORS)]
            if is_transfer:
                if 'ls' not in kwargs and 'linestyle' not in kwargs:
                    kwargs['ls'] = '-.'
            self.plot_single_eval(npz, k, record_median=record_median, with_deviation=with_deviation, ax=ax, **kwargs)

    def plot_single_eval(self, res_dict, eval_key=None, record_median=False, with_deviation=False,
                         ax=None, **plt_kwargs):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 7))

        x = res_dict['timesteps']
        ys = res_dict['results']

        if record_median:
            to_plot = np.array([np.median(yy) for yy in ys])
        else:
            to_plot = np.array([np.mean(yy) for yy in ys])

        if 'label' not in plt_kwargs:
            plt_kwargs['label'] = eval_key
        ax.plot(x, to_plot, **plt_kwargs)

        if with_deviation:
            if record_median:
                perc_25 = np.array([np.percentile(yy, 25) for yy in ys])
                perc_75 = np.array([np.percentile(yy, 75) for yy in ys])
                ax.fill_between(x, perc_25, perc_75, alpha=0.2, color=plt_kwargs['color'])
            else:
                stds = np.array([np.std(yy) for yy in ys])
                ax.fill_between(x, to_plot - stds, to_plot + stds, alpha=0.2, color=plt_kwargs['color'])

    def get_path_results(self, path_to_idx, cfg_idx_to_path, gamma):
        res = {}
        path_to_cfg_idx = {v: k for k, v in cfg_idx_to_path.items()}
        for p, idx in path_to_idx.items():
            per_idx_results = self.load_results_for_indices(idx)

            # aggregate episode data
            res[path_to_cfg_idx[p]] = {
                SRC_KEY: self.mean_discounted_rewards(per_idx_results, gamma, SRC_KEY),
                TGT_KEY: self.mean_discounted_rewards(per_idx_results, gamma, TGT_KEY),
                TSF_KEY: self.mean_discounted_rewards(per_idx_results, gamma, TSF_KEY),
            }

        return res

    def mean_discounted_rewards(self, npz_res_dict, gamma, key):
        returns = [
            self.discounted_return_results(npz_res_dict[i][key][RESULTS_KEY], gamma)
            for i in npz_res_dict
        ]

        # pad with NaN values
        max_len = max(map(lambda a: a.shape[0], returns))
        padded_returns = np.array([
            np.vstack([
                r,
                np.full((max_len - r.shape[0],) + r.shape[1:], np.nan)
            ])
            for r in returns
        ])

        return {
            TIMESTEPS_KEY: max([npz_res_dict[i][key][TIMESTEPS_KEY] for i in npz_res_dict], key=lambda v: len(v)),
            RESULTS_KEY: np.mean(padded_returns, axis=-1).T  # transpose to make it [timestep, cfg]
        }

    @staticmethod
    def discounted_return_results(npz_res, gamma):
        return np.array([[np.dot(ep, [gamma ** i
                                      for i in range(len(ep))])
                          for ep in run]
                         for run in npz_res])

    def load_results_for_indices(self, idx):
        results = {}
        for i in idx:
            p = Path(self.exp_path_dict[i]) / LOGS_DIR / EVAL_LOG_DIR
            tsf_files = list(p.glob(f'*{TRANSFER_FROM_MIDFIX}*'))

            if len(tsf_files) == 0:
                raise IndexError(f'No transfer results found for experiment {i}')
            if len(tsf_files) > 1:
                raise IndexError(f'multiple transfer results found for experiment {i}')

            tsf_filename = tsf_files[0].name  # exactly 1 tsf file

            match = re.match(fr'(.+){TRANSFER_FROM_MIDFIX}(.+)', str(tsf_filename))
            src_context_name = match.group(1)
            tgt_context_name = match.group(2)

            results[i] = {
                SRC_KEY: self.load_exp_eval_in_path(p / src_context_name),
                TGT_KEY: self.load_exp_eval_in_path(p / tgt_context_name),
                TSF_KEY: self.load_exp_eval_in_path(p / tsf_filename)
            }

        return results

    def load_exp_eval_in_path(self, p):
        return np.load(p / EVALUATIONS_FILE, allow_pickle=True)

    def plot_experiments_training(self):
        # TODO
        pass

    def print_all_experiments(self, cfg_constraints=None):
        self.__print_exp_dict(self.exp_obj_dict, self.exp_path_dict, cfg_constraints)

    def print_seed_agg_experiments(self, cfg_constraints=None):
        self.__print_exp_dict(self.exp_obj_dict_seed_agg, self.exp_path_dict_seed_agg, cfg_constraints)

    def print_fold_agg_experiments(self, cfg_constraints=None):
        self.__print_exp_dict(self.exp_obj_dict_fold_agg, self.exp_path_dict_fold_agg, cfg_constraints)

    def print_fold_and_seed_agg_experiments(self, cfg_constraints=None):
        self.__print_exp_dict(self.exp_obj_dict_fold_and_seed_agg, self.exp_path_dict_fold_and_seed_agg,
                              cfg_constraints)

    def __print_exp_dict(self, d, cfg_idx_to_path, cfg_constraints=None):
        for i, exp in d.items():
            if not self.__check_cfg_constraints(exp, cfg_constraints):
                continue
            print(f'{i}: {cfg_idx_to_path[i]}')
            print()

    @staticmethod
    def __check_cfg_constraints(exp, constraints):
        if not constraints:
            return True

        if isinstance(constraints, list):
            return any(all(getattr(exp.cfg, k) == v for k, v in constraints_dict.items())
                       for constraints_dict in constraints)

        return all(getattr(exp.cfg, k) == v for k, v in constraints.items())

    @classmethod
    def __filter_out_unconstrained(cls, path_to_idx_d, idx_to_obj_d, constraints):
        if not constraints:
            return path_to_idx_d  # don't go over items if no constraints

        # for each path keep only configurations that are
        d = {k: list(filter(lambda i: cls.__check_cfg_constraints(idx_to_obj_d[i], constraints), v))
             for k, v in path_to_idx_d.items()}
        d = {k: v for k, v in d.items() if v}

        return d

    @staticmethod
    def load_all_run_dump_paths(path):
        run_paths = []
        for root, dirs, files in os.walk(path):
            if LOGS_DIR in dirs:
                run_paths.append(root)
            if MODELS_DIR in root or LOGS_DIR in root:
                continue
        return run_paths

    def make_exp_for_path(self, path):
        no_exp_name_repr = path.replace(str(self.exp_dump_dir) + '/', '', 1)
        no_fold_repr = re.sub(r'/fold-[0-9]+', '', no_exp_name_repr)  # handle for CV experiment
        cfg = ExperimentConfiguration.from_repr_value(no_fold_repr)
        return self.exp_type(cfg)