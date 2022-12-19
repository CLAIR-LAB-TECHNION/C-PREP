import glob
import re
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from .configurations import *
from .experiment import Experiment
from .with_transfer import WithTransferExperiment

EVALUATIONS_FILE = 'evaluations.npz'

SRC_KEY = 'src'
TST_KEY = 'test'
TGT_KEY = 'tgt'
TSF_KEY = 'tsf'

TIMESTEPS_KEY = 'timesteps'
RESULTS_KEY = 'results'
RETURNS_KEY = 'returns'

COLORS = list(mpl.colors.BASE_COLORS.keys())


class ResultsHandler:
    def __init__(self, dump_dir: os.PathLike = None):
        self.exp_dump_dir = Path(dump_dir or EXPERIMENTS_DUMPS_DIR) / RUNS_DIR

        done_paths, self.failed_experiments, self.incomplete_experiments = self.load_all_run_dump_paths()
        self.exp_path_dict = dict(enumerate(done_paths, 1))
        self.path_to_idx = {v: [k] for k, v in self.exp_path_dict.items()}
        self.exp_obj_dict = {i: [self.make_exp_for_path(p)] for i, p in self.exp_path_dict.items()}

        self.path_to_idx_seed_agg = self.__get_exp_agg(r'/seed-[0-9]+')
        self.exp_path_dict_seed_agg = dict(enumerate(self.path_to_idx_seed_agg.keys(), 1))
        self.exp_obj_dict_seed_agg = {k: [self.exp_obj_dict[i][0]
                                          for i in self.path_to_idx_seed_agg[p]]
                                      for k, p in self.exp_path_dict_seed_agg.items()}

        # if exp_type == CVTransferExperiment:
        #     self.path_to_idx_fold_agg = self.__get_exp_agg(CFG_VALS_SEP + r'fold-[0-9]+')
        #     self.exp_path_dict_fold_agg = dict(enumerate(self.path_to_idx_fold_agg.keys(), 1))
        #     self.exp_obj_dict_fold_agg = {k: [self.exp_obj_dict[i][0]
        #                                       for i in self.path_to_idx_fold_agg[p]]
        #                                   for k, p in self.exp_path_dict_fold_agg.items()}
        #
        #     self.path_to_idx_fold_and_seed_agg = self.__get_exp_agg(CFG_VALS_SEP + r'seed-[0-9]+/fold-[0-9]+')
        #     self.exp_path_dict_fold_and_seed_agg = dict(enumerate(self.path_to_idx_fold_and_seed_agg.keys(), 1))
        #     self.exp_obj_dict_fold_and_seed_agg = {k: [self.exp_obj_dict[i][0]
        #                                                for i in self.path_to_idx_fold_and_seed_agg[p]]
        #                                            for k, p in self.exp_path_dict_fold_and_seed_agg.items()}
        #
        # else:
        #     self.path_to_idx_fold_agg = {}
        #     self.exp_path_dict_fold_agg = {}
        #     self.exp_obj_dict_fold_agg = {}
        #
        #     self.path_to_idx_fold_and_seed_agg = {}
        #     self.exp_path_dict_fold_and_seed_agg = {}
        #     self.exp_obj_dict_fold_and_seed_agg = {}

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

    def get_experiment_contexts_envs_and_agents(self, exp_idx):
        exp = self.exp_obj_dict[exp_idx][0]
        src_context, tgt_context = exp.load_or_sample_contexts()
        src_train_env = exp.get_single_rm_env_for_context_set(src_context)
        src_eval_env = exp.get_single_rm_env_for_context_set(src_context)
        src_agent = exp.load_agent_for_env(src_train_env, 'src', force_load=True)

        if isinstance(exp, WithTransferExperiment):
            tgt_train_env = exp.get_single_rm_env_for_context_set(tgt_context)
            tgt_eval_env = exp.get_single_rm_env_for_context_set(tgt_context)
            tgt_agent = exp.load_agent_for_env(tgt_train_env, 'tgt', force_load=True)
            tsf_agent = exp.transfer_agent(src_train_env, tgt_train_env, tgt_eval_env)

            return (src_context, src_train_env, src_eval_env, src_agent,
                    tgt_context, tgt_train_env, tgt_eval_env, tgt_agent,
                    tsf_agent)

        else:
            return src_context, src_train_env, src_eval_env, src_agent

    def get_cfg_idx_to_path_to_seed_idx_maps(self, experiments_idx=None, cfg_constraints=None, exp_agg_type=None):

        # get correct object dict
        if exp_agg_type is None:
            cfg_idx_to_path = self.exp_path_dict
            path_to_idx = self.path_to_idx.copy()
        elif exp_agg_type == "seed":
            cfg_idx_to_path = self.exp_path_dict_seed_agg
            path_to_idx = self.path_to_idx_seed_agg.copy()
        elif exp_agg_type == "fold":
            cfg_idx_to_path = self.exp_path_dict_fold_agg
            path_to_idx = self.path_to_idx_fold_agg.copy()
        elif exp_agg_type == "fold_and_seed":
            cfg_idx_to_path = self.exp_path_dict_fold_and_seed_agg
            path_to_idx = self.path_to_idx_fold_and_seed_agg.copy()
        else:
            raise ValueError(f'bad agg type "{exp_agg_type}". can be None, "seed", "fold", or "fold_and_seed"')

        # filter experiments according to given indices
        if experiments_idx:
            toss_idx = set(cfg_idx_to_path.keys()) - set(experiments_idx)
            for i in toss_idx:
                path_to_idx.pop(cfg_idx_to_path[i])

        # filter experiments according to constraints
        path_to_idx = self.__filter_out_unconstrained(path_to_idx, cfg_constraints)
        cfg_idx_to_path = {k: v for k, v in cfg_idx_to_path.items() if v in path_to_idx}  # only keep needed

        return path_to_idx, cfg_idx_to_path

    def dump_experiment_logs(self,
                             experiments_idx=None,
                             cfg_constraints=None,
                             exp_agg_type=None,  # None, "seed", "fold", or "fold_and_seed"
                             evals_only=True,  # TODO support False
                             varying_attrs=None,
                             cfg_title=None):

        path_to_idx, _ = self.get_cfg_idx_to_path_to_seed_idx_maps(experiments_idx,
                                                                   cfg_constraints,
                                                                   exp_agg_type)
        for p in path_to_idx:
            if evals_only:
                copy_paths = glob.glob(p + '/**/evaluations.npz', recursive=True)
            else:
                copy_paths = [p]  # TODO this is not yet supported

            for cp_path in copy_paths:
                prefix = cfg_title or ''
                if varying_attrs:
                    for attr in reversed(varying_attrs):
                        match = re.match(f'.*\/{attr}-([^\/]+)\/.*', cp_path)
                        if not match:
                            match = re.match(f'.*{attr}-([^\/]+)\/.*', cp_path)

                        if not match:
                            attr_val = 'NONE'
                        else:
                            attr_val = match.group(1)

                        prefix = attr + '-' + attr_val + '/' + prefix

                prefix = self.exp_dump_dir / 'eval_dumps' / prefix
                out_path = cp_path
                if prefix:
                    out_path = out_path.replace(p, str(prefix))

                os.makedirs(str(Path(out_path).parent), exist_ok=True)
                shutil.copy(cp_path, out_path)

    def plot_experiments_eval(self,
                              experiments_idx=None,
                              plot_kwargs_per_idx=None,
                              cfg_constraints=None,
                              exp_agg_type=None,  # None, "seed", "fold", or "fold_and_seed"
                              record_median=False,
                              record_returns=False,
                              with_deviation=False,
                              src_xlim=None,
                              tgt_xlim=None,
                              l_bound=None,
                              u_bound=None,
                              show_src_scratch=True,
                              show_src_test=True,
                              show_tgt_scratch=True,
                              show_tgt_transfer=True,
                              save_path=None,
                              axes=None,
                              **save_kwargs):

        path_to_idx, cfg_idx_to_path = self.get_cfg_idx_to_path_to_seed_idx_maps(experiments_idx,
                                                                                 cfg_constraints,
                                                                                 exp_agg_type)

        all_res = self.get_path_results(path_to_idx, cfg_idx_to_path, record_returns)

        if not experiments_idx:
            experiments_idx = list(cfg_idx_to_path.keys())

        self.__plot_compare_evals(
            src_evals={k: all_res[k].get(SRC_KEY) for k in experiments_idx if k in all_res},
            tst_evals={k: all_res[k].get(TST_KEY) for k in experiments_idx if k in all_res},
            tgt_evals={k: all_res[k].get(TGT_KEY) for k in experiments_idx if k in all_res},
            tsf_evals={k: all_res[k].get(TSF_KEY) for k in experiments_idx if k in all_res},
            l_bound=l_bound,
            u_bound=u_bound,
            show_src_scratch=show_src_scratch,
            show_src_test=show_src_test,
            show_tgt_scratch=show_tgt_scratch,
            show_tgt_transfer=show_tgt_transfer,
            src_xlim=src_xlim,
            tgt_xlim=tgt_xlim,
            plt_kwargs=plot_kwargs_per_idx or {},
            record_returns=record_returns,
            record_median=record_median,
            with_deviation=with_deviation,
            axes=axes
        )

        if save_path is not None:
            plt.savefig(save_path, **save_kwargs)
        else:
            plt.show()

    def __plot_compare_evals(self, src_evals, tst_evals, tgt_evals, tsf_evals, l_bound, u_bound, show_src_scratch,
                             show_src_test, show_tgt_scratch, show_tgt_transfer, src_xlim, tgt_xlim, plt_kwargs,
                             record_returns=False, record_median=False, with_deviation=False, axes=None):
        # check which axes are required
        has_src = any(e is not None for e in src_evals)
        has_tst = any(e is not None for e in tst_evals)
        has_tgt = any(e is not None for e in tgt_evals)
        has_tsf = any(e is not None for e in tsf_evals)
        display_src = has_src and show_src_scratch
        display_tst = has_tst and show_src_test
        display_tgt = has_tgt and show_tgt_scratch
        display_tsf = has_tsf and show_tgt_transfer
        use_src_axis = display_src or display_tst
        use_tgt_axis = display_tgt or display_tsf

        if axes is None:
            if use_src_axis and use_tgt_axis:
                _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            elif use_src_axis:
                _, ax1 = plt.subplots(1, 1, figsize=(15, 7))
                ax2 = None
            elif use_tgt_axis:
                ax1 = None
                _, ax2 = plt.subplots(1, 1, figsize=(15, 7))
            else:
                ax1, ax2 = None, None
                plt.figure()  # figure for showing
        else:
            ax1, ax2 = axes

        y_label = ('median' if record_median else 'average') + (' return' if record_returns else ' acc reward')

        if use_src_axis:
            self.__handle_single_axis(ax1, src_evals, tst_evals, display_src, display_tst, record_median,
                                      with_deviation, l_bound, u_bound, src_xlim, y_label, plt_kwargs)

        if use_tgt_axis:
            self.__handle_single_axis(ax2, tgt_evals, tsf_evals, display_tgt, display_tsf, record_median,
                                      with_deviation, l_bound, u_bound, tgt_xlim, y_label, plt_kwargs)

    def __handle_single_axis(self, ax, evals_1, evals_2, display_1, display_2, record_median, with_deviation, l_bound,
                             u_bound,
                             xlim, y_label, plt_kwargs):
        if display_1 and display_2:
            ax.set_title(f'Policy performance training on source context and testing on target context')
        elif display_1:
            ax.set_title(f'Policy performance training on source context')
        elif display_2:
            ax.set_title(f'Policy performance testing on target while training on source')
        ax.set_xlabel('timesteps')
        ax.set_ylabel(y_label)
        if u_bound is not None:
            ax.axhline(u_bound, ls='--')
        if l_bound is not None:
            ax.axhline(l_bound, ls='--')
        if display_1:
            self.__plot_evals(evals_1, plt_kwargs, record_median=record_median, with_deviation=with_deviation, ax=ax)
        if display_2:
            self.__plot_evals(evals_2, plt_kwargs, is_test=display_1, record_median=record_median,
                              with_deviation=with_deviation, ax=ax)
        ax.set_xlim(xlim)
        ax.legend()
        ax.grid()

    def __plot_evals(self, evals, plt_kwargs, is_transfer=False, is_test=False, record_median=False,
                     with_deviation=False, ax=None):
        if isinstance(plt_kwargs, dict):
            plt_kwargs = [plt_kwargs.copy() for _ in range(len(evals))]

        assert len(evals) == len(plt_kwargs)
        for i, ((k, npz), kwargs) in enumerate(zip(evals.items(), plt_kwargs)):
            if npz is None:
                continue
            if 'color' not in kwargs:
                kwargs['color'] = COLORS[i % len(COLORS)]
            if is_transfer:
                if 'ls' not in kwargs and 'linestyle' not in kwargs:
                    kwargs['ls'] = '-.'
            if is_test:
                if 'ls' not in kwargs and 'linestyle' not in kwargs:
                    kwargs['ls'] = '-.'
            self.__plot_single_eval(npz, k, is_transfer=is_transfer, is_test=is_test, record_median=record_median,
                                    with_deviation=with_deviation, ax=ax, **kwargs)

    def __plot_single_eval(self, res_dict, eval_key=None, is_transfer=False, is_test=False, record_median=False,
                           with_deviation=False,
                           ax=None, **plt_kwargs):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 7))

        x = res_dict['timesteps']
        ys = res_dict['results']

        if record_median:
            to_plot = np.array([np.nanmedian(yy) for yy in ys])
        else:
            to_plot = np.array([np.nanmean(yy) for yy in ys])

        if 'label' not in plt_kwargs:
            plt_kwargs['label'] = str(eval_key)
        if is_transfer:
            plt_kwargs['label'] += ' tsf'
        elif is_test:
            plt_kwargs['label'] += ' tst'
        ax.plot(x, to_plot, **plt_kwargs)

        if with_deviation:
            if record_median:
                perc_25 = np.array([np.percentile(yy, 25) for yy in ys])
                perc_75 = np.array([np.percentile(yy, 75) for yy in ys])
                ax.fill_between(x, perc_25, perc_75, alpha=0.2, color=plt_kwargs['color'])
            else:
                stds = np.array([np.std(yy) for yy in ys])
                ax.fill_between(x, to_plot - stds, to_plot + stds, alpha=0.2, color=plt_kwargs['color'])

    def get_path_results(self, path_to_idx, cfg_idx_to_path, record_returns):
        val_key = RETURNS_KEY if record_returns else RESULTS_KEY

        res = {}
        path_to_cfg_idx = {v: k for k, v in cfg_idx_to_path.items()}
        for p, idx in tqdm(path_to_idx.items(), desc='getting all results'):
            per_idx_results = self.load_results_for_indices(idx)

            # aggregate episode data
            if self.__path_is_tgt(p):
                res[path_to_cfg_idx[p]] = {
                    TGT_KEY: self.mean_discounted_rewards(per_idx_results, TGT_KEY, val_key),
                    TSF_KEY: self.mean_discounted_rewards(per_idx_results, TSF_KEY, val_key),
                }
            else:
                res[path_to_cfg_idx[p]] = {
                    SRC_KEY: self.mean_discounted_rewards(per_idx_results, SRC_KEY, val_key),
                }

                if self.exp_obj_dict[path_to_cfg_idx[p]][0].cfg.exp_kwargs['use_tgt_for_test']:
                    res[path_to_cfg_idx[p]][TST_KEY] = self.mean_discounted_rewards(per_idx_results, TST_KEY, val_key)

        return res

    def mean_discounted_rewards(self, npz_res_dict, stage_key, val_key):
        returns = [
            npz_res_dict[i][stage_key][val_key]
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
            TIMESTEPS_KEY: max([npz_res_dict[i][stage_key][TIMESTEPS_KEY] for i in npz_res_dict], key=lambda v: len(v)),
            RESULTS_KEY: np.nanmean(padded_returns, axis=-1).T  # transpose to make it [timestep, cfg]
        }

    def load_results_for_indices(self, idx):
        results = {}
        for i in idx:
            p = Path(self.exp_path_dict[i]) / LOGS_DIR / EVAL_LOG_DIR
            src_path = p / 'src'
            tst_path = p / 'test'
            tgt_path = p / 'tgt'
            tsf_path = p / 'tsf'

            if self.__path_is_tgt(p):
                results[i] = {
                    TGT_KEY: self.load_exp_eval_in_path(tgt_path),
                    TSF_KEY: self.load_exp_eval_in_path(tsf_path)
                }

            else:
                results[i] = {
                    SRC_KEY: self.load_exp_eval_in_path(src_path),
                }

                # load test set if expected
                if self.exp_obj_dict[i][0].cfg.exp_kwargs['use_tgt_for_test']:
                    results[i][TST_KEY] = self.load_exp_eval_in_path(tst_path)

        return results

    def load_exp_eval_in_path(self, p):
        return np.load(p / EVALUATIONS_FILE, allow_pickle=True)

    def plot_experiments_training(self):
        # TODO
        pass

    def print_experiments(self, exp_agg_type=None, cfg_constraints=None):
        if exp_agg_type is None:
            self.print_all_experiments(cfg_constraints)
        elif exp_agg_type == "seed":
            self.print_seed_agg_experiments(cfg_constraints)
        elif exp_agg_type == "fold":
            self.print_fold_agg_experiments(cfg_constraints)
        elif exp_agg_type == "fold_and_seed":
            self.print_fold_and_seed_agg_experiments(cfg_constraints)
        else:
            raise ValueError(f'bad agg type "{exp_agg_type}". can be None, "seed", "fold", or "fold_and_seed"')

    def print_all_experiments(self, cfg_constraints=None):
        self.__print_exp_dict(self.exp_obj_dict, self.exp_path_dict, self.path_to_idx, cfg_constraints)

    def print_seed_agg_experiments(self, cfg_constraints=None):
        self.__print_exp_dict(self.exp_obj_dict_seed_agg, self.exp_path_dict_seed_agg, self.path_to_idx_seed_agg,
                              cfg_constraints)

    def print_fold_agg_experiments(self, cfg_constraints=None):
        self.__print_exp_dict(self.exp_obj_dict_fold_agg, self.exp_path_dict_fold_agg, self.path_to_idx_fold_agg,
                              cfg_constraints)

    def print_fold_and_seed_agg_experiments(self, cfg_constraints=None):
        self.__print_exp_dict(self.exp_obj_dict_fold_and_seed_agg, self.exp_path_dict_fold_and_seed_agg,
                              self.path_to_idx_fold_and_seed_agg, cfg_constraints)

    def __print_exp_dict(self, d, cfg_idx_to_path, path_to_idx, cfg_constraints=None):
        d = dict(filter(lambda kv: all(self.__check_cfg_constraints_for_exp(exp, cfg_constraints) for exp in kv[1]),
                        d.items()))
        print(f'num experiments: {len(d)}\n')
        for i, exp_list in d.items():
            print(f'{i} ({len(path_to_idx[cfg_idx_to_path[i]])} runs): {cfg_idx_to_path[i]}')
            print()

    @classmethod
    def __check_cfg_constraints_for_exp(cls, exp, constraints):
        if not constraints:
            return True

        if not isinstance(constraints, list):
            constraints = [constraints]

        return any(cls.__check_single_cfg_constraint_for_cfg(vars(exp.cfg), constraints_dict)
                   for constraints_dict in constraints)

    @classmethod
    def __check_single_cfg_constraint_for_cfg(cls, cfg, constraint):
        if callable(constraint) and not isinstance(constraint, type):
            return constraint(cfg)
        elif not isinstance(cfg, dict) and not isinstance(constraint, dict):

            # ensure same iterable type
            if isinstance(cfg, list):
                cfg = tuple(cfg)
            if isinstance(constraint, list):
                constraint = tuple(constraint)

            return cfg == constraint
        elif isinstance(cfg, dict) and isinstance(constraint, dict):
            # check there are any differences
            for k, v in constraint.items():
                if k not in cfg or not cls.__check_single_cfg_constraint_for_cfg(cfg[k], v):
                    return False
            return True
        else:
            return False

    def __filter_out_unconstrained(self, path_to_idx_d, constraints):
        if not constraints:
            return path_to_idx_d  # don't go over items if no constraints

        # for each path keep only configurations that are
        d = {k: list(filter(lambda i: self.__check_cfg_constraints_for_exp(self.exp_obj_dict[i][0], constraints), v))
             for k, v in path_to_idx_d.items()}
        d = {k: v for k, v in d.items() if v}

        return d

    def load_all_run_dump_paths(self):
        done_paths = []
        failed_paths = []
        inc_paths = []

        with tqdm(desc='collecting dumps') as pb:
            for root, dirs, files in os.walk(self.exp_dump_dir):
                root = Path(root)
                root_parent = root.parent
                if root.name == 'models' and 'logs' in set(p.name for p in root.parent.iterdir()):
                    root = str(root_parent)
                    dirs = [d.name for d in root_parent.iterdir() if d.is_dir()]
                    files = [f.name for f in root_parent.iterdir() if f.is_file()]
                else:
                    continue

                if 'DONE' in files and 'FAIL' not in files:
                    done_paths.append(root)
                    pb.update()
                elif 'FAIL' in files:
                    failed_paths.append(root)
                    pb.update()
                elif MODELS_DIR in dirs or LOGS_DIR in dirs:
                    inc_paths.append(root)
                    pb.update()

        done_paths = sorted(done_paths, key=lambda x: x.split(CFG_VALS_SEP))
        failed_paths = sorted(failed_paths, key=lambda x: x.split(CFG_VALS_SEP))
        inc_paths = sorted(inc_paths, key=lambda x: x.split(CFG_VALS_SEP))
        return done_paths, failed_paths, inc_paths

    def make_exp_for_path(self, path):
        no_exp_name_repr = path.replace(str(self.exp_dump_dir) + '/', '', 1)
        # no_fold_repr = re.sub(r'/fold-[0-9]+', '', no_exp_name_repr)  # handle for CV experiment
        if not self.__path_is_tgt(no_exp_name_repr):
            no_exp_name_repr += '/tsf_kwargs-((no_transfer-True))'

        cfg = TransferConfiguration.from_repr_value(no_exp_name_repr)
        return Experiment(cfg)

    def __path_is_tgt(self, p):
        return 'tsf_kwargs' in str(p)
