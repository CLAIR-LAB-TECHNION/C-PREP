import glob
import re
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from .configurations import *
from .experiment import Experiment, SRC_TASK_NAME, TST_TASK_NAME, TGT_TASK_NAME, TSF_TASK_NAME, DONE_FILE, FAIL_FILE

PLOT_LINE_STYLES = [
    '-',  # solid
    '--',  # dashed
    ':',  # dotted
    '.-',  #dotdash
]


EVALUATIONS_FILE = 'evaluations.npz'

TIMESTEPS_KEY = 'timesteps'
RESULTS_KEY = 'results'
RETURNS_KEY = 'returns'

COLORS = list(mpl.colors.TABLEAU_COLORS.keys())


class ResultsHandler:
    def __init__(self, dump_dir: os.PathLike = None):
        self.exp_dump_dir = Path(dump_dir or EXPERIMENTS_DUMPS_DIR) / RUNS_DIR

        self.done_exps, self.failed_exps, self.inc_exps = self.load_all_run_dump_paths()

        self.idx_to_path = dict(enumerate(self.done_exps, 1))
        self.path_to_idx = {v: k for k, v in self.idx_to_path.items()}
        self.idx_to_exp_objs = {i: self.make_exps_for_path(p) for i, p in self.idx_to_path.items()}

        self.path_to_idx_seed_agg = self.__get_exp_agg(r'/seed-[0-9]+', self.idx_to_path)
        self.idx_to_path_seed_agg = {i: k for k, v in self.path_to_idx_seed_agg.items() for i in v}
        self.seed_agg_indexing = dict(enumerate(self.path_to_idx_seed_agg, 1))

    def __get_exp_agg(self, agg_cfg_pattern, exps):
        exp_dict_agg = {}
        for i, p1 in exps.items():
            exp_cfg_repr_cleaned = re.sub(agg_cfg_pattern, '', p1)

            if exp_cfg_repr_cleaned in exp_dict_agg:
                continue
            exp_dict_agg.setdefault(exp_cfg_repr_cleaned, []).append(i)

            for j, p2 in exps.items():
                exp2_cfg_repr_no_seed = re.sub(agg_cfg_pattern, '', p2)

                if exp_cfg_repr_cleaned == exp2_cfg_repr_no_seed and i != j:
                    exp_dict_agg.setdefault(exp_cfg_repr_cleaned, []).append(j)

        return exp_dict_agg

    # def get_experiment_contexts_envs_and_agents(self, exp_idx):
    #     exp = self.exp_obj_dict[exp_idx][0]
    #     src_context, tgt_context = exp.load_or_sample_contexts()
    #     src_train_env = exp.get_single_rm_env_for_context_set(src_context)
    #     src_eval_env = exp.get_single_rm_env_for_context_set(src_context)
    #     src_agent = exp.load_agent_for_env(src_train_env, 'src', force_load=True)
    #
    #     if isinstance(exp, WithTransferExperiment):
    #         tgt_train_env = exp.get_single_rm_env_for_context_set(tgt_context)
    #         tgt_eval_env = exp.get_single_rm_env_for_context_set(tgt_context)
    #         tgt_agent = exp.load_agent_for_env(tgt_train_env, 'tgt', force_load=True)
    #         tsf_agent = exp.transfer_agent(src_train_env, tgt_train_env, tgt_eval_env)
    #
    #         return (src_context, src_train_env, src_eval_env, src_agent,
    #                 tgt_context, tgt_train_env, tgt_eval_env, tgt_agent,
    #                 tsf_agent)
    #
    #     else:
    #         return src_context, src_train_env, src_eval_env, src_agent

    def get_agg_specific_idx_maps(self, exp_agg_type):
        # get correct object dict
        if exp_agg_type is None:
            idx_to_path = self.idx_to_path.copy()
            path_to_idx = {p: [i] for p, i in self.path_to_idx.items()}
        elif exp_agg_type == "seed":
            idx_to_path = self.seed_agg_indexing.copy()
            path_to_idx = self.path_to_idx_seed_agg.copy()
        else:
            raise ValueError(f'bad agg type "{exp_agg_type}". can be None, "seed", "fold", or "fold_and_seed"')

        return path_to_idx, idx_to_path

    def get_cfg_idx_to_path_to_seed_idx_maps(self, experiments_idx=None, cfg_constraints=None, exp_agg_type=None):
        path_to_idx, idx_to_path = self.get_agg_specific_idx_maps(exp_agg_type)

        # filter experiments according to given indices
        if experiments_idx:
            toss_idx = set(idx_to_path.keys()) - set(experiments_idx)
            for i in toss_idx:
                path_to_idx.pop(idx_to_path[i])

        # filter experiments according to constraints
        idx_to_tsk_idx = self.__filter_out_unconstrained(path_to_idx, idx_to_path, cfg_constraints)

        return idx_to_tsk_idx, idx_to_path

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
                        match = re.match(f'.*/{attr}-([^/]+)/.*', cp_path)
                        if not match:
                            match = re.match(f'.*{attr}-([^/]+)/.*', cp_path)

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
                              record_iqm=False,
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

        idx_to_tsk_idx, idx_to_path = self.get_cfg_idx_to_path_to_seed_idx_maps(experiments_idx,
                                                                                cfg_constraints,
                                                                                exp_agg_type)

        all_res = self.get_path_results(idx_to_tsk_idx, idx_to_path, record_returns)

        if not experiments_idx:
            experiments_idx = list(idx_to_path.keys())

        self.__plot_compare_evals(
            src_evals={k: all_res[k].get(SRC_TASK_NAME) for k in experiments_idx if k in all_res},
            tst_evals={k: all_res[k].get(TST_TASK_NAME) for k in experiments_idx if k in all_res},
            tgt_evals={k: all_res[k].get(TGT_TASK_NAME) for k in experiments_idx if k in all_res},
            tsf_evals={k: [all_res[k][tsk] for tsk in all_res[k] if tsk.startswith(TSF_TASK_NAME)]
                       for k in experiments_idx if k in all_res},
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
            record_iqm=record_iqm,
            with_deviation=with_deviation,
            axes=axes
        )

        if save_path is not None:
            plt.savefig(save_path, **save_kwargs)
        else:
            plt.show()

    def __plot_compare_evals(self, src_evals, tst_evals, tgt_evals, tsf_evals, l_bound, u_bound, show_src_scratch,
                             show_src_test, show_tgt_scratch, show_tgt_transfer, src_xlim, tgt_xlim, plt_kwargs,
                             record_returns=False, record_median=False, record_iqm=False, with_deviation=False,
                             axes=None):
        # check which axes are required
        has_src = any(e for e in src_evals.values())
        has_tst = any(e for e in tst_evals.values())
        has_tgt = any(e for e in tgt_evals.values())
        has_tsf = any(e for e in tsf_evals.values())
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

        y_label = (('IQM' if record_iqm else 'median' if record_median else 'average') +
                   (' return' if record_returns else ' accumulated reward'))

        if use_src_axis:
            self.__handle_single_axis(ax1, src_evals, tst_evals, display_src, display_tst, 'tst', record_median,
                                      record_iqm, with_deviation, l_bound, u_bound, src_xlim, y_label, plt_kwargs)

        if use_tgt_axis:
            self.__handle_single_axis(ax2, tgt_evals, tsf_evals, display_tgt, display_tsf, 'tsf', record_median,
                                      record_iqm, with_deviation, l_bound, u_bound, tgt_xlim, y_label, plt_kwargs)

    def __handle_single_axis(self, ax, evals_1, evals_2, display_1, display_2, exp_title, record_median, record_iqm,
                             with_deviation, l_bound, u_bound, xlim, y_label, plt_kwargs):
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
            self.__plot_evals(evals_1, plt_kwargs, record_median=record_median, record_iqm=record_iqm,
                              with_deviation=with_deviation, ax=ax)
        if display_2:
            self.__plot_evals(evals_2, plt_kwargs, exp_type=exp_title if display_1 else '', record_median=record_median,
                              record_iqm=record_iqm, with_deviation=with_deviation, ax=ax)
        ax.set_xlim(xlim)
        ax.legend()
        ax.grid()

    def __plot_evals(self, evals, plt_kwargs, exp_type='', record_median=False, record_iqm=False,
                     with_deviation=False, ax=None):
        if isinstance(plt_kwargs, dict):
            plt_kwargs = [plt_kwargs.copy() for _ in range(len(evals))]

        assert len(evals) == len(plt_kwargs)
        for i, ((k, npz), kwargs) in enumerate(zip(evals.items(), plt_kwargs)):
            if not npz:
                continue

            if 'color' not in kwargs:
                kwargs['color'] = COLORS[i % len(COLORS)]

            if exp_type == 'tsf' or isinstance(npz, list):
                for j, npz_tsf in enumerate(npz):
                    kwargs_tsf = kwargs.copy()

                    if 'ls' not in kwargs and 'linestyle' not in kwargs:
                        ls_idx = j % (len(PLOT_LINE_STYLES) - 1) + 1 if exp_type == 'tsf' else j % len(PLOT_LINE_STYLES)
                        kwargs_tsf['ls'] = PLOT_LINE_STYLES[ls_idx]

                    self.__plot_single_eval(npz_tsf, k, exp_type=exp_type + f'{j + 1}', record_median=record_median,
                                            record_iqm=record_iqm,
                                            with_deviation=with_deviation, ax=ax, **kwargs_tsf)

            else:
                if exp_type == 'tst':
                    if 'ls' not in kwargs and 'linestyle' not in kwargs:
                        kwargs['ls'] = '-.'

                self.__plot_single_eval(npz, k, exp_type=exp_type, record_median=record_median, record_iqm=record_iqm,
                                        with_deviation=with_deviation, ax=ax, **kwargs)

    def __plot_single_eval(self, res_dict, eval_key=None, exp_type='', record_median=False, record_iqm=False,
                           with_deviation=False,
                           ax=None, **plt_kwargs):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 7))

        x = res_dict['timesteps']
        ys = res_dict['results']

        if record_iqm:
            to_plot = np.array([self.__nan_iqm(yy) for yy in ys])
        elif record_median:
            to_plot = np.array([np.nanmedian(yy) for yy in ys])
        else:
            to_plot = np.array([np.nanmean(yy) for yy in ys])

        if 'label' not in plt_kwargs:
            plt_kwargs['label'] = str(eval_key)
        if exp_type:
            plt_kwargs['label'] += f' {exp_type}'
        ax.plot(x, to_plot, **plt_kwargs)

        if with_deviation:
            if record_median:
                perc_25 = np.array([np.percentile(yy, 25) for yy in ys])
                perc_75 = np.array([np.percentile(yy, 75) for yy in ys])
                ax.fill_between(x, perc_25, perc_75, alpha=0.2, color=plt_kwargs['color'])
            else:
                stds = np.array([np.std(yy) for yy in ys])
                ax.fill_between(x, to_plot - stds, to_plot + stds, alpha=0.2, color=plt_kwargs['color'])

    def get_path_results(self, idx_to_tsk_idx, idx_to_path, record_returns):
        val_key = RETURNS_KEY if record_returns else RESULTS_KEY

        res = {}
        # path_to_agg_idx = {v: k for k, v in idx_to_path.items()}
        for agg_idx, task_to_idx in tqdm(idx_to_tsk_idx.items(), desc='getting all results'):
            per_idx_results = self.load_task_results_for_indices(task_to_idx)

            for tsk in task_to_idx:
                # aggregate episode data
                res.setdefault(agg_idx, {})[tsk] = self.mean_discounted_rewards(per_idx_results, tsk, val_key)

            # # aggregate episode data
            # if self.__path_is_tgt(p):
            #     res[path_to_agg_idx[p]] = {
            #         TGT_KEY: self.mean_discounted_rewards(per_idx_results, TGT_KEY, val_key),
            #         TSF_KEY: self.mean_discounted_rewards(per_idx_results, TSF_KEY, val_key),
            #     }
            # else:
            #     res[path_to_agg_idx[p]] = {
            #         SRC_KEY: self.mean_discounted_rewards(per_idx_results, SRC_KEY, val_key),
            #     }
            #
            #     if self.exp_obj_dict[path_to_agg_idx[p]][0].cfg.exp_kwargs['use_tgt_for_test']:
            #         res[path_to_agg_idx[p]][TST_KEY] = self.mean_discounted_rewards(per_idx_results, TST_KEY, val_key)

        return res

    def mean_discounted_rewards(self, npz_res_dict, stage_key, val_key):
        returns = [
            npz_res_dict[i][stage_key][val_key]
            for i in npz_res_dict
            if stage_key in npz_res_dict[i]
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
            TIMESTEPS_KEY: max([npz_res_dict[i][stage_key][TIMESTEPS_KEY]
                                for i in npz_res_dict
                                if stage_key in npz_res_dict[i]], key=lambda v: len(v)),
            RESULTS_KEY: np.nanmean(padded_returns, axis=-1).T  # transpose to make it [timestep, cfg]
        }

    def load_task_results_for_indices(self, task_to_idx):
        results = {}
        for tsk, idx in task_to_idx.items():
            for i in idx:
                exp = self.idx_to_exp_objs[i][tsk]

                true_tsk_name = TSF_TASK_NAME if tsk.startswith(TSF_TASK_NAME) else tsk

                results.setdefault(i, {})[tsk] = self.load_exp_eval_in_path(exp.eval_log_dir(true_tsk_name))

        return results

    def load_exp_eval_in_path(self, p):
        return np.load(p / EVALUATIONS_FILE, allow_pickle=True)

    def plot_experiments_training(self):
        # TODO
        pass

    def print_experiments(self, exp_agg_type=None, cfg_constraints=None):
        path_to_idx, idx_to_path = self.get_agg_specific_idx_maps(exp_agg_type=exp_agg_type)
        idx_to_task_idx = self.__filter_out_unconstrained(path_to_idx, idx_to_path, cfg_constraints)

        print(f'num experiments: {len(idx_to_task_idx)}\n')
        for i, tsk_to_idx in idx_to_task_idx.items():
            print(f'{i}: {idx_to_path[i]}')
            for tsk, tsk_idx in tsk_to_idx.items():
                print(' ' * len(f'{i}: ') + tsk + f': {len(tsk_idx)} runs')
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

    def __filter_out_unconstrained(self, path_to_idx, idx_to_path, constraints):
        # if not constraints:
        #     return path_to_idx_d  # don't go over items if no constraints

        # for each path keep only configurations that uphold constraints
        d = {}
        for i, p in idx_to_path.items():
            for j in path_to_idx[p]:
                for tsk, exp in self.idx_to_exp_objs[j].items():
                    if self.__check_cfg_constraints_for_exp(exp, constraints):
                        d.setdefault(i, {}).setdefault(tsk, []).append(j)

        d = {k: v for k, v in d.items() if v}  # paths with no matching experiments

        return d

    def load_all_run_dump_paths(self):
        done_exps = {}
        failed_exps = {}
        inc_exps = {}

        dump_dirs = []
        with tqdm(desc='collecting dumps') as pb:
            for root, dirs, files in os.walk(self.exp_dump_dir):
                if SRC_TASK_NAME in dirs:  # a dump dir
                    dump_dirs.append(root)
                    pb.update()

        for dp in tqdm(dump_dirs, desc='arranging dumps'):
            dp_path = Path(dp)

            for tsk in [SRC_TASK_NAME, TST_TASK_NAME, TGT_TASK_NAME]:
                tsk_dir = dp_path / tsk
                if (tsk_dir / DONE_FILE).exists() and not (tsk_dir / FAIL_FILE).exists():
                    done_exps.setdefault(dp, []).append(tsk)
                elif (tsk_dir / FAIL_FILE).exists():
                    failed_exps.setdefault(dp, []).append(tsk)
                else:
                    inc_exps.setdefault(dp, []).append(tsk)

            all_tsf_dir = dp_path / TSF_TASK_NAME
            for tsf_kwargs in map(lambda x: x.name, all_tsf_dir.iterdir()):
                if not tsf_kwargs.startswith('tsf_kwargs'):
                    continue
                tsf_dir = all_tsf_dir / tsf_kwargs
                tsf_specific_task = f'{TSF_TASK_NAME}/{tsf_kwargs}'
                if (tsf_dir / DONE_FILE).exists() and not (tsf_dir / FAIL_FILE).exists():
                    done_exps.setdefault(dp, []).append(tsf_specific_task)
                elif (tsf_dir / FAIL_FILE).exists():
                    failed_exps.setdefault(dp, []).append(tsf_specific_task)
                else:
                    inc_exps.setdefault(dp, []).append(tsf_specific_task)

        return done_exps, failed_exps, inc_exps

    def make_exps_for_path(self, path):
        exp_repr = path.replace(str(self.exp_dump_dir) + '/', '', 1)

        out = {}
        for tsk in self.done_exps[path]:
            if tsk.startswith(TSF_TASK_NAME):
                exp_tsk_repr = exp_repr + '/' + tsk.split('/', 1)[1]
            else:
                exp_tsk_repr = exp_repr + '/tsf_kwargs-((no_transfer-True))'

            cfg = TransferConfiguration.from_repr_value(exp_tsk_repr)

            out[tsk] = Experiment(cfg)

        return out

    @staticmethod
    def __path_is_tgt(p):
        return 'tsf_kwargs' in str(p)

    @staticmethod
    def __nan_iqm(data):
        # clean data from `nan` values
        data = data[~np.isnan(data)]

        # not supported for tiny dataset.
        # return regular mean
        if len(data) < 3:
            return np.mean(data)

        # calculate data within IQR
        q3, q1 = np.percentile(data, [75, 25])
        iqr_data = data[(q1 <= data) & (data <= q3)]

        # calculate mean of data in IQR (aka IQM)
        iqm = np.mean(iqr_data)

        return iqm