from sklearn.model_selection import LeavePOut
from tqdm.auto import tqdm

from .configurations import DEFAULT_NUM_TGT_SAMPLES_FOR_CV
from .with_transfer import WithTransferExperiment


class CVTransferExperiment(WithTransferExperiment):
    def __init__(self, *args, num_tgt_samples=DEFAULT_NUM_TGT_SAMPLES_FOR_CV, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tgt_samples = num_tgt_samples
        self._cur_fold = None

    def run(self, *contexts):
        for context_set in tqdm(contexts, desc='contexts for CV'):
            self.run_single_context_set(context_set)

    def run_single_context_set(self, context_set):
        # split data by choosing all combinations of src and tgt context sets according to the given sizes
        lpo = LeavePOut(p=self.num_tgt_samples)

        # iterate splits
        for self._cur_fold, (src_idx, tgt_idx) in enumerate(tqdm(lpo.split(context_set),
                                                                 total=lpo.get_n_splits(context_set),
                                                                 desc='context set CV splits')):
            # use indices to extract src and tgt context sets
            src_context_set = [context_set[i] for i in src_idx]
            tgt_context_set = [context_set[i] for i in tgt_idx]

            # run WithTransferExperiment as intended with current fold
            super(CVTransferExperiment, self).run(src_context_set, tgt_context_set)

        self._cur_fold = None

    @property
    def exp_dump_dir(self):
        orig_dir = super().dump_dir
        return orig_dir if self._cur_fold is None else orig_dir / f'fold-{self._cur_fold}'
