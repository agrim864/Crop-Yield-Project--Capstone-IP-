from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG


class YearForwardSplit:
    """
    Forward-chaining split by year.

    For each fold:
      train: years < test_year
      test : year == test_year

    Uses the last n_splits years as test folds (when possible).
    Skips folds that don't meet min sample requirements.
    """

    def __init__(
        self,
        n_splits: int = 4,
        min_train_samples: int | None = None,
        min_test_samples: int | None = None,
    ):
        self.n_splits = int(n_splits)
        # default: use CONFIG.cv.min_samples_cv if provided
        ms = int(getattr(CONFIG.cv, "min_samples_cv", 8))
        self.min_train_samples = int(min_train_samples) if min_train_samples is not None else ms
        self.min_test_samples = int(min_test_samples) if min_test_samples is not None else max(2, ms // 2)

    def split(self, X: pd.DataFrame, y=None, groups=None):
        years = pd.to_numeric(X["year"], errors="coerce").to_numpy()
        mask_ok = np.isfinite(years)
        years = years[mask_ok].astype(int)

        idx_all = np.arange(len(X))[mask_ok]
        unique_years = np.sort(np.unique(years))

        if len(unique_years) < 3:
            return

        # choose test years
        if len(unique_years) < self.n_splits + 2:
            # fallback: last year as test, everything before as train
            test_years = [unique_years[-1]]
        else:
            test_years = list(unique_years[-self.n_splits:])

        yielded_any = False

        for ty in test_years:
            train_mask = years < ty
            test_mask = years == ty

            train_idx = idx_all[train_mask]
            test_idx = idx_all[test_mask]

            if len(train_idx) < self.min_train_samples:
                continue
            if len(test_idx) < self.min_test_samples:
                continue

            yielded_any = True
            yield train_idx, test_idx

        # final fallback if everything got skipped
        if not yielded_any and len(unique_years) >= 2:
            cutoff = unique_years[-2]
            train_idx = np.where(pd.to_numeric(X["year"], errors="coerce").to_numpy() <= cutoff)[0]
            test_idx = np.where(pd.to_numeric(X["year"], errors="coerce").to_numpy() == unique_years[-1])[0]
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
