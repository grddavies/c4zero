from typing import Any

import joblib
from tqdm.auto import tqdm


class EasyDict(dict):
    """
    Convenience class that behaves like a dict but allows access with the attribute syntax.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class ProgressParallel(joblib.Parallel):
    """joblib Parallel with additional args to pass to tqdm progress bar"""
    def __init__(self, use_tqdm=True, total=None, leave=True, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        self._leave = leave
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(
            disable=not self._use_tqdm, total=self._total, leave=self._leave
        ) as self._pbar:
            return super(ProgressParallel, self).__call__(*args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
