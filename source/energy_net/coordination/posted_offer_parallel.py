"""Spawn-only parallel evaluation of independent deterministic controls.

Workers receive paths, a configuration, integer case IDs and static template
parameters. Live factories, neural models and initialized HiGHS objects are
never forked or pickled. Random-response calibration remains in the caller.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from multiprocessing import get_context
import os
from pathlib import Path
import sys


_WORKER_FACTORY = None
_WORKER_USERS = None
_WORKER_EVALUATION = None
_WORKER_INITIALIZATIONS = 0
_WORKER_THREAD_LIMIT = None


def _initialize_worker(legacy_path, users_path, config):
    global _WORKER_FACTORY, _WORKER_USERS, _WORKER_EVALUATION
    global _WORKER_INITIALIZATIONS, _WORKER_THREAD_LIMIT
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _WORKER_THREAD_LIMIT = threadpool_limits(limits=1)
    scripts = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0,scripts)
    _WORKER_EVALUATION = import_module("evaluate_posted_offers")
    _WORKER_FACTORY = _WORKER_EVALUATION.posted_episode_factory(Path(legacy_path),config=config)
    _WORKER_USERS = _WORKER_EVALUATION.load_users(Path(users_path))
    _WORKER_INITIALIZATIONS += 1


@dataclass(frozen=True)
class _CandidateTask:
    family: str
    candidate_index: int
    parameters: dict
    cases: tuple


@dataclass(frozen=True)
class _CandidateResult:
    family: str
    candidate_index: int
    rows: list
    process_id: int
    initializations: int


def _evaluate_candidate(task):
    evaluation = _WORKER_EVALUATION
    if evaluation is None:
        raise RuntimeError("A deterministic-template worker must be initialized")
    if task.family=="raw":
        controller = evaluation.FixedRawOffer(task.parameters["action"])
    elif task.family=="matrix":
        controller = evaluation.matrix_controller(task.parameters)
    elif task.family=="value_share":
        controller = evaluation.ValueShareGreedy(task.parameters["rho"])
    else:
        raise ValueError("Only raw, matrix and value-share deterministic templates can run in workers")
    rows,_ = evaluation.run_policy(_WORKER_FACTORY,_WORKER_USERS,task.cases,
        f"{task.family}_{task.candidate_index}",controller=controller)
    return _CandidateResult(task.family,task.candidate_index,rows,os.getpid(),_WORKER_INITIALIZATIONS)


class SpawnedTemplateEvaluator:
    """Persistent factory/users per spawned process, ordered rows per template.

    This callable implements the optional ``select_controls`` candidate
    evaluator. It does not calculate means, choose winners, fit response
    models, write results or evaluate extra user alternatives. Those operations
    remain in the original parent selector and calibration code.
    """

    def __init__(self,legacy_path,users_path,*,config=None,workers=4):
        if isinstance(workers,bool) or int(workers)!=workers or workers<1:
            raise ValueError("Template worker count must be a positive integer")
        self.workers = int(workers)
        self.legacy_path,self.users_path = str(Path(legacy_path)),str(Path(users_path))
        self.config = config
        self.worker_initializations = {}
        self._pool = None

    def __enter__(self):
        if self._pool is not None:
            raise RuntimeError("The template evaluator is already running")
        self._pool = ProcessPoolExecutor(max_workers=self.workers,
            mp_context=get_context("spawn"),initializer=_initialize_worker,
            initargs=(self.legacy_path,self.users_path,self.config))
        return self

    def __exit__(self,exc_type,exc_value,traceback):
        if self._pool is not None:
            self._pool.shutdown(wait=True,cancel_futures=exc_type is not None)
            self._pool = None

    def __call__(self,family,parameters,cases):
        if self._pool is None:
            raise RuntimeError("Use SpawnedTemplateEvaluator as a context manager")
        if family not in ("raw","matrix","value_share"):
            raise ValueError("Only deterministic template families may be parallelized")
        tasks = tuple(_CandidateTask(family,order,deepcopy(values),tuple(int(case) for case in cases))
                      for order,values in enumerate(parameters))
        # map preserves submission order even if workers finish out of order.
        for expected,result in zip(tasks,self._pool.map(_evaluate_candidate,tasks,chunksize=1)):
            if (result.family,result.candidate_index)!=(expected.family,expected.candidate_index):
                raise RuntimeError("A template result lost its declared candidate order")
            self.worker_initializations[result.process_id] = result.initializations
            yield result.rows
