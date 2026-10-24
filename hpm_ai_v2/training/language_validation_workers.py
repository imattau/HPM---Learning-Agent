"""Bounded process-pool helpers for independent validation and evaluation jobs."""
from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

_SWEEP_MODEL = None
_SWEEP_TEXTS = None
_EVAL_MODELS = None
_EVAL_DOCUMENTS = None

def _init_sweep_worker(model, texts):
    global _SWEEP_MODEL, _SWEEP_TEXTS
    _SWEEP_MODEL, _SWEEP_TEXTS = model, texts
def _evaluate_sweep_point(point):
    kind, alpha = point
    if kind == "characters":
        _SWEEP_MODEL.char_alpha = alpha
        return kind, alpha, _SWEEP_MODEL.evaluate_characters(_SWEEP_TEXTS)
    _SWEEP_MODEL.alpha = alpha
    metrics = (_SWEEP_MODEL.evaluate(_SWEEP_TEXTS, include_surface=False)
               if hasattr(_SWEEP_MODEL, "surface_word_probability")
               else _SWEEP_MODEL.evaluate(_SWEEP_TEXTS))
    return kind, alpha, metrics
def _init_evaluation_workers(models, documents):
    global _EVAL_MODELS, _EVAL_DOCUMENTS
    _EVAL_MODELS, _EVAL_DOCUMENTS = models, documents
def _evaluate_metric_job(job):
    name, model_key, metric_type, documents_key = job
    model = _EVAL_MODELS[model_key]
    documents = _EVAL_DOCUMENTS[documents_key]
    if metric_type == "characters":
        metrics = model.evaluate_characters(documents)
    else:
        metrics = model.evaluate(documents)
    return name, metrics
def _run_validation_sweeps(model, texts, points, workers, progress_callback=None):
    """Score independent smoothing values in bounded worker processes."""
    if workers <= 1 or len(points) < 2:
        results = []
        for kind, alpha in points:
            if kind == "characters":
                model.char_alpha = alpha
                metrics = model.evaluate_characters(texts)
            else:
                model.alpha = alpha
                metrics = (model.evaluate(texts, include_surface=False)
                           if hasattr(model, "surface_word_probability")
                           else model.evaluate(texts))
            result = (kind, alpha, metrics)
            results.append(result)
            if progress_callback:
                progress_callback(result, len(results), len(points))
        return results
    # Fork shares the fitted count tables copy-on-write and avoids serializing
    # the large HFN graph into every worker. Fall back to serial on platforms
    # without fork rather than silently changing the resource profile.
    if "fork" not in multiprocessing.get_all_start_methods():
        return _run_validation_sweeps(model, texts, points, 1)
    context = multiprocessing.get_context("fork")
    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(points)), mp_context=context,
                             initializer=_init_sweep_worker, initargs=(model, texts)) as pool:
        futures = [pool.submit(_evaluate_sweep_point, point) for point in points]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if progress_callback:
                progress_callback(result, len(results), len(points))
    point_order = {point: index for index, point in enumerate(points)}
    results.sort(key=lambda result: point_order[(result[0], result[1])])
    return results
def _run_metric_jobs(models, documents, jobs, workers, progress_callback=None):
    """Evaluate independent held-out metrics in bounded forked workers."""
    if workers <= 1 or len(jobs) < 2 or "fork" not in multiprocessing.get_all_start_methods():
        results = []
        for job in jobs:
            name, model_key, metric_type, document_key = job
            model, texts = models[model_key], documents[document_key]
            metrics = model.evaluate_characters(texts) if metric_type == "characters" else model.evaluate(texts)
            result = (name, metrics)
            results.append(result)
            if progress_callback:
                progress_callback(result, len(results), len(jobs))
        return results
    context = multiprocessing.get_context("fork")
    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context,
                             initializer=_init_evaluation_workers, initargs=(models, documents)) as pool:
        futures = [pool.submit(_evaluate_metric_job, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if progress_callback:
                progress_callback(result, len(results), len(jobs))
    return results
