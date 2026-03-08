r"""Batch inference and evaluation entry point for music transcription.

Usage example::

    from torch.utils.data import DataLoader
    from audio_understanding.datasets.maestro import MAESTRO
    from audio_understanding.eval.transcription.batch_eval import batch_evaluate

    dataset = MAESTRO(root="/data/maestro-v3.0.0", split="test", crop=None)

    def my_inference_fn(data):
        # ... run model on data["audio"] ...
        return tokens  # list[str]

    results = batch_evaluate(
        dataset=dataset,
        inference_fn=my_inference_fn,
        fps=100,
    )
    print(results["note_onset"])   # {"precision": ..., "recall": ..., "f1": ...}
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from audio_understanding.eval.transcription.metrics import instrument_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _avg_metric(metrics_list: list[dict]) -> dict:
    r"""Average precision / recall / F1 across a list of metric dicts."""
    if not metrics_list:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": float(np.mean([m["precision"] for m in metrics_list])),
        "recall":    float(np.mean([m["recall"]    for m in metrics_list])),
        "f1":        float(np.mean([m["f1"]        for m in metrics_list])),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def batch_evaluate(
    dataset,
    inference_fn: Callable[[dict], list[str]],
    fps: float,
    include_program: bool = False,
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    r"""Run inference and evaluation over every sample in *dataset*.

    The function iterates through *dataset*, calls *inference_fn* on each
    item to obtain a token list, then delegates to the dataset's own
    ``evaluate`` method for per-sample scoring.  After all samples are
    processed, per-sample scores are averaged and (for Slakh2100 with
    instrument metadata) an instrument summary is produced.

    Args:
        dataset: a :class:`~audio_understanding.datasets.maestro.MAESTRO` or
            :class:`~audio_understanding.datasets.slakh2100.Slakh2100`
            instance.  The dataset **must** expose an ``evaluate`` method.
        inference_fn: callable that accepts a single ``__getitem__`` output
            dict and returns a flat list of MIDI token strings (model output).
        fps: frames-per-second value used when encoding tokens (must match the
            value used during training / inference).
        include_program: whether the model output tokens include
            ``program=X`` fields.  Passed through to
            :meth:`Slakh2100.evaluate`; ignored for MAESTRO.
        max_samples: optional cap on the number of samples evaluated.
        verbose: print a progress bar with ``tqdm`` when available.

    Returns:
        Dict with the following keys (depending on dataset type):

        * ``"n_samples"``       – number of successfully evaluated samples
        * ``"note_onset"``      – average onset F1 dict
        * ``"note_offset"``     – average offset F1 dict
        * ``"program_aware"``   – average program-aware F1 dict (*when
          available*)
        * ``"instrument_summary"`` – output of
          :func:`~audio_understanding.eval.transcription.metrics.instrument_summary`
          (*when per-instrument data is available*)
    """
    n_total = len(dataset)
    if max_samples is not None:
        n_total = min(n_total, max_samples)

    dataset_name = type(dataset).__name__
    is_slakh = dataset_name == "Slakh2100"

    # Per-sample accumulators
    onset_list: list[dict] = []
    offset_list: list[dict] = []
    program_aware_list: list[dict] = []
    per_inst_list: list[dict] = []

    iterator = range(n_total)
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"Evaluating {dataset_name}")
        except ImportError:
            pass

    for i in iterator:
        data = dataset[i]

        try:
            tokens = inference_fn(data)
        except Exception as exc:
            if verbose:
                print(f"[batch_evaluate] inference failed for sample {i}: {exc}")
            continue

        try:
            eval_kwargs: dict = {"data": data, "output_tokens": tokens, "fps": fps}
            if is_slakh:
                eval_kwargs["include_program"] = include_program
            result = dataset.evaluate(**eval_kwargs)
        except Exception as exc:
            if verbose:
                print(f"[batch_evaluate] evaluation failed for sample {i}: {exc}")
            continue

        onset_list.append(result["note_onset"])
        offset_list.append(result["note_offset"])

        if "program_aware" in result:
            program_aware_list.append(result["program_aware"])
        if "per_instrument" in result:
            per_inst_list.append(result["per_instrument"])

    summary: dict = {
        "n_samples": len(onset_list),
        "note_onset": _avg_metric(onset_list),
        "note_offset": _avg_metric(offset_list),
    }

    if program_aware_list:
        summary["program_aware"] = _avg_metric(program_aware_list)

    if per_inst_list:
        summary["instrument_summary"] = instrument_summary(per_inst_list)

    return summary
