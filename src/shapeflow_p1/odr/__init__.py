"""Adapter layer over the pinned Open Deep Research vendor.

Split by dependency on purpose:

- ``close_reason``, ``checkpoints`` and ``hooks`` are **langchain-free**. They model the
  frozen state the patch captures and the strategy contract it dispatches to, using plain
  serializable data and content-addressed hashes. They import nothing from vendor, so they
  run and are tested in the light dev environment.

- ``adapter`` and the concrete P0/P1 strategies live at the boundary where langchain and
  the patched vendor are importable. ``adapter`` is the only place that converts between
  langchain message objects and the frozen dataclasses here.

Keeping the checkpoint/strategy shapes independent of langchain also makes them stable
against langchain version churn: a checkpoint is bytes and structure, never a live
``AIMessage`` whose repr could shift under a dependency bump.
"""
