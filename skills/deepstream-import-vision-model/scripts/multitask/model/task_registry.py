#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Canonical public categories and internal execution-task aliases."""

from __future__ import annotations


SUPPORTED_CATEGORIES = (
    "object-detection",
    "image-classification",
    "semantic-segmentation",
    "2d-pose",
    "monocular-depth",
)

CATEGORY_TO_TASK = {
    "object-detection": "detection",
    "image-classification": "classification",
    "semantic-segmentation": "segmentation",
    "2d-pose": "pose",
    "monocular-depth": "depth",
}

CATEGORY_ALIASES = {
    "detection": "object-detection",
    "object-detection": "object-detection",
    "classification": "image-classification",
    "image-classification": "image-classification",
    "secondary-classification": "image-classification",
    "segmentation": "semantic-segmentation",
    "semantic-segmentation": "semantic-segmentation",
    "pose": "2d-pose",
    "2d-pose": "2d-pose",
    "keypoints": "2d-pose",
    "depth": "monocular-depth",
    "monocular-depth": "monocular-depth",
}

# "generic" is an internal fail-closed sentinel used only when auto-routing
# lacks a declared task. It is not a public category or executable task.
INTERNAL_TASKS = {
    "auto",
    "detection",
    "classification",
    "segmentation",
    "pose",
    "depth",
    "generic",
}

TASK_ALIASES = {
    "classify": "classification",
    "keypoints": "pose",
    **{alias: CATEGORY_TO_TASK[category] for alias, category in CATEGORY_ALIASES.items()},
}

SUPPORTED_TASK_ALIASES = frozenset(CATEGORY_ALIASES)


def task_key(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def category_for(value: str) -> str | None:
    return CATEGORY_ALIASES.get(task_key(value))


def execution_task_for(value: str) -> str | None:
    category = category_for(value)
    return CATEGORY_TO_TASK.get(category) if category else None
