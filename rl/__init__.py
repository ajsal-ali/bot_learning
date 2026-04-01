"""
GO-BDX Reinforcement Learning Package

Contains:
    - GoBdxEnv: Gymnasium environment for bipedal walking
    - CurriculumScheduler: Success-based stage progression
    - Training scripts and utilities
"""

from .go_bdx_env import GoBdxEnv
from .curriculum import CurriculumScheduler

__all__ = ['GoBdxEnv', 'CurriculumScheduler']
