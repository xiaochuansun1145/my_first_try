"""Minimal semantic communication components for encoder feature transmission."""

from src.semantic_rtdetr.semantic_comm.mdvsc import MDVSCOutput, ProjectMDVSC
from src.semantic_rtdetr.semantic_comm.stage2_model import Stage2MDVSC, Stage2Output

__all__ = ["ProjectMDVSC", "MDVSCOutput", "Stage2MDVSC", "Stage2Output"]