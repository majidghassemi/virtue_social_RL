"""Agent roles: index 0 is the teacher, then learners, then bystanders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TEACHER = 0


@dataclass(frozen=True)
class RoleConfig:
    num_learners: int = 3
    num_bystanders: int = 3

    def __post_init__(self):
        if self.num_learners < 1:
            raise ValueError("need at least one learner")
        if self.num_bystanders < 0:
            raise ValueError("num_bystanders must be >= 0")

    @property
    def num_agents(self) -> int:
        return 1 + self.num_learners + self.num_bystanders

    @property
    def learner_ids(self) -> np.ndarray:
        return np.arange(1, 1 + self.num_learners)

    @property
    def bystander_ids(self) -> np.ndarray:
        return np.arange(1 + self.num_learners, self.num_agents)

    def masks(self) -> dict[str, np.ndarray]:
        n = self.num_agents
        teacher = np.zeros(n, bool)
        teacher[TEACHER] = True
        learner = np.zeros(n, bool)
        learner[self.learner_ids] = True
        bystander = np.zeros(n, bool)
        bystander[self.bystander_ids] = True
        return {"teacher": teacher, "learner": learner, "bystander": bystander}

    @classmethod
    def from_dict(cls, d: dict) -> "RoleConfig":
        return cls(num_learners=int(d["num_learners"]), num_bystanders=int(d["num_bystanders"]))
