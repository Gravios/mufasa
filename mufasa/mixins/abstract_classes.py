from __future__ import annotations

import os
from abc import ABC, abstractmethod

import pandas as pd


class AbstractFeatureExtraction(ABC):

    @abstractmethod
    def __init__(self, config_path: str | os.PathLike):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def save(self, data: pd.DataFrame, save_path: str | os.PathLike):
        pass
