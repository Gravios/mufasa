"""
mufasa.ui_qt.pages.features_page
================================

Features workbench page. Hosts the feature-subsets extractor.

Previous iterations carried an empty "Full feature extraction (full
project)" placeholder section pending an inline-form port. Empty
sections in the UI look like broken features rather than
unimplemented ones, so it was removed. When the full-project
extractor is wired inline it can be appended here.
"""
from __future__ import annotations

from typing import Optional

from mufasa.ui_qt.forms.features import FeatureSubsetExtractorForm
from mufasa.ui_qt.workbench import WorkflowPage


def build_features_page(workbench,
                        config_path: Optional[str] = None) -> WorkflowPage:
    page = workbench.add_page("Features", icon_name="features")
    page.add_section("Compute feature subsets",
                     [(FeatureSubsetExtractorForm, {})])
    return page


__all__ = ["build_features_page"]
