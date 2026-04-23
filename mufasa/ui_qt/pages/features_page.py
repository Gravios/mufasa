"""
mufasa.ui_qt.pages.features_page
================================

Features workbench page. Hosts the feature-subsets extractor.
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
    # Placeholder — the full feature-extractor run lives elsewhere
    # (usually launched from the project-setup flow); a dedicated form
    # can be added here if/when wanted.
    page.add_section("Full feature extraction (full project)", [])
    return page


__all__ = ["build_features_page"]
