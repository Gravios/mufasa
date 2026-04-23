"""Mufasa Qt inline operation forms — replace popup-per-operation with
consolidated forms hosted inside workbench pages.

Adding a new form:

1. Subclass :class:`mufasa.ui_qt.workbench.OperationForm`.
2. Set ``title`` and optional ``description``.
3. Implement :meth:`build` to populate ``self.body_layout``.
4. Implement :meth:`collect_args` (validate inputs, return a dict).
5. Implement :meth:`target(**kwargs)` — runs in a worker thread.
6. Register the form on a :class:`mufasa.ui_qt.workbench.WorkflowPage`
   via ``page.add_section("Section name", [(MyForm, {})])``.
"""
