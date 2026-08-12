"""Compatibility names needed to read the first public experimental models.

The four bundled N31 pipelines were serialized before the fold-local reducer
received its stable public module. Python's ``joblib`` stores a class by its
module and class name, so old files refer to
``_bundled_fold_local_reducer.FoldLocalSpearmanReducer``. Registering that
one legacy module name here lets a public checkout load the old artifacts
without importing the historical N31 training runner or following a local
environment-variable override.

New model artifacts should import ``FoldLocalSpearmanReducer`` directly from
``scripts.koff_ml.fold_local_spearman_reducer`` and will not need this bridge.
"""

from __future__ import annotations

import sys
from types import ModuleType

from scripts.koff_ml.fold_local_spearman_reducer import FoldLocalSpearmanReducer


def install_legacy_reducer_alias() -> None:
    """Expose the one historical pickle name through the stable public class."""

    module_name = "_bundled_fold_local_reducer"
    if module_name in sys.modules:
        return
    module = ModuleType(module_name)
    module.FoldLocalSpearmanReducer = FoldLocalSpearmanReducer
    sys.modules[module_name] = module
