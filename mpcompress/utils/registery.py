"""
Lightweight registry and dynamic class instantiation utilities.

Supported resolution strategies (in order):
1) Fully-qualified name: "package.module.Class" via importlib
2) Short name: look up from the local registry (registered with @register("Name"))

Notes:
- No wildcard imports; no implicit module scanning. This keeps behavior explicit
  and avoids circular imports.
- Prefer using fully-qualified names in configs for maximum clarity.
"""

import importlib
import sys
from typing import Any, Dict, Type, Optional, Callable


class Registry:
    """Simple name -> class registry."""

    def __init__(self):
        self._registry: Dict[str, Type] = {}

    def register(self, name: str) -> Callable:
        """Decorator to register a class under the given name."""

        def decorator(cls: Type) -> Type:
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Optional[Type]:
        """Return the class by name, or None if not found."""
        return self._registry.get(name)

    def list_registered(self) -> list:
        """Return a list of all registered names."""
        return list(self._registry.keys())


_mpcompress_registry = Registry()


def register(name: str):
    """Public decorator to register a class in the global registry."""
    return _mpcompress_registry.register(name)


def list_registered_classes() -> list:
    """List all names currently registered in the global registry."""
    return _mpcompress_registry.list_registered()


def get_obj_from_str(string: str, reload: bool = False) -> Type:
    """Resolve a class object from string.

    Two supported formats:
    1) Fully-qualified: "package.module.Class" (preferred)
    2) Short name: "Class" — resolved from the local registry only
    """
    if "." in string:
        # Fully-qualified import using importlib (sys.modules first, then import)
        module_name, class_name = string.rsplit(".", 1)
        module_imp = sys.modules.get(module_name)
        if module_imp is None:
            module_imp = importlib.import_module(module_name)

        if module_imp is None:
            raise ImportError(f"Module '{module_name}' cannot be imported")
        if reload:
            importlib.reload(module_imp)
        return getattr(module_imp, class_name)
    else:
        # Short name: only resolve from registry
        registered_cls = _mpcompress_registry.get(string)
        if registered_cls is not None:
            return registered_cls
        raise ImportError(
            f"'{string}' not found in registry. Use a fully-qualified name "
            "like 'package.module.Class' or register it via @register(name)."
        )


def instantiate_class(config: Dict[str, Any], **kwargs) -> Any:
    """Instantiate a class from a config dict.

    Args:
        config: dict that must contain the key 'type' (class path or registry name)
        **kwargs: extra keyword arguments passed to the class constructor

    Returns:
        Instantiated object

    Raises:
        KeyError: if 'type' is missing in the config
        ImportError: if the type cannot be resolved to a class
    """
    config = config.copy()
    if "type" not in config:
        raise KeyError(f"Expected key 'type' to instantiate. Got config: {config}")

    cls_name = config.pop("type")
    try:
        cls = get_obj_from_str(cls_name)
        return cls(**config, **kwargs)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot import class '{cls_name}': {e}")


def instantiate_transforms(config, **kwargs):
    """Instantiate a transform pipeline from config.

    This expects a top-level 'type' and optionally a nested 'transforms' list,
    where each item is itself a config for another transform.
    """
    if "type" not in config:
        raise KeyError("Expected key 'type' to instantiate.")

    config = config.copy()
    cls = config.pop("type")
    transform_cls = get_obj_from_str(cls)

    if "transforms" in config:
        items = config.pop("transforms")
        transforms = [instantiate_class(t) for t in items]
        return transform_cls(transforms=transforms, **config, **kwargs)
    return transform_cls(**config, **kwargs)
