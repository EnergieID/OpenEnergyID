"""
Dynamic Pydantic models for PVLib classes.

This module provides utilities to generate Pydantic models from PVLib classes,
enabling serialization, validation, and conversion between Pydantic and PVLib objects.
It also provides helper functions for recursive conversion and type overlays for
special PVLib constructs.
"""

from collections.abc import Mapping, Sequence
from inspect import Parameter, isclass, signature
from typing import Annotated, Any, Literal, Union

from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import Array, FixedMount, PVSystem, SingleAxisTrackerMount
from pydantic import BaseModel, ConfigDict, Field, create_model
from typing_inspect import get_args, get_origin

# Registry of PVLib classes for which models can be generated
REGISTRY: dict[str, type] = {
    "Location": Location,
    "FixedMount": FixedMount,
    "SingleAxisTrackerMount": SingleAxisTrackerMount,
    "Array": Array,
    "PVSystem": PVSystem,
    "ModelChain": ModelChain,
}

# Cache for generated Pydantic models
MODEL_CACHE: dict[type, type[BaseModel]] = {}


def normalize_annotation(ann: Any) -> Any:
    """
    Normalize type annotations for use in Pydantic models.

    Converts unknown or third-party types (e.g., numpy, pandas) to Any.
    """
    if ann in (Parameter.empty, None):
        return Any
    try:
        mod = getattr(ann, "__module__", "")
        if mod.startswith(("numpy", "pandas")):
            return Any
    except Exception:
        pass
    return ann


# Overlay for type hints that need to be replaced in generated models
TYPE_OVERLAY: dict[type, dict[str, Any]] = {}


def pyd_model_from_type(py_type: type) -> type[BaseModel]:
    """
    Recursively create or retrieve a Pydantic model for a given PVLib class.

    Args:
        py_type: The PVLib class type.

    Returns:
        A dynamically created Pydantic model class.
    """
    if py_type in MODEL_CACHE:
        return MODEL_CACHE[py_type]

    # Use the __init__ signature to determine fields
    sig = signature(py_type.__init__)
    fields = {}

    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        default = Field(...) if param.default is Parameter.empty else Field(param.default)

        # Use overlay type if available, otherwise normalized annotation
        ann = TYPE_OVERLAY.get(py_type, {}).get(pname, normalize_annotation(param.annotation))

        origin = get_origin(ann)

        # Recursively generate models for PVLib class fields
        if isclass(ann) and ann in REGISTRY.values():
            ann = pyd_model_from_type(ann)

        # Handle lists/unions of PVLib classes
        elif origin in (list, list):
            (t,) = get_args(ann) or (Any,)
            if isclass(t) and t in REGISTRY.values():
                t = pyd_model_from_type(t)  # type: ignore
            ann = list[t]
        elif origin in (Union,):
            args = []
            for t in get_args(ann):
                if isclass(t) and t in REGISTRY.values():
                    t = pyd_model_from_type(t)
                args.append(t)
            ann = Union[tuple(args)]  # type: ignore

        fields[pname] = (ann, default)

    # Create the Pydantic model dynamically
    model = create_model(
        py_type.__name__ + "Model",
        __base__=BaseModel,
        __config__=ConfigDict(extra="allow"),
        __doc__=py_type.__doc__,
        **fields,
    )
    # Attach a back-reference to the PVLib class
    setattr(model, "_pvlib_type", py_type)
    MODEL_CACHE[py_type] = model
    return model


def _filter_kwargs(pv_type: type, kwargs: dict) -> dict:
    """
    Remove keys from kwargs that are not accepted by the PVLib class constructor.

    Args:
        pv_type: The PVLib class type.
        kwargs: Dictionary of keyword arguments.

    Returns:
        Filtered dictionary with only accepted keys.
    """
    sig = signature(pv_type.__init__)
    params = sig.parameters
    if any(p.kind is Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs  # accepts **kwargs, keep all
    allowed = {n for n in params if n != "self"}
    return {k: v for k, v in kwargs.items() if k in allowed}


def to_pv(obj: Any) -> Any:
    """
    Recursively convert a Pydantic model (or nested structure) to a PVLib object.

    Handles BaseModel instances, mappings, sequences, and primitives.

    Args:
        obj: The object to convert.

    Returns:
        The corresponding PVLib object or primitive.
    """
    # 1) Pydantic model → recurse on attributes (NOT model_dump)
    if isinstance(obj, BaseModel):
        pv_type = getattr(obj.__class__, "_pvlib_type", None)

        # Collect declared fields
        kwargs = {name: to_pv(getattr(obj, name)) for name in obj.__class__.model_fields}

        # Include extras if extra="allow"
        extra = getattr(obj, "__pydantic_extra__", None)
        if isinstance(extra, dict):
            for k, v in extra.items():
                kwargs[k] = to_pv(v)

        if pv_type is None:
            # Not a pvlib-backed model: return plain nested dict
            return kwargs

        # Special case: ModelChain(system, location, **kwargs)
        if pv_type is ModelChain:
            system = kwargs.pop("system")
            location = kwargs.pop("location")
            return ModelChain(system, location, **_filter_kwargs(pv_type, kwargs))

        # General case: keyword construction
        return pv_type(**_filter_kwargs(pv_type, kwargs))

    # 2) Containers
    if isinstance(obj, Mapping):
        return {k: to_pv(v) for k, v in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return type(obj)(to_pv(v) for v in obj)  # type: ignore

    # 3) Primitives
    return obj


# --- Model definitions for PVLib classes ---

# Location model
LocationModel = pyd_model_from_type(Location)

# FixedMount model with discriminator field
FixedMountModel = create_model(
    "FixedMountModel",
    kind=(Literal["fixed"], Field("fixed", frozen=True)),  # tag
    __base__=pyd_model_from_type(FixedMount),
    __config__=ConfigDict(extra="allow"),
)

# SingleAxisTrackerMount model with discriminator field
TrackerMountModel = create_model(
    "SingleAxisTrackerMountModel",
    kind=(Literal["tracker"], Field("tracker", frozen=True)),
    __base__=pyd_model_from_type(SingleAxisTrackerMount),
    __config__=ConfigDict(extra="allow"),
)

# Union type for mount models, with discriminator
MountUnion = Annotated[Union[FixedMountModel, TrackerMountModel], Field(discriminator="kind")]

# Overlay Array.mount type with MountUnion
TYPE_OVERLAY.update({Array: {"mount": MountUnion}})

# Array model
ArrayModel = pyd_model_from_type(Array)

# Overlay PVSystem.arrays type to allow list, single, or None
TYPE_OVERLAY.update({PVSystem: {"arrays": list[ArrayModel] | ArrayModel | None}})

# PVSystem model
PVSystemModel = pyd_model_from_type(PVSystem)

# Overlay ModelChain system/location types
TYPE_OVERLAY.update(
    {
        ModelChain: {
            "system": PVSystemModel,
            "location": LocationModel,
        }
    }
)

# ModelChain model
ModelChainModel = create_model(
    "ModelChainModel",
    type=(Literal["modelchain"], Field("modelchain", frozen=True)),  # tag
    __base__=pyd_model_from_type(ModelChain),
    __config__=ConfigDict(extra="allow"),
)
