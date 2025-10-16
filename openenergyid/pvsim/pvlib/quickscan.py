"""
Quick PV system and model chain models for rapid PV simulation setup.

Provides convenience Pydantic models for specifying PV system parameters
with simplified fields for module and inverter sizing, and for configuring
a PVLib ModelChain for quick simulations.
"""

from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from .models import ModelChainModel, PVSystemModel


class QuickPVSystemModel(PVSystemModel):
    """
    Model for quickly specifying a PV system with simplified sizing fields.

    Allows specifying module and inverter power directly, and automatically
    derives detailed parameters for use with PVLib.
    """

    p_module: float = Field(default=420, gt=0, description="Module max DC power in W")
    p_inverter: float = Field(
        gt=0,
        description="Inverter max power in W",
    )
    inverter_efficiency: float = Field(
        default=0.96, gt=0, le=1, description="PVWatts efficiency used to derive PDC0 from PAC."
    )
    module_parameters: dict[str, float] = Field(default={"pdc0": 420, "gamma_pdc": -0.003})
    module_type: str = Field(default="glass_polymer", description="Type of PV module")
    racking_model: str = Field(default="open_rack", description="Type of racking model")

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _apply_p_inverter_to_pdc0(self):
        """
        If p_inverter is provided, set inverter_parameters['pdc0'] accordingly.

        Raises:
            ValueError: If both p_inverter and inverter_parameters['pdc0'] are provided.
        """
        p_inverter = self.p_inverter
        inv_params: dict[str, Any] = self.inverter_parameters or {}  # type: ignore

        # choose policy: reject or overwrite if user also sent pdc0
        if "pdc0" in inv_params:
            if inv_params["pdc0"] != p_inverter / self.inverter_efficiency:
                raise ValueError("Provide either 'pac' or 'inverter_parameters.pdc0', not both.")
            return self

        inv_params = dict(inv_params)  # avoid mutating shared defaults
        inv_params["pdc0"] = p_inverter / self.inverter_efficiency
        object.__setattr__(self, "inverter_parameters", inv_params)
        return self

    @model_validator(mode="after")
    def _p_module_to_module_parameters(self):
        """
        Set module_parameters['pdc0'] to the value of p_module.
        """
        p_module = self.p_module
        mod_params: dict[str, float] = self.module_parameters
        mod_params = dict(mod_params)  # avoid mutating shared defaults
        mod_params["pdc0"] = p_module
        object.__setattr__(self, "module_parameters", mod_params)
        return self

    @model_validator(mode="after")
    def _push_settings_to_arrays(self):
        """
        If settings are specified for the entire system, but not per array,
        and there are arrays defined, push the settings to each array.
        """
        if hasattr(self, "arrays") and self.arrays is not None:  # type: ignore
            for array in self.arrays:  # type: ignore
                if not array.module_parameters:
                    object.__setattr__(array, "module_parameters", self.module_parameters)
                if not array.module_type:
                    object.__setattr__(array, "module_type", self.module_type)
                if not array.mount.racking_model:
                    object.__setattr__(array.mount, "racking_model", self.racking_model)
        return self


class QuickScanModelChainModel(ModelChainModel):
    """
    ModelChain model for quick PV simulation setup using QuickPVSystemModel.

    Sets default AOI and DC models for PVWatts.
    """

    type: Literal["quickscan"] = Field("quickscan", frozen=True)  # tag
    system: QuickPVSystemModel
    aoi_model: str = "physical"
    dc_model: str = "pvwatts"
