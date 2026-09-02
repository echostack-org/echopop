"""
Pydantic and Pandera validators for spatial methods.

Covers coordinate manipulation, method parameterization, and related utilities.
"""

import re

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series
from pydantic import ConfigDict, Field, field_validator, model_validator

from ..core.validators import BaseDataFrame, BaseDictionary


class MeshDF(BaseDataFrame):
    """Pandera schema for validating the kriging mesh DataFrame."""

    longitude: Series[float] | None = pa.Field(ge=-180.0, le=180.0, nullable=False)
    latitude: Series[float] | None = pa.Field(ge=-90.0, le=90.0, nullable=False)
    x: Series[float] | None = pa.Field(nullable=False)
    y: Series[float] | None = pa.Field(nullable=False)
    area: Series[float] | None = pa.Field(nullable=False)
    fraction: Series[float] | None = pa.Field(nullable=False)

    class Config(BaseDataFrame.Config):
        """Pandera config for ``MeshDF``."""

        title = "kriging mesh DataFrame"

    @classmethod
    def pre_validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-validate and coerce the mesh DataFrame before pandera schema checks."""
        # Check for joint longitude-latitude
        lon_lat = {"longitude", "latitude"} <= set(df.columns)

        # Check for joint x-y
        x_y = {"x", "y"} <= set(df.columns)

        # Raise error if no complete pairs exist
        if not (lon_lat or x_y):
            raise ValueError(
                "Mesh DataFrame requires either paired ('longitude', 'latitude') or ('x', 'y') "
                "coordinate columns."
            )

        # Check for area/fraction
        if len({"area", "fraction"}.difference(set(df.columns))) == 2:
            raise ValueError(
                "Mesh DataFrame requires either an 'area' column (nmi^2), or a 'fraction' column "
                "representing the proportion of a reference area that applies to all grid cells."
            )

        return df


class TransectsDF(BaseDataFrame):
    """Pandera schema for validating along-transect survey results DataFrames."""

    longitude: Series[float] | None = pa.Field(ge=-180.0, le=180.0, nullable=False)
    latitude: Series[float] | None = pa.Field(ge=-90.0, le=90.0, nullable=False)
    x: Series[float] | None = pa.Field(nullable=False)
    y: Series[float] | None = pa.Field(nullable=False)

    class Config(BaseDataFrame.Config):
        """Pandera config for ``TransectsDF``."""

        title = "along-transect survey results DataFrame"

    @classmethod
    def pre_validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-validate and coerce the transects DataFrame before pandera schema checks."""
        # Check for joint longitude-latitude
        lon_lat = {"longitude", "latitude"} <= set(df.columns)

        # Check for joint x-y
        x_y = {"x", "y"} <= set(df.columns)

        # Raise error if no complete pairs exist
        if not (lon_lat or x_y):
            raise ValueError(
                "Transect DataFrame requires either paired ('longitude', 'latitude') or ('x', 'y') "
                "coordinate columns."
            )

        return df


class ValidateHullCropArgs(BaseDictionary):
    """Validation model for convex hull mesh-cropping arguments."""

    transects: pd.DataFrame
    mesh: pd.DataFrame
    num_nearest_transects: int = Field(gt=0)
    mesh_buffer_distance: float = Field(ge=0.0)
    projection: str
    coordinate_names: tuple[str, str]
    model_config = ConfigDict(title="hull convex method for kriging mesh cropping")

    @field_validator("mesh", mode="after")
    def validate_mesh(cls, v):
        """Validate the mesh DataFrame against the ``MeshDF`` pandera schema."""
        # Validate with pandera
        return MeshDF.validate(v)

    @field_validator("transects", mode="after")
    def validate_transects(cls, v):
        """Validate the transects DataFrame against the ``TransectsDF`` pandera schema."""
        # Validate with pandera
        return TransectsDF.validate(v)

    @field_validator("projection", mode="before")
    def validate_init(cls, v):
        """Coerce and validate the EPSG projection string."""
        # ---- Convert to a string if read in as an integer
        if isinstance(v, int | float):
            v = str(v)
        # ---- Convert to lowercase
        v = v.lower()
        # ---- Mold the entry into the expected format that includes a preceding 'epsg:'
        if not v.startswith("epsg"):
            v = "epsg:" + v
        # ---- Ensure that the colon is present
        if ":" not in v:
            v = "epsg:" + v.split("epsg")[1]
        # ---- Evaluate whether the pre-validator succeeded in finding an acceptable format
        if not re.match(r"^epsg:\d+$", v):
            raise ValueError(
                f"Echopop cannot parse the defined EPSG code ('{v}'). EPSG codes most be formatted "
                f"with strings beginning with 'epsg:' followed by the integer number code (e.g. "
                f"'epsg:4326')."
            )
        # ---- Return the pre-validated entry
        return v

    @model_validator(mode="after")
    def validate_coordinate_overlap(self):
        """Ensure mesh and transect DataFrames share the same coordinate column names."""
        # Get the mesh and transects DataFrames
        mesh = self.mesh
        transects = self.transects

        # Check for joint longitude-latitude
        if all([{"longitude", "latitude"} <= set(df.columns) for df in [mesh, transects]]):
            return self

        # Check for joint x-y
        if all([{"x", "y"} <= set(df.columns) for df in [mesh, transects]]):
            return self

        # Raise error if no shared complete pairs exist
        raise ValueError(
            "Coordinates for `transects` and `mesh` DataFrames must share the same coordinate "
            "names, which are expected to either be ('longitude', 'latitude') or ('x', 'y')."
        )
