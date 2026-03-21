"""
Construct the poststratification frame from census data.
"""

from __future__ import annotations

import pandas as pd

from mobmrp.config import MRPConfig


def build_poststrat_frame(
    geo_proportions: pd.DataFrame,
    gender_proportions: pd.DataFrame,
    config: MRPConfig,
) -> pd.DataFrame:
    """Build the joint poststratification frame under conditional independence.

    .. math::

        N(\\text{socioeco}, \\text{gender}, \\text{area})
        = N_{\\text{total}}(\\text{area})
          \\times P(\\text{socioeco} \\mid \\text{area})
          \\times P(\\text{gender} \\mid \\text{parent area})

    Parameters
    ----------
    geo_proportions : DataFrame
        One row per *fine_geographic_unit* with columns:

        * ``fine_geographic_unit`` identifier
        * ``geographic_unit`` identifier (parent unit)
        * ``"total_pop"`` -- total population (households or persons)
        * one column per socioeconomic level holding the **proportion**
          of that level in the area

    gender_proportions : DataFrame
        One row per *geographic_unit* with columns:

        * ``geographic_unit`` identifier
        * one column per gender level holding the **proportion** of that
          gender in the unit

    config : MRPConfig

    Returns
    -------
    DataFrame
        One row per ``(fine_geo_unit, socioeco, gender)`` with a
        ``census_pop`` column.
    """
    cols = config.columns
    fine_geo = cols.fine_geographic_unit
    geo_unit = cols.geographic_unit

    rows: list[dict] = []
    for _, area in geo_proportions.iterrows():
        parent = str(area[geo_unit])

        gender_row = gender_proportions[
            gender_proportions[geo_unit].astype(str) == parent
        ]
        if gender_row.empty:
            continue
        gender_row = gender_row.iloc[0]

        total_pop = float(area["total_pop"])

        for socioeco in config.socioeco_levels:
            p_socioeco = float(area[socioeco])
            for gender in config.gender_levels:
                p_gender = float(gender_row[gender])
                pop = total_pop * p_socioeco * p_gender
                if pop > 0:
                    rows.append(
                        {
                            fine_geo: area[fine_geo],
                            geo_unit: parent,
                            cols.socioeconomic_group: socioeco,
                            cols.gender: gender,
                            cols.census_pop: pop,
                        }
                    )

    frame = pd.DataFrame(rows)

    if config.socioeco_levels:
        cat_dtype = pd.CategoricalDtype(
            categories=config.socioeco_levels, ordered=True
        )
        frame[cols.socioeconomic_group] = frame[cols.socioeconomic_group].astype(
            cat_dtype
        )

    return frame


def build_poststrat_frame_from_microdata(
    census_microdata: pd.DataFrame,
    geo_proportions: pd.DataFrame,
    config: MRPConfig,
    census_gender_col: str = "sex",
    census_gender_map: dict | None = None,
    census_geo_col: str | None = None,
) -> pd.DataFrame:
    """Build the poststratification frame from person-level census data.

    This is a convenience wrapper around :func:`build_poststrat_frame`.  It
    first computes gender proportions per *geographic_unit* from the census
    microdata, then delegates to the main function.

    Parameters
    ----------
    census_microdata : DataFrame
        Person-level census records.  Must contain a gender column and a
        geographic-unit column.
    geo_proportions : DataFrame
        Same as in :func:`build_poststrat_frame`.
    config : MRPConfig
    census_gender_col : str
        Column in *census_microdata* with gender codes.
    census_gender_map : dict, optional
        Mapping from raw census gender codes to ``config.gender_levels``
        (e.g. ``{1: "M", 2: "F"}``).
    census_geo_col : str, optional
        Geographic-unit column in *census_microdata*.  Defaults to
        ``config.columns.geographic_unit``.

    Returns
    -------
    DataFrame
        Output of :func:`build_poststrat_frame`.
    """
    cols = config.columns
    geo_col = census_geo_col or cols.geographic_unit

    micro = census_microdata.copy()
    if census_gender_map:
        micro[census_gender_col] = micro[census_gender_col].map(census_gender_map)

    # Compute gender proportions per geographic unit
    counts = micro.groupby([geo_col, census_gender_col]).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)
    gender_props = counts.div(totals, axis=0).reset_index()
    gender_props = gender_props.rename(columns={geo_col: cols.geographic_unit})

    # Ensure gender columns match config.gender_levels
    for gl in config.gender_levels:
        if gl not in gender_props.columns:
            raise ValueError(
                f"Gender level '{gl}' not found after mapping.  "
                f"Available: {list(gender_props.columns)}"
            )

    return build_poststrat_frame(geo_proportions, gender_props, config)
