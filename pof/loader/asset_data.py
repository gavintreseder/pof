import logging

import pandas as pd

from pof.units import scale_units


class SimpleFleet:
    def __init__(self, filename=None):

        self.df_age = None
        self.df_age_forecast = None
        self.df_forecast = None
        self.df_task_table = None
        self.filename = filename

    def load(self, filename=None):
        if filename is None:
            if self.filename is None:
                logging.warning("A filename is required")
            else:
                filename = self.filename

        df_age = pd.read_csv(filename)
        df_age = (
            df_age.loc[df_age["pole_material"] == "Timber"]
            .groupby(by=["age"])[["count"]]
            .count()
            .rename(columns={"count": "assets"})
        )
        df_age.index = df_age.index.astype(int)

        self.df_age = df_age

        logging.info("Fleet Data loaded")

        return self.df_age

    def calc_age_forecast(self, start_year, end_year, current_year=None):
        """ Get the simple population age that would have been expected between start and end without accounting for failures"""
        ## Quicker option

        # year_range = np.linspace(2015, 2019, 5, dtype=int)

        # df_forecast = pd.concat({year:df_age for year in year_range}, names=['year']).reset_index()
        # df_forecast['age']

        if current_year is None:
            current_year = start_year

        min_age = 0
        max_age = 100
        forecast_age = {}

        for year in range(start_year, end_year + 1):

            # Copy the df and adjust the age
            df_year = self.df_age.copy()

            #
            age_shift = year - current_year
            df_year.index = df_year.index + age_shift
            forecast_age[year] = df_year

        df_forecast = pd.concat(forecast_age, names=["year"]).reset_index()
        df_forecast = df_forecast.loc[
            (df_forecast["age"] > min_age) & (df_forecast["age"] < max_age)
        ]
        df_forecast = df_forecast.pivot(index="age", columns="year").fillna(0)

        df_forecast.columns = df_forecast.columns.droplevel()
        df_forecast = df_forecast.reset_index().melt(
            id_vars="age", value_name="assets"
        )[["age", "year", "assets"]]

        self.df_age_forecast = df_forecast

        return df_forecast

    def get_task_forecast(self, df_erc=None, units=None):
        """ Get the number of assets that would require each task by year """

        # Scale to ensure age merge
        df_age_forecast, __ = scale_units(
            self.df_age_forecast, "years", units
        )  

        # Merge with the estimated risk cost
        df = df_age_forecast.merge(df_erc, left_on="age", right_on="time")

        # Calculated population outcomes
        df["pop_quantity"] = df["assets"] * df["quantity"]
        df["pop_cost"] = df["pop_quantity"] * df["cost"]

        # Get Task forecast
        self.df_forecast_task = (
            df.groupby(by=["year", "task", "active"])[["pop_quantity", "pop_cost"]]
            .sum()
            .reset_index()
        )

        return self.df_forecast_task

    def get_cf_ff_forecast(self, df_erc=None):
        """ Get the number of assets that would require each task by year """

        # Merge with the estimated risk cost
        df = self.df_forecast.merge(df_erc, left_on="age", right_on="time")

        # Calculated population outcomes
        df["pop_quantity"] = df["assets"] * df["quantity"]
        df["pop_cost"] = df["pop_quantity"] * df["cost"]

        # Get Task forecast
        self.df_forecast_task = (
            df.groupby(by=["year", "task", "active", "system_impact"])[
                ["pop_quantity", "pop_cost"]
            ]
            .sum()
            .reset_index()
        )

        return self.df_forecast_task


# class AssetData():


#     def __init__(self):

#         NotImplemented


# class PoleData(AssetData):

#     def __init__(self):

#         self.pole_strength = 14

#         self.agd = 320
#         self.czd = 300
#         self.wt = 100