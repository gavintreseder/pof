import logging
from datetime import date

import pandas as pd
import scipy.stats as ss
import numpy as np
import dask.array as da
import dask.dataframe as dd


class FleetData:
    """
    A class that contains all the fleet data in the structured format that we want to use
    """

    def __init__(
        self, asset_info=None, condition=None, csq_info=None, pandas_or_dask="dask"
    ):

        self.asset_info = asset_info
        self.condition = condition
        self.csq_info = csq_info

        self.pandas_or_dask = pandas_or_dask

        if self.pandas_or_dask == "pandas":
            # Data Validation
            self._validate_data_pandas()

            # Data Preparation
            self._calculate_age_pandas()

            self.field_types = None
            self._field_types()

            # Data Summaries
            self.asset_summary = None
            self._summarise_asset_pandas()

            self.condition_summary = None
            self._summarise_condition_pandas()

            self.csq_summary = None
            self._summarise_csq_pandas()

        else:

            # Data Validation
            self._validate_data()

            # Data Preparation
            self._calculate_age()

            self.field_types = None
            self._field_types()

            # Data Summaries
            self.asset_summary = None
            self._summarise_asset()

            self.condition_summary = None
            self._summarise_condition()

            self.csq_summary = None
            self._summarise_csq()

        logging.debug("FleetData object created")

    # ************** Validation Functions *****************
    def _validate_data(self):
        """
        Checks the quality of data and logs a percentage match of asset ids across data sources.
        """

        logging.debug("Validating data")

        asset_info_assets = self.asset_info["asset_id"].drop_duplicates().reset_index()
        condition_assets = self.condition["asset_id"].drop_duplicates().reset_index()
        csq_info_assets = self.csq_info["asset_id"].drop_duplicates().reset_index()

        match_cond = asset_info_assets.merge(
            condition_assets, on="asset_id"
        ).index.size.compute()
        union_cond = asset_info_assets.merge(
            condition_assets, how="outer", on="asset_id"
        ).index.size.compute()

        asset_condition_match = match_cond / union_cond

        match_csq = asset_info_assets.merge(
            csq_info_assets, on="asset_id"
        ).index.size.compute()
        union_csq = asset_info_assets.merge(
            csq_info_assets, how="outer", on="asset_id"
        ).index.size.compute()

        asset_csq_match = match_csq / union_csq

        logging.info(
            f"{asset_condition_match:.2%} asset ids match in asset info and condition. {asset_csq_match:.2%} asset ids match in asset info and consequence."
        )
        logging.debug("Data validated")

    def _validate_data_pandas(self):
        """
        Checks the quality of data and logs a percentage match of asset ids across data sources.
        """

        logging.debug("Validating data")

        asset_info_assets = self.asset_info["asset_id"].drop_duplicates().reset_index()
        condition_assets = self.condition["asset_id"].drop_duplicates().reset_index()
        csq_info_assets = self.csq_info["asset_id"].drop_duplicates().reset_index()

        match_cond = asset_info_assets.merge(condition_assets, on="asset_id").index.size
        union_cond = asset_info_assets.merge(
            condition_assets, how="outer", on="asset_id"
        ).index.size

        asset_condition_match = match_cond / union_cond

        match_csq = asset_info_assets.merge(csq_info_assets, on="asset_id").index.size
        union_csq = asset_info_assets.merge(
            csq_info_assets, how="outer", on="asset_id"
        ).index.size

        asset_csq_match = match_csq / union_csq

        logging.info(
            f"{asset_condition_match:.2%} asset ids match in asset info and condition. {asset_csq_match:.2%} asset ids match in asset info and consequence."
        )
        logging.debug("Data validated")

    # ************** Preparation Functions *****************
    def _calculate_age(self):
        """
        Turns installation date into an age.
        """
        logging.debug("Converting date to age")

        today = date.today()
        self.asset_info["age"] = self.asset_info["installation_date"].apply(
            lambda x: today.year
            - x.year
            - ((today.month, today.day) < (x.month, x.day)),
            meta=("installation_date", "datetime64[ns]"),
        )

        logging.debug("Date converted")

    def _calculate_age_pandas(self):
        """
        Turns installation date into an age.
        """
        logging.debug("Converting date to age")

        today = date.today()

        self.asset_info["age"] = self.asset_info["installation_date"].apply(
            lambda x: today.year
            - x.year
            - ((today.month, today.day) < (x.month, x.day)),
        )
        logging.debug("Date converted")

    def _field_types(self, threshold=5):
        """
        Returns data types for columns in asset info, condition data and consequence model.
        Threshold is the number of unique int values that toggles between categorical and numerical.
        """

        logging.debug("Obtaining field types")

        types = self.asset_info.dtypes.to_dict()
        types.update(self.condition.dtypes.to_dict())
        types.update(self.csq_info.dtypes.to_dict())

        for col, item in types.items():

            if col == "age":
                types[col] = "numerical"
            elif item == "object":
                types[col] = "categorical"
            elif item == "int64":
                if col in self.asset_info.columns:
                    if "dollars" in col:
                        types[col] = "numerical"
                    elif len(self.asset_info[col].unique()) <= threshold:
                        types[col] = "categorical"
                    else:
                        types[col] = "numerical"
                elif col in self.condition.columns:
                    if "condition" in col:
                        types[col] = "numerical"
                    elif len(self.condition[col].unique()) <= threshold:
                        types[col] = "categorical"
                    else:
                        types[col] = "numerical"
                elif col in self.csq_info.columns:
                    if len(self.csq_info[col].unique()) <= threshold:
                        types[col] = "categorical"
                    else:
                        types[col] = "numerical"
            else:
                types[col] = "numerical"

        self.field_types = types

        logging.debug("Field types obtained")

    # ************** Summary Functions *****************
    def _summarise_asset(self):
        """
        Summary of asset data.
        """

        logging.debug("Summarising asset data")
        self.asset_summary = self.asset_info.compute()
        logging.debug("Asset data summarised")

    def _summarise_asset_pandas(self):
        """
        Summary of asset data.
        """

        logging.debug("Summarising asset data")
        self.asset_summary = self.asset_info
        logging.debug("Asset data summarised")

    def _summarise_condition(self):
        """
        Summarises the condition data into format describing perfect_condition condition, current_condition condition,
        and whether an asset has experienced condition loss.
        """
        logging.debug("Summarising condition")

        condition_summary = self.condition[["asset_id", "condition_name"]]

        logging.debug("Summarising condition: obtaining perfect condition")

        perfect_condition = (
            self.condition.groupby(["asset_id", "condition_name"])["condition_after"]
            .max()
            .reset_index()
            .rename(columns={"condition_after": "perfect_condition"})
        )
        condition_summary = condition_summary.merge(
            perfect_condition, on=["asset_id", "condition_name"]
        )

        logging.debug("Summarising condition: obtaining current condition")

        idx_recent_date = (
            self.condition.groupby(["asset_id", "condition_name"])["date"].transform(
                max,
                meta=("date", "datetime64[ns]"),
            )
            == self.condition["date"]
        )

        current_condition = self.condition[idx_recent_date][
            ["asset_id", "condition_name", "condition_after"]
        ].rename(columns={"condition_after": "current_condition"})

        condition_summary = condition_summary.merge(
            current_condition, on=["asset_id", "condition_name"]
        )

        logging.debug("Summarising condition: calculating condition loss")

        condition_summary["condition_loss"] = da.where(
            condition_summary["perfect_condition"].to_dask_array()
            == condition_summary["current_condition"].to_dask_array(),
            False,
            True,
        )

        logging.debug("Summarising condition: separating data per condition name")

        condition_summary = condition_summary.drop_duplicates().reset_index(drop=True)

        condition_name_list = self.condition["condition_name"].unique().compute()

        self.condition_summary = condition_summary[
            ["asset_id", "condition_name"]
        ].rename(
            columns={
                "condition_name": "dummy",
            }
        )

        self.condition_summary["dummy"] = 0
        self.condition_summary = self.condition_summary.drop_duplicates().reset_index(
            drop=True
        )

        for condition_name in condition_name_list:

            self.condition_summary = (
                self.condition_summary.merge(
                    condition_summary[
                        condition_summary["condition_name"] == condition_name
                    ],
                    how="outer",
                    on="asset_id",
                )
                .rename(
                    columns={
                        "perfect_condition": condition_name + "_perfect_condition",
                        "current_condition": condition_name + "_current_condition",
                        "condition_loss": condition_name + "_condition_loss",
                    }
                )
                .drop(columns={"condition_name"})
            )

        self.condition_summary = self.condition_summary.drop(columns={"dummy"})

        self.condition_summary = self.condition_summary.compute()

        logging.debug("Condition summarised")

    def _summarise_condition_pandas(self):
        """
        Summarises the condition data into format describing perfect_condition condition, current_condition condition,
        and whether an asset has experienced condition loss.
        """
        logging.debug("Summarising condition")

        condition_summary = self.condition[["asset_id", "condition_name"]]

        logging.debug("Summarising condition: obtaining perfect condition")

        perfect_condition = (
            self.condition.groupby(["asset_id", "condition_name"])["condition_after"]
            .max()
            .reset_index()
            .rename(columns={"condition_after": "perfect_condition"})
        )
        condition_summary = condition_summary.merge(
            perfect_condition, on=["asset_id", "condition_name"]
        )

        logging.debug("Summarising condition: obtaining current condition")

        idx_recent_date = (
            self.condition.groupby(["asset_id", "condition_name"])["date"].transform(
                max,
            )
            == self.condition["date"]
        )

        current_condition = self.condition[idx_recent_date][
            ["asset_id", "condition_name", "condition_after"]
        ].rename(columns={"condition_after": "current_condition"})

        condition_summary = condition_summary.merge(
            current_condition, on=["asset_id", "condition_name"]
        )

        logging.debug("Summarising condition: calculating condition loss")

        condition_summary["condition_loss"] = True

        condition_summary["condition_loss"] = condition_summary["condition_loss"].where(
            condition_summary["perfect_condition"]
            == condition_summary["current_condition"],
            False,
        )

        logging.debug("Summarising condition: separating data per condition name")

        condition_summary = condition_summary.drop_duplicates().reset_index(drop=True)

        condition_name_list = self.condition["condition_name"].unique()

        self.condition_summary = condition_summary[
            ["asset_id", "condition_name"]
        ].rename(
            columns={
                "condition_name": "dummy",
            }
        )

        self.condition_summary["dummy"] = 0
        self.condition_summary = self.condition_summary.drop_duplicates().reset_index(
            drop=True
        )

        for condition_name in condition_name_list:

            self.condition_summary = (
                self.condition_summary.merge(
                    condition_summary[
                        condition_summary["condition_name"] == condition_name
                    ],
                    how="outer",
                    on="asset_id",
                )
                .rename(
                    columns={
                        "perfect_condition": condition_name + "_perfect_condition",
                        "current_condition": condition_name + "_current_condition",
                        "condition_loss": condition_name + "_condition_loss",
                    }
                )
                .drop(columns={"condition_name"})
            )

        self.condition_summary = self.condition_summary.drop(columns={"dummy"})

        logging.debug("Condition summarised")

    def _summarise_csq(self):
        """
        Summary of consequence data. Not currently used.
        """

        logging.debug("Summarising consequence model data")
        self.csq_summary = self.csq_info.compute()
        logging.debug("Consequence model data summarised")

    def _summarise_csq_pandas(self):
        """
        Summary of consequence data. Not currently used.
        """

        logging.debug("Summarising consequence model data")
        self.csq_summary = self.csq_info
        logging.debug("Consequence model data summarised")

    def get_population_data(self, by, remove, n_bins=None, keep_original=True):
        """
        Returns a summary of asset data defined by user.
        """

        logging.debug("Filtering asset data")
        asset_info_filtered = self._filter_asset_info(by, remove)

        logging.debug("Filtering condition data")
        condition_filtered = self._filter_condition(by)

        logging.debug("Filtering consequence data: NOTIMPLEMENTED")
        # TODO: Add csq_filtered

        logging.debug("Merging filtered data")
        df_summary = asset_info_filtered.merge(condition_filtered, on="asset_id")
        # TODO: df_summary = df_summary.merge(self.csq_filtered, on="asset_id")

        if n_bins:
            if keep_original:
                logging.debug("Obtaining bins for numerical data, n_bins = %s", n_bins)
                df_summary, bin_list = self._get_bins(
                    df=df_summary,
                    by=by,
                    n_bins=n_bins,
                    keep_original=keep_original,
                )
                for att in bin_list:
                    by[att] = []
            else:
                logging.debug("Obtaining bins for numerical data, n_bins = %s", n_bins)
                df_summary, bin_list = self._get_bins(
                    df=df_summary,
                    by=by,
                    n_bins=n_bins,
                    keep_original=keep_original,
                )

        attributes_to_keep = list(by.keys())
        df_summary = df_summary[["asset_id"] + attributes_to_keep]

        logging.debug("Population data complete")

        return df_summary

    def get_population_summary(self, by, remove, n_bins=None, keep_original=True):
        """
        Returns a population summary of asset data defined by user.
        """

        df_summary = self.get_population_data(
            by=by,
            remove=remove,
            n_bins=n_bins,
            keep_original=keep_original,
        )

        logging.debug("Summarising population data")
        groupby_filter = list(by.keys())
        df_sum = (
            df_summary.groupby(groupby_filter, dropna=False)["asset_id"]
            .count()
            .reset_index()
            .rename(columns={"asset_id": "count"})
        )

        df_sum = df_sum[df_sum["count"] > 0].reset_index(drop=True)

        logging.debug("Population summary complete")

        return df_sum

    def _filter_asset_info(self, by, remove):
        """
        Returns asset data of assets with traits decribed by user filters.
        """
        conditions = []
        for att, trait in by.items():
            if att in self.field_types:
                if trait:
                    conditions.append((self.asset_summary[att].isin(trait)))

        if remove:
            for att, trait in remove.items():
                conditions.append((~self.asset_summary[att].isin(trait)))

        if conditions:
            cond = conditions[0]

            for c in conditions[1:]:
                cond = cond & c

            asset_info_filtered = self.asset_summary[cond]
        else:
            asset_info_filtered = self.asset_summary

        return asset_info_filtered

    def _filter_condition(self, by):
        """
        Returns condition data of assets with traits decribed by user filters.
        """
        conditions = []
        for att, trait in by.items():
            for condition_type in [
                "condition_loss",
                "perfect_condition",
                "current_condition",
            ]:
                if condition_type in att:
                    if condition_type == "condition_loss" and trait:
                        conditions.append((self.condition_summary[att].isin(trait)))
        if conditions:
            cond = conditions[0]

            for c in conditions[1:]:
                cond = cond & c

            condition_filtered = self.condition_summary[cond]
        else:
            condition_filtered = self.condition_summary

        return condition_filtered

    def _filter_consequence_model(self, by, remove):
        """
        NOTIMPLEMENTED: Returns consequence data of assets with traits decribed by user filters.
        """

        conditions = []
        for att, trait in by.items():
            if att in self.field_types:
                if trait:
                    conditions.append((self.csq_summary[att].isin(trait)))

        if remove:
            for att, trait in remove.items():
                conditions.append((~self.csq_summary[att].isin(trait)))

        if conditions:
            cond = conditions[0]

            for c in conditions[1:]:
                cond = cond & c

            consequence_filtered = self.csq_summary[cond]
        else:
            consequence_filtered = self.csq_summary

        return NotImplemented

    def _get_bins(self, df, by, n_bins, keep_original, suffix="_bin"):
        """
        Turns chosen numerical data into binned data.
        """
        bin_list = []
        for att, trait in by.items():
            if att in self.field_types:
                if self.field_types[att] == "numerical":
                    logging.debug("Obtaining bins for %s", att)
                    bins = np.round(
                        np.linspace(
                            np.floor(df[att].min()) - 0.01,
                            np.ceil(df[att].max()) + 0.01,
                            n_bins,
                        )
                    )
                    labels = bins[1:]
                    if keep_original:
                        bin_list.append(att + suffix)
                        df[att + suffix] = pd.cut(
                            df[att], bins=bins, labels=labels, duplicates="drop"
                        )
                        df[att + suffix] = df[att + suffix].astype(float)
                    else:
                        df[att] = pd.cut(
                            df[att], bins=bins, labels=labels, duplicates="drop"
                        )
                        df[att] = df[att].astype(float)

        return df, bin_list

    # ************** Generate Data Functions *****************
    @classmethod
    def _gen_fleet_data(cls, n_assets=100):
        """
        Generation of dummy fleet data used to develop summary methods.
        """

        att_1 = ["type_1", "type_2"]
        att_2 = ["var_1", "var_2", "var_3"]
        att_3 = [5, 10, 15, 20]
        installation_date = [
            "2019-01-19",
            "2018-04-03",
            "2017-08-24",
            "2016-08-21",
            "2015-07-26",
            "2014-08-27",
            "2013-04-09",
            "2012-08-15",
            "2011-01-22",
            "2010-02-02",
        ]
        code = ["code_1", "code_2"]
        dates = ["2020-10-10", "2020-05-10", "2020-01-10", "2019-10-10"]
        condition_before = [90, 100, 100]
        condition_after = [50, 90, 100]

        asset_info = pd.DataFrame()

        asset_info["asset_id"] = np.array(
            ["asset_" + str(xi) for xi in np.arange(0, n_assets, 1)]
        )
        asset_info["att_1"] = (
            att_1 * int(n_assets / len(att_1)) + att_1[: n_assets % len(att_1)]
        )
        asset_info["att_2"] = (
            att_2 * int(n_assets / len(att_2)) + att_2[: n_assets % len(att_2)]
        )
        asset_info["att_3"] = (
            att_3 * int(n_assets / len(att_3)) + att_3[: n_assets % len(att_3)]
        )

        asset_info["installation_date"] = np.random.choice(
            list(installation_date), size=n_assets
        )
        asset_info["installation_date"] = pd.to_datetime(
            asset_info["installation_date"], format="%Y-%m-%d", errors="coerce"
        )

        asset_info["numerical"] = ss.norm.rvs(loc=50, scale=10, size=n_assets)

        condition = pd.DataFrame()
        condition["asset_id"] = list(asset_info["asset_id"]) * 2
        condition["condition_name"] = (
            code * int(2 * n_assets / len(code)) + code[: 2 * n_assets % len(code)]
        )
        condition["date"] = np.random.choice(list(dates), size=2 * n_assets)
        condition["date"] = pd.to_datetime(
            condition["date"], format="%Y-%m-%d", errors="coerce"
        )
        condition["condition_before"] = np.random.choice(
            list(condition_before), size=2 * n_assets
        )
        condition["condition_after"] = np.random.choice(
            list(condition_after), size=2 * n_assets
        )

        csq_info = pd.DataFrame()
        csq_info["asset_id"] = asset_info["asset_id"]
        csq_info["total_csq"] = np.random.randint(0, 10000, size=n_assets)

        return cls(
            asset_info=dd.from_pandas(asset_info, npartitions=1).reset_index(),
            condition=dd.from_pandas(condition, npartitions=1).reset_index(),
            csq_info=dd.from_pandas(csq_info, npartitions=1).reset_index(),
        )

    # ****************************************************************

    def get_asset(self, key=None):
        """
        Takes a key and returns as AssetData class for that asset
        """

        asset = "asset that matches that key"

        NotImplemented

        return asset

    def get_representative_asset(self):

        """
        Return the 'average asset'
        """

    def get_youngest_asset(self):

        NotImplemented


if __name__ == "__main__":
    print("ok")
