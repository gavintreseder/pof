import pandas as pd
import scipy.stats as ss
import numpy as np
import datetime
from datetime import date
import math
import logging


class FleetData:

    """
    A class that contains all the fleet data in the structured format that we want to use
    """

    # Primary Key -> Asset ID
    # Asset IDs

    def __init__(self, asset_info=None, condition=None, csq_info=None):

        self.asset_info = asset_info
        self.condition = condition
        self.csq_info = csq_info

        # Humanise Names

        # Data Validation
        self._validate_data()

        # Data Preparation
        self._calculate_age()

        self.field_types = None
        self._field_types()

        # Data Summaries
        self.condition_summary = None
        self._summarise_condition()

        self.csq_summary = None
        self._summarise_csq()

    # ************** Validation Functions *****************
    def _validate_data(self):
        """
        Checks the quality of data and logs a percentage match of asset ids across data sources.
        """

        asset_info_assets = self.asset_info["asset_id"].unique()
        condition_assets = self.condition["asset_id"].unique()
        csq_info_assets = self.csq_info["asset_id"].unique()

        asset_condition_match = (
            len(set(asset_info_assets) & set(condition_assets))
            / len(set(asset_info_assets) | set(condition_assets))
        ) * 100
        asset_csq_match = (
            len(set(asset_info_assets) & set(csq_info_assets))
            / len(set(asset_info_assets) | set(csq_info_assets))
        ) * 100

        logging.info(
            "%s percent asset ids match in asset info and condition. %s percent asset ids match in asset info and consequence.",
            asset_condition_match,
            asset_csq_match,
        )

    # ************** Preparation Functions *****************
    def _calculate_age(self, string="installation_date"):
        """
        Turns installation date into an age.
        """

        today = date.today()
        self.asset_info["age"] = self.asset_info[string].apply(
            lambda x: today.year
            - x.year
            - ((today.month, today.day) < (x.month, x.day))
        )

    def _field_types(self, threshold=5):
        """
        Returns data types for columns in asset info. Threshold is the number of unique int values
        that toggles between categorical and numerical.
        """

        types = self.asset_info.dtypes.to_dict()

        for key, item in types.items():
            if key == "age":
                types[key] = "numerical"
            elif item == "object":
                types[key] = "categorical"
            elif item == "int64":
                if len(self.asset_info[key].unique()) <= threshold:
                    types[key] = "categorical"
                else:
                    types[key] = "numerical"
            else:
                types[key] = "numerical"

        self.field_types = types

    # ************** Summary Functions *****************
    def _summarise_condition(self, key="asset_id"):
        """
        Summarises the condition data into format describing perfect condition, current condition,
        and whether an asset has experienced condition loss.
        """

        condition_summary = pd.DataFrame()
        condition_summary[["asset_id", "detail_code"]] = self.condition[
            ["asset_id", "detail_code"]
        ]

        perfect = (
            self.condition.iloc[
                self.condition.groupby(["asset_id", "detail_code"])["before_value"].agg(
                    pd.Series.idxmax
                )
            ]
            .set_index("asset_id")[["detail_code", "before_value"]]
            .reset_index()
            .rename(columns={"before_value": "perfect"})
        )
        condition_summary = condition_summary.merge(
            perfect, on=["asset_id", "detail_code"]
        )

        condition_summary["current"] = (
            self.condition.sort_values("inspection_date")
            .groupby(["asset_id", "detail_code"])
            .last()["after_value"]
            .reset_index(drop=True)
        )

        # In cases of current = renewal with no previous records of this condition
        condition_summary["perfect"] = np.where(
            condition_summary["perfect"] < condition_summary["current"],
            condition_summary["current"],
            condition_summary["perfect"],
        )

        condition_summary["condition_loss"] = np.where(
            condition_summary["perfect"] == condition_summary["current"],
            False,
            True,
        )

        condition_summary = condition_summary.pivot(
            index="asset_id",
            columns="detail_code",
            values=["perfect", "current", "condition_loss"],
        )
        condition_summary.columns = condition_summary.columns.swaplevel(0, 1)
        condition_summary.sort_index(axis=1, level=0, inplace=True)

        self.condition_summary = condition_summary.reset_index()

    def _summarise_csq(self, key="asset_id"):
        """
        Summary of consequence data.
        """
        # TODO

        self.csq_summary = NotImplemented

    def get_population_summary(self, by, filter, n_increments=10, condition_loss=None):

        idx_a = self._return_index_asset_info(by, filter)
        idx_c, code_list = self._return_index_condition(by)

        condition_summary = self.condition_summary.loc[idx_c][
            ["asset_id"] + code_list
        ].drop_duplicates()
        condition_summary.columns = condition_summary.columns.map("_".join).str.strip(
            "_"
        )

        # TODO: csq summary

        df_summary = self.asset_info.loc[idx_a].merge(condition_summary, on="asset_id")

        df_summary = self._get_bins(df_summary, by, n_increments)

        groupby_filter = list(by.keys())

        df_summary = (
            df_summary.groupby(groupby_filter)["asset_id"]
            .count()
            .reset_index()
            .rename(columns={"asset_id": "count"})
        )

        df_summary = df_summary[
            df_summary["count"] != 0
        ]  # Delete if you want to keep zero counts
        if condition_loss is not None:
            pass  # TODO add in overall condition_loss filter

        return df_summary

    def _return_index_asset_info(self, by, filter):
        """
        Returns index if assets with traits decribed by user filters.
        """

        idx = []
        for att, trait in by.items():
            if trait:
                idx_list = np.where((self.asset_info[att].isin(trait)))
                if not idx:
                    idx = set(idx_list[0])
                else:
                    idx = set(idx) & set(idx_list[0])

        for att, trait in filter.items():
            idx_list = np.where((~self.asset_info[att].isin(trait)))
            if not idx:
                idx = set(idx_list[0])
            else:
                idx = set(idx) & set(idx_list[0])
        idx = list(idx)

        return idx

    def _return_index_condition(self, by):

        idx = []
        code_list = []
        for att, trait in by.items():
            for condition_type in ["condition_loss", "perfect", "current"]:
                if condition_type in att:
                    code = att.replace("_" + condition_type, "")
                    code_list.append(code)
                    if condition_type == "condition_loss" and trait:
                        idx_list_1 = np.where(
                            (pd.notnull(self.condition_summary[code]))
                        )[0]
                        idx_list_2 = np.where(
                            (self.condition_summary[code]["condition_loss"] == trait)
                        )[0]
                        idx_list = set(idx_list_1) & set(idx_list_2)
                    else:
                        idx_list = np.where((pd.notnull(self.condition_summary[code])))[
                            0
                        ]
                    if not idx:
                        idx = set(idx_list)
                    else:
                        idx = set(idx) & set(idx_list)
        idx = list(idx)
        code_list = list(set(code_list))

        if code_list == []:
            idx = self.condition_summary.index

        return idx, code_list

    def _get_bins(self, df, by, n_increments):
        """
        Turns chosen numerical data into binned data.
        """
        for att, trait in by.items():
            if att in self.field_types:
                if self.field_types[att] == "numerical":
                    bins = np.unique(
                        np.round(
                            np.linspace(
                                math.floor(df[att].min() - 0.01),
                                math.ceil(df[att].max()),
                                n_increments + 1,
                            )
                        )
                    )
                    labels = bins[1:]
                    df[att] = pd.cut(df[att], bins=bins, labels=labels)

        return df

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
        code = ["code_1", "code_2", "code_3"]
        inspection_date = ["2020-10-10", "2020-05-10", "2020-01-10", "2019-10-10"]
        before_value = [90, 100, 100]
        after_value = [50, 90, 100]

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
        condition["detail_code"] = (
            code * int(2 * n_assets / len(code)) + code[: 2 * n_assets % len(code)]
        )
        condition["inspection_date"] = np.random.choice(
            list(inspection_date), size=2 * n_assets
        )
        condition["before_value"] = np.random.choice(
            list(before_value), size=2 * n_assets
        )
        condition["after_value"] = np.random.choice(
            list(after_value), size=2 * n_assets
        )

        csq_info = pd.DataFrame()
        csq_info["asset_id"] = asset_info["asset_id"]
        csq_info["csq"] = np.random.randint(0, 10, size=n_assets)

        return cls(asset_info=asset_info, condition=condition, csq_info=csq_info)

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
