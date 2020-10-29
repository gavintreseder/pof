import pandas as pd
import scipy.stats as ss
import numpy as np
import datetime
from datetime import date
import math
import logging
import dask.array as da
import dask.dataframe as dd
import dask.distributed
from distributed import Client

client = Client(processes=False)


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

        asset_info_assets = self.asset_info["asset_id"].unique().reset_index()
        condition_assets = self.condition["asset_id"].unique().reset_index()
        csq_info_assets = self.csq_info["asset_id"].unique().reset_index()

        match_cond = asset_info_assets.merge(condition_assets, on="asset_id").index.size
        union_cond = asset_info_assets.merge(
            condition_assets, how="outer", on="asset_id"
        ).index.size

        asset_condition_match = (match_cond / union_cond) * 100

        match_csq = asset_info_assets.merge(csq_info_assets, on="asset_id").index.size
        union_csq = asset_info_assets.merge(
            csq_info_assets, how="outer", on="asset_id"
        ).index.size

        asset_csq_match = (match_csq / union_csq) * 100

        logging.info(
            "%s percent asset ids match in asset info and condition. %s percent asset ids match in asset info and consequence.",
            asset_condition_match,
            asset_csq_match,
        )

    # ************** Preparation Functions *****************
    def _calculate_age(self):
        """
        Turns installation date into an age.
        """

        today = date.today()
        self.asset_info["age"] = self.asset_info["installation_date"].apply(
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

    # ************** Summary Functions *****************
    def _summarise_condition(self):
        """
        Summarises the condition data into format describing perfect_condition condition, current_condition condition,
        and whether an asset has experienced condition loss.
        """

        condition_summary = self.condition[["asset_id", "condition_name"]]

        self.condition["condition_after"] = self.condition["condition_after"].fillna(0)

        # perfect_condition = (
        #     self.condition.iloc[
        #         self.condition.groupby(["asset_id", "condition_name"])[
        #             "condition_after"
        #         ].agg(dd.Series.idxmax)
        #     ]
        #     .set_index("asset_id")[["condition_name", "condition_after"]]
        #     .reset_index()
        #     .rename(columns={"condition_after": "perfect_condition"})
        # )
        perfect_condition = (
            self.condition.groupby(["asset_id", "condition_name"])["condition_after"]
            .max()
            .reset_index()
            .rename(columns={"condition_after": "perfect_condition"})
        )

        condition_summary = condition_summary.merge(
            perfect_condition, on=["asset_id", "condition_name"]
        )

        # current_condition = (
        #     self.condition.sort_values("date")
        #     .groupby(["asset_id", "condition_name"])
        #     .last()["condition_after"]
        #     .reset_index()
        #     .rename(columns={"condition_after": "current_condition"})
        # )

        idx_recent_date = (
            self.condition.groupby(["asset_id", "condition_name"])["date"].transform(
                max
            )
            == self.condition["date"]
        )

        current_condition = self.condition[idx_recent_date][
            ["asset_id", "condition_name", "condition_after"]
        ].rename(columns={"condition_after": "current_condition"})

        condition_summary = condition_summary.merge(
            current_condition, on=["asset_id", "condition_name"]
        )

        condition_summary["condition_loss"] = da.where(
            condition_summary["perfect_condition"].to_dask_array()
            == condition_summary["current_condition"].to_dask_array(),
            False,
            True,
        )

        condition_summary = condition_summary.drop_duplicates().reset_index(drop=True)

        condition_name_list = client.persist(self.condition["condition_name"].unique())

        condition_summary_2 = condition_summary[["asset_id", "condition_name"]].rename(
            columns={
                "condition_name": "dummy",
            }
        )

        condition_summary_2["dummy"] = 0
        condition_summary_2 = condition_summary_2.drop_duplicates().reset_index(
            drop=True
        )

        for condition_name in condition_name_list:

            condition_summary_2 = (
                condition_summary_2.merge(
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

        condition_summary_2 = condition_summary_2.drop(columns={"dummy"})
        self.condition_summary = condition_summary_2

    def _summarise_csq(self):
        """
        Summary of consequence data. Not currently used.
        """

        self.csq_summary = self.csq_info

    def get_population_summary(self, by, remove, n_bins=10):
        """
        Returns a summary of asset data defined by user.
        """

        # filter categorical
        asset_info_filtered = self._filter_asset_info(by, remove)
        condition_filtered = self._filter_condition(by)
        # TODO: Add csq_filtered

        # merge asset, condition, csq
        df_summary = asset_info_filtered.merge(condition_filtered, on="asset_id")
        # TODO: df_summary = df_summary.merge(self.csq_filtered, on="asset_id")

        # bin numerical
        # df_summary = self._get_bins(df_summary, by, n_bins)

        # compute dataframe
        df_summary = client.persist(df_summary)

        # filter numerical
        groupby_filter = list(by.keys())

        # group and count by filter
        df_sum = (
            df_summary.groupby(groupby_filter, dropna=False)["asset_id"]
            .count()
            .reset_index()
            .rename(columns={"asset_id": "count"})
        )

        # drop zero counts
        df_summary = df_sum[df_sum["count"] > 0].reset_index(drop=True)

        return df_summary

    def get_population_summary_legacy(self, by, remove, n_bins=10):
        """
        Returns a summary of asset data defined by user.
        """

        # filter categorical
        idx_a = self._return_index_asset_info(by, remove)
        print(idx_a)
        idx_c, code_list = self._return_index_condition(by)
        print(idx_c)

        condition_summary = self.condition_summary.loc[idx_c].drop_duplicates()
        # TODO add in condition code filter

        # merge asset, condition, csq
        df_summary = self.asset_info.loc[idx_a].merge(condition_summary, on="asset_id")
        # df_summary = df_summary.merge(self.csq_summary, on="asset_id")

        # bin numerical
        df_summary = self._get_bins(df_summary, by, n_bins)

        # filter numerical
        groupby_filter = list(by.keys())

        # group and count by filter
        df_summary = (
            df_summary.groupby(groupby_filter, dropna=False)["asset_id"]
            .count()
            .reset_index()
            .rename(columns={"asset_id": "count"})
        )

        # drop zero counts
        df_summary = df_summary[
            df_summary["count"] != 0
        ]  # Delete if you want to keep zero counts

        return df_summary

    def _filter_asset_info(self, by, remove):
        """
        Returns index if assets with traits decribed by user filters.
        """
        conditions = []
        for att, trait in by.items():
            if att in self.field_types:
                if trait:
                    conditions.append((self.asset_info[att].isin(trait)))

        if remove:
            for att, trait in remove.items():
                conditions.append((~self.asset_info[att].isin(trait)))

        if conditions:
            cond = conditions[0]

            for c in conditions[1:]:
                cond = cond & c

            asset_info_filtered = self.asset_info[cond]
        else:
            asset_info_filtered = self.asset_info

        return asset_info_filtered

    def _filter_condition(self, by):

        conditions = []
        for att, trait in by.items():
            for condition_type in [
                "condition_loss",
                "perfect_condition",
                "current_condition",
            ]:
                if condition_type in att:
                    code = att.replace("_" + condition_type, "")
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

    def _return_index_asset_info(self, by, remove):
        """
        Returns index if assets with traits decribed by user filters.
        """

        idx = []
        for att, trait in by.items():

            if att in self.field_types:

                if trait:

                    idx_list = da.where((self.asset_info[att].isin(trait)))
                    print(idx_list)
                    if not idx:
                        idx = set(idx_list[0])
                    else:
                        idx = set(idx) & set(idx_list[0])

        if remove:
            for att, trait in remove.items():

                idx_list = da.where((~self.asset_info[att].isin(trait).to_dask_array()))
                if not idx:
                    idx = set(idx_list[0])
                else:
                    idx = set(idx) & set(idx_list[0])

        if idx == []:
            idx = self.asset_info.index

        idx = list(idx)

        return idx

    def _return_index_condition(self, by):
        # TODO return to get

        idx = []
        code_list = []
        for att, trait in by.items():
            for condition_type in [
                "condition_loss",
                "perfect_condition",
                "current_condition",
            ]:
                if condition_type in att:
                    code = att.replace("_" + condition_type, "")
                    code_list.append(code)
                    if condition_type == "condition_loss" and trait:
                        idx_list = np.where(
                            (self.condition_summary[code]["condition_loss"] == trait[0])
                        )[0]
                        if not idx:
                            idx = set(idx_list)
                        else:
                            idx = set(idx) & set(idx_list)
        idx = list(idx)
        code_list = list(set(code_list))

        if code_list == []:
            code_list = set(self.condition_summary.columns.get_level_values(0))
            code_list.remove("asset_id")
            code_list = list(idx)
        if idx == []:
            idx = list(self.condition_summary.index)

        return idx, code_list

    def _get_bins(self, df, by, n_bins):
        """
        Turns chosen numerical data into binned data.
        """
        for att, trait in by.items():
            if att in self.field_types:
                if self.field_types[att] == "numerical":
                    bins = da.unique(
                        da.round(
                            da.linspace(
                                da.floor(df[att].min() - 0.01),
                                da.ceil(df[att].max() + 0.01),
                                n_bins,
                            )
                        )
                    ).compute()
                    labels = bins[1:]
                    df[att] = df[att].map_partitions(pd.cut, bins, labels=labels)

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
        code = ["code_1", "code_2"]
        date = ["2020-10-10", "2020-05-10", "2020-01-10", "2019-10-10"]
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
        condition["date"] = np.random.choice(list(date), size=2 * n_assets)
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