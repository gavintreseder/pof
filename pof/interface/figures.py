"""

"""

from typing import List

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# from pof import Component, FailureMode


def get_color_map(df, column):

    if column == "source":
        colors = px.colors.qualitative.Plotly
    elif column == "task":
        colors = px.colors.qualitative.Bold
    else:
        colors = px.colors.qualitative.Safe

    color_map = dict(zip(df[column].unique(), colors))

    return color_map


def make_ms_fig(
    df, y_axis="cost_cumulative", keep_axis=False, units="unknown", prev=None
):
    try:
        color_map = get_color_map(df=df, column="task")
        df = df[df["active"]]

        # Format the labels
        labels = {label: label.replace("_", " ").title() for label in list(df)}
        labels["time"] = f"Age ({units})"

        px_args = dict(
            data_frame=df,
            x="time",
            y=y_axis,
            color="task",
            color_discrete_map=color_map,
            labels=labels,
            title="Maintenance Strategy Costs",
        )

        fig = px.area(**px_args, line_group="failure_mode")

        y_max = calc_y_max(keep_axis=keep_axis, method="sum", prev=prev)
        if y_max is not None:
            fig.update_yaxes(range=[0, y_max])

        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)

        fig = update_visibility(fig, prev)

    except Exception as error:
        raise (error)
        fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text="Error Producing Maintenance Strategy Costs")
            )
        )
    return fig


def update_pof_fig(df, keep_axis=False, units="unknown", prev=None):

    try:

        color_map = get_color_map(df=df, column="source")

        df = df[df["active"] == True]

        # Make columns presentable
        # df.columns = df.columns.str.replace("_", " ").str.title()
        col_names = {"time": f"Age ({units})"}
        df.rename(columns=col_names, inplace=True)

        fig = px.line(
            df,
            x=col_names["time"],
            y="pof",
            color="source",
            color_discrete_map=color_map,
            line_dash="strategy",
            line_group="strategy",
            title="Probability of Failure given Maintenance Strategy",
        )

        col_names = {"time": f"Age ({units})"}

        fig.layout.yaxis.tickformat = ",.0%"
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)
        fig.update_xaxes(title_text=col_names["time"])

        y_max = calc_y_max(keep_axis=keep_axis, method="max", prev=prev)
        if y_max is not None:
            fig.update_yaxes(range=[0, y_max])

        fig = update_visibility(fig, prev)

    except:
        fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text="Producing Probability of Failure - Error")
            )
        )

    return fig


def update_condition_fig(
    df, ecl, conf=0.95, keep_axis=False, units="unknown", prev=None
):
    """ Updates the condition figure"""

    # TODO remake this so that it takes a df in the right format and filter, rather than repeat expected condition twice

    try:
        subplot_titles = [x.replace("_", " ").title() for x in list(ecl.keys())]

        fig = make_subplots(
            rows=len(ecl),
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=subplot_titles,
        )

        cmap = px.colors.qualitative.Safe
        idx = 1
        time = df["time"]
        for cond_name, cond in ecl.items():
            # Format the y_axis titles
            y_title = "".join(
                [x[0] for x in cond_name.replace("_", " ").title().split(" ")]
            )

            # Format the data for plotting
            length = len(cond["mean"])
            # time = np.linspace(
            #     0, length - 1, length, dtype=int
            # )  # TODO take time as a variable
            # x = np.append(time, time[::-1])
            y = df["y" + str(idx)]

            # Add the boundary
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=y,
                    fill="toself",
                    fillcolor="rgba" + cmap[idx][3:-2] + ",0.2)",
                    line_color="rgba(255,255,255,0)",
                    showlegend=False,
                    name=cond_name,
                ),
                row=idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=cond["mean"],
                    line_color=cmap[idx],
                    name=cond_name,
                    showlegend=False,
                ),
                row=idx,
                col=1,
            )
            fig.update_yaxes(  # Not using y_max method as maximum condition (perfect) will always be max
                title_text=y_title + " Measure",
                row=idx,
                automargin=True,
                range=[0, max(y)],
            )

            idx = idx + 1

        col_names = {"time": f"Age ({units})"}

        fig.update_traces(mode="lines")
        fig.update_xaxes(
            title_text=col_names["time"],
            row=len(ecl),
            automargin=True,
        )
        fig.update_layout(
            title="Expected Condition (Confidence = " + f"{conf}" + ")",
            legend_traceorder="normal",
        )

    except:
        fig = go.Figure(
            layout=go.Layout(title=go.layout.Title(text="Error Producing Condition"))
        )

    return fig


def make_sensitivity_fig(
    df_plot,
    var_name="",
    y_axis="",
    keep_axis=False,
    units="unknown",
    prev=None,
):

    var = var_name.split("-")[-1]

    title_var = var.replace("_", " ").title()

    try:

        # Add line dashes
        df_plot[" "] = "  "
        df_plot.loc[df_plot["source"].isin(["risk", "direct", "total"]), " "] = "   "

        # Add the colours
        color_map = get_color_map(df=df_plot, column="source")
        if "active" in df_plot:
            df_plot = df_plot.loc[df_plot["active"]]

        # Adjust the labels

        fig = px.line(
            df_plot,
            x=var,
            y=y_axis,
            color="source",
            color_discrete_map=color_map,
            line_dash=" ",
            title=f"Sensitivity of Risk/Cost to {title_var}",
        )

        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)

        y_max = calc_y_max(keep_axis=keep_axis, method="max", prev=prev)
        if y_max is not None:
            fig.update_yaxes(range=[0, y_max])

        if var in (
            "t_delay",
            "t_interval",
            "pf_interval",
            "pf_std",
            "alpha",
            "beta",
            "gamma",
        ):
            col_names = {"time": f"{var} ({units})"}
            fig.update_xaxes(title_text=col_names["time"])

        fig = update_visibility(fig, prev)
        # fig.update_layout(xaxis=dict(tickvals=df_plot[var].tolist()))

    except Exception as error:
        raise error
        fig = go.Figure(
            layout=go.Layout(title=go.layout.Title(text=f"Error Producing {title_var}"))
        )

    return fig


def make_task_forecast_fig(df, y_axis="pop_quantity", keep_axis=False, prev=None):

    title = "Population Forecast"

    try:
        color_map = get_color_map(df=df, column="task")

        df = df[df["active"]]

        fig = px.line(
            df,
            x="year",
            y=y_axis,
            color="task",
            color_discrete_map=color_map,
            title=title,
        )
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)

        y_max = calc_y_max(keep_axis=keep_axis, method="max", prev=prev)
        if y_max is not None:
            fig.update_yaxes(range=[0, y_max])

        fig = update_visibility(fig, prev)

    except Exception as error:
        fig = go.Figure(
            layout=go.Layout(title=go.layout.Title(text=f"Error Producing {title}"))
        )

        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)
        fig.update_layout(xaxis=dict(tickvals=df["year"].tolist()))

    return fig


def make_pop_table_fig(df):

    alignment = {"fm": "left"}
    align = [[alignment[h]] if h in alignment else ["right"] for h in df.columns]

    tbl = go.Table(
        header=dict(values=list(df.columns), align=align),
        cells=dict(values=[df[col] for col in list(df)], align=align, format=[".2f"]),
    )

    fig = go.Figure(data=[tbl])

    fig.update_layout(title="Forecast Summary")

    return fig


def make_table_fig(df, title = "Forecast Summary"):
    """
    Produces a dash table to summarise key statistics by failure mode
    """
    # Format the decimal places for float columns
    int_cols = list(df)
    int_cols.remove("fm")
    df.loc[:, int_cols] = df.loc[:, int_cols].astype(float).round(2)
    df.fillna("-", inplace=True)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=list(df.columns), align="left"),
                cells=dict(
                    values=[df[col] for col in list(df)],
                    align="left",
                ),
            )
        ]
    )

    fig.update_layout(title=title)

    return fig


def humanise(data):
    # Not used
    if isinstance(pd.DataFrame):
        data.columns = data.columns.str.replace("_", " ").str.title()

    elif isinstance(str):
        data = data.replace("_", " ").title()


def update_visibility(curr, prev=None):
    """Updates the visibility based on the visibility previously selected"""
    if prev is not None:
        visibilities = {d.get("name"): d.get("visible") for d in prev["data"]}
        # This is updating both tasks

        for trace in curr["data"]:
            if trace.name in list(visibilities.keys()):
                trace.update(visible=visibilities[trace.name])
            else:
                trace.update(visible=True)

    return curr


def calc_y_max(keep_axis, method, prev, test=False):
    """ Determines the y_max value of the previous chart """

    scale = 1.05
    y_max = None
    if keep_axis:
        if prev is not None:
            # Data stored differently in test/notebooks TODO this could be better
            if test:
                range_prev = prev["layout"].yaxis.range
            else:
                range_prev = prev["layout"].get("yaxis").get("range")

            # If the y_limit already exists use last one, otherwise calculate a new y_max
            if range_prev is not None:
                y_max = prev["layout"].get("yaxis").get("range")[1]
            else:
                y_vals = []
                for trace in prev["data"]:
                    # Data stored differently in test/notebooks
                    if test:
                        y_vals.append(max(trace.y))
                    else:
                        y_vals.append(max(trace.get("y")))

                if method == "sum":  # If area chart
                    y_max = sum(y_vals) * scale
                elif method == "max":  # For all other charts
                    y_max = max(y_vals) * scale

    return y_max
