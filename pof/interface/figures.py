import pandas as pd
import numpy as np

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pof import Component, FailureMode


def get_color_map(df, column, colour_scheme=None):

    if colour_scheme == "plotly":
        colors = px.colors.qualitative.Plotly
    elif colour_scheme == "bold":
        colors = px.colors.qualitative.Bold
    else:
        colors = px.colors.qualitative.Safe

    color_map = dict(zip(df[column].unique(), colors))

    return color_map


def update_cost_fig(local):
    try:
        df = local.expected_risk_cost_df()

        df.columns = df.columns.str.replace("_", " ").str.title()

        color_map = get_color_map(df=df, column="Task", colour_scheme="bold")

        df = df[df["Fm Active"] == True]
        df = df[df["Task Active"] == True]

        if isinstance(local, Component):
            fig = px.area(
                df,
                x="Time",
                y="Cost Cumulative",
                color="Task",
                color_discrete_map=color_map,
                line_group="Failure Mode",
                title="Maintenance Strategy Costs",
            )
        elif isinstance(local, FailureMode):
            fig = px.area(
                df,
                x="Time",
                y="Cost Cumulative",
                color="Task",
                color_discrete_map=color_map,
                title="Maintenance Strategy Costs",
            )
        else:
            raise TypeError("local must be Component of FailureMode")
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)
        fig.update_layout(
            legend_traceorder="reversed",
        )

    except:
        fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text="Error Producing Maintenance Strategy Costs")
            )
        )
    return fig


def update_pof_fig(local):

    try:
        pof = dict(
            maint=pd.DataFrame(local.expected_pof(t_end=200)),
            no_maint=pd.DataFrame(local.expected_untreated(t_end=200)),
        )

        # OLD CODE
        # df = pd.concat(pof).rename_axis(["strategy", "time"]).reset_index()
        # df.index.names = ["strategy", "key"]
        # df = df.rename(columns={"variable": "source"})

        # df = df.melt(
        #     id_vars=["time", "strategy", "fm_active"],
        #     var_name="source",
        #     value_name="pof",
        # )

        df = pd.concat(pof).rename_axis(["strategy", "key"]).reset_index()

        df_pof = df[df["key"] == "pof"]
        df_pof = pd.melt(
            df_pof, id_vars=["strategy"], value_vars=df_pof.columns[2:]
        ).rename(columns={"variable": "source", "value": "pof"})
        df_pof = df_pof.explode("pof", ignore_index=True)

        df_active = df[df["key"] == "fm_active"]
        df_active = pd.melt(
            df_active, id_vars=["strategy"], value_vars=df_active.columns[2:]
        ).rename(columns={"variable": "source", "value": "fm_active"})

        df_time = df[df["key"] == "time"]
        df_time = pd.melt(
            df_time, id_vars=["strategy"], value_vars=df_time.columns[2:]
        ).rename(columns={"variable": "source", "value": "time"})
        df_time = df_time.explode("time", ignore_index=True)["time"]

        df = df_pof.merge(df_active, on=["strategy", "source"])
        df["time"] = df_time

        color_map = get_color_map(df=df, column="source", colour_scheme="plotly")

        df = df[df["fm_active"] == True]

        fig = px.line(
            df,
            x="time",
            y="pof",
            color="source",
            color_discrete_map=color_map,
            line_dash="strategy",
            line_group="strategy",
            title="Probability of Failure given Maintenance Strategy",
        )
        fig.layout.yaxis.tickformat = ",.0%"
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)
    except:
        fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text="Error Producing Probability of Failure")
            )
        )

    return fig


def update_condition_fig(local, conf=0.95):
    """ Updates the condition figure"""

    try:

        ecl = local.expected_condition()

        subplot_titles = [x.replace("_", " ").title() for x in list(ecl.keys())]

        fig = make_subplots(
            rows=len(ecl),
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=subplot_titles,
        )

        cmap = px.colors.qualitative.Safe
        idx = 1

        for cond_name, cond in ecl.items():
            # Format the data for plotting
            length = len(cond["mean"])
            time = np.linspace(
                0, length - 1, length, dtype=int
            )  # TODO take time as a variable
            x = np.append(time, time[::-1])
            y = np.append(cond["upper"], cond["lower"][::-1])

            # Add the boundary
            fig.add_trace(
                go.Scatter(
                    x=x,
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
            fig.update_yaxes(
                title_text="".join(
                    [x[0] for x in cond_name.replace("_", " ").title().split(" ")]
                )
                + " Measure",
                row=idx,
                automargin=True,
            )
            idx = idx + 1

        fig.update_traces(mode="lines")
        fig.update_xaxes(title_text="Time", row=len(ecl), automargin=True)
        fig.update_layout(
            title="Expected Condition (Confidence = " + f"{conf}" + ")",
            legend_traceorder="normal",
        )

    except:
        fig = go.Figure(
            layout=go.Layout(title=go.layout.Title(text="Error Producing Condition"))
        )

    return fig


def make_inspection_interval_fig(local, t_min=0, t_max=10, step=1, n_iterations=10):

    try:
        df = local.expected_inspection_interval(
            t_min=t_min, t_max=t_max, step=step, n_iterations=n_iterations
        )

        df_plot = df.melt(
            id_vars="inspection_interval", var_name="source", value_name="cost"
        )

        fig = px.line(
            df_plot,
            x="inspection_interval",
            y="cost",
            color="source",
            title="Risk v Cost at different Inspection Intervals",
        )
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)

    except:
        fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text="Error Producing Inspection Interval")
            )
        )

    return fig


def make_sensitivity_fig(
    local, var_name="", t_min=0, t_max=10, step_size=1, n_iterations=10
):

    var = var_name.split("-")[-1]

    title_var = var.replace("_", " ").title()

    try:
        df, df_active = local.expected_sensitivity(
            var_name=var_name,
            t_min=t_min,
            t_max=t_max,
            step_size=step_size,
            n_iterations=n_iterations,
        )

        df_plot = df.melt(id_vars=var, var_name="source", value_name="cost")

        df_plot = df_plot.merge(df_active, on="source")

        color_map = get_color_map(df=df_plot, column="source")

        df_plot = df_plot[df_plot["fm_active"] == True]
        df_plot = df_plot[df_plot["task_active"] == True]
        print(df_plot)

        fig = px.line(
            df_plot,
            x=var,
            y="cost",
            color="source",
            color_discrete_map=color_map,
            title="Risk v Cost at different " + f"{title_var}" + "s",
        )
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)

    except:
        fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text="Error Producing Inspection Interval")
            )
        )

    return fig


def humanise(data):
    # Not used
    if isinstance(pd.DataFrame):
        data.columns = data.columns.str.replace("_", " ").str.title()
        data
    elif isinstance(str):
        data = data.replace("_", " ").title()
