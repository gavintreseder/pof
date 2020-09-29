import pandas as pd
import numpy as np

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pof import Component, FailureMode


def update_cost_fig(local):
    try:
        df = local.expected_risk_cost_df(t_end=200)
        df.columns = df.columns.str.replace("_", " ").str.title()

        if isinstance(local, Component):
            fig = px.area(
                df,
                x="Time",
                y="Cost Cumulative",
                color="Task",
                line_group="Failure Mode",
                title="Maintenance Strategy Costs",
            )
        elif isinstance(local, FailureMode):
            fig = px.area(
                df,
                x="Time",
                y="Cost Cumulative",
                color="Task",
                title="Maintenance Strategy Costs",
            )
        else:
            raise TypeError("local must be Component of FailureMode")
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

        df = pd.concat(pof).rename_axis(["strategy", "time"]).reset_index()
        df = df.melt(id_vars=["time", "strategy"], var_name="source", value_name="pof")

        fig = px.line(
            df,
            x="time",
            y="pof",
            color="source",
            line_dash="strategy",
            line_group="strategy",
            title="Probability of Failure given Maintenance Strategy",
        )
        fig.layout.yaxis.tickformat = ",.0%"
    except:
        fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text="Error Producing Probability of Failure")
            )
        )

    return fig


def update_condition_fig(local):
    """ Updates the condition figure"""

    try:

        ecl = local.expected_condition_loss()

        fig = make_subplots(
            rows=len(ecl), shared_xaxes=True, vertical_spacing=0.05, title="Condition"
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
            fig.update_yaxes(title_text=cond_name.replace("_", " ").title(), row=idx)
            idx = idx + 1

        fig.update_traces(mode="lines")
        fig.update_xaxes(title_text="Time", row=len(ecl))

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
    except:
        fig = go.Figure(
            layout=go.Layout(title=go.layout.Title(text="Error Producing Condition"))
        )

    return fig


def humanise(data):
    # Not used
    if isinstance(pd.DataFrame):
        data.columns = data.columns.str.replace("_", " ").str.title()
        data
    elif isinstance(str):
        data = data.replace("_", " ").title()
