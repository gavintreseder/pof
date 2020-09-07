import pandas as pd
import numpy as np

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go

def update_cost_fig(local):
    try:
        df = local.expected_risk_cost_df(t_end=200)
        
        fig = px.area(
            df,
            x="time",
            y="cost_cumulative",
            color="task",
            line_group='failure_mode',
            title="Maintenance Strategy Costs",
        )
    except:
        fig = go.Figure(title = "Error Producing Maintenance Strategy Costs")
    return fig

def update_pof_fig(local):

    try:
        pof = dict(
            treated = pd.DataFrame(local.expected_pof(t_end=200)),
            untreated = pd.DataFrame(local.expected_untreated(t_end=200)),
        )

        df = pd.concat(pof).rename_axis(['strategy', 'time']).reset_index()
        df = df.melt(id_vars = ['time', 'strategy'], var_name='source', value_name='pof')

        fig = px.line(df,x='time', y='pof', color='source', line_dash = 'strategy', line_group='strategy', title = 'Probability of Failure')
    except:
        fig = go.Figure(title = "Error Producing Probability of Failure")

    return fig

def update_condition_fig(local):
    """ Updates the condition figure"""

    try:
        fig = go.Figure()
        ecl = local.expected_condition_loss()

        cmap = px.colors.qualitative.Safe
        ci = 0

        for cond_name, cond in ecl.items():
            # Format the data for plotting
            length = len(cond['mean'])
            time = np.linspace(0,length  -1, length, dtype=int) # TODO take time as a variable
            x = np.append(time,time[::-1])
            y = np.append(cond['upper'], cond['lower'][::-1])

            # Add the boundary
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                fill='toself',
                fillcolor='rgba' + cmap[ci][3:-2] + ',0.2)',
                line_color='rgba(255,255,255,0)',
                showlegend=False,
                name=cond_name,
            ))
            fig.add_trace(go.Scatter(
                x=time, y=cond['mean'],
                line_color=cmap[ci],
                name=cond_name,
            ))

            ci = ci + 1

            fig.update_traces(mode='lines')
            fig.update_xaxes(title_text='Time')
            fig.update_yaxes(title_text='Condition (%)')
    except:
        fig = go.Figure(title = "Error Producing Condition")

    return fig