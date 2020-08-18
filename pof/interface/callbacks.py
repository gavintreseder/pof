from dash.dependencies import Input, Output
import plotly.express as px

from ..app import app


@app.callback(
    Output(component_id="maintenance_strategy", component_property="figure"),
    [
        Input(component_id="graph_y_limit", component_property="value"),
        Input(component_id="p_effective", component_property="value"),
        Input(component_id="task_checklist", component_property="value"),
        Input(component_id="consequence", component_property="value"),
        Input(component_id="inspection_interval", component_property="value"),
    ],
)
def update_maintenance_strategy(
    graph_y_limit, p_effective, tasks, consequence, inspection_interval
):


    fm_local = copy.deepcopy(fm)
    fm_local.dash_update(dash_id)

    fm_local.mc_timeline(t_end=200, n_iterations=100)
    df = fm_local.expected_cost_df()

    fig = px.area(
        df,
        x="time",
        y="cost_cumulative",
        color="task",
        title="Maintenance Strategy Costs",
    )

    fig.update_yaxes(range=[0, graph_y_limit])

    return fig