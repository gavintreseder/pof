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

    p_effective = p_effective if p_effective is not None else 0
    consequence = consequence if consequence is not None else 0
    inspection_interval = inspection_interval if inspection_interval is not None else 0
    graph_y_limit = graph_y_limit if graph_y_limit is not None else 10000

    if tasks is None:
        tasks = []

    for task_name, task in fm_local.tasks.items():
        if task_name in tasks:
            task.active = True
        else:
            task.active = False

    # fm.conditions['wall_thickness'] = Condition(100, 0, 'linear', [-5])
    fm_local.tasks["inspection"].p_effective = p_effective / 100
    fm_local.cof.risk_cost_total = consequence
    fm_local.tasks["inspection"].t_interval = inspection_interval
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