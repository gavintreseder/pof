from dash.dependencies import Input, Output
import plotly.express as px

from ..app import app


def update_pof_figure():
    pof = comp.expected_pof(t_end=200)

    df = pd.DataFrame(pof)
    df.index.name = 'time'
    df = df.reset_index().melt(id_vars = 'time', var_name='source', value_name='pof')

    fig = px.line(df,x='time', y='pof', color='source')
    return fig

update_pof_figure().show()