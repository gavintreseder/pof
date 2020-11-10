import logging

import dash_core_components as dcc
import dash_html_components as html


class DashLogger(logging.StreamHandler):
    """
    A class to enable logging to be outptu to a Dash object.

    Inspired by: https://www.pueschel.dev/python,/dash,/plotly/2019/06/28/dash-logs.html

    Usage:

    logger = logging.getLogger(__name__)

    dash_logger = DashLogger(stream=sys.stdout)
    logger.addHandler(dash_logger)


    """

    def __init__(self, stream=None):
        super().__init__(stream=stream)
        self.logs = list()

    def emit(self, record):
        try:
            msg = self.format(record)
            self.logs.append(msg)
            self.logs = self.logs[-1000:]
            self.flush()
        except Exception:
            self.handleError(record)

    def get_layout(self, prefix=""):
        layout = [
            dcc.Interval(id="log-update", interval=1 * 1000),  # in milliseconds
            html.Div(id=prefix + "log"),
        ]

        return layout


if __name__ == "__main__":
    dash_logger = DashLogger()
    print("DashLogger - Ok")
