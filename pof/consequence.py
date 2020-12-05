from pof.pof_base import PofBase


class Consequence(PofBase):
    def __init__(self, risk_cost_total=50000):

        self.risk_cost_total = risk_cost_total

    def _load(self, *args, **kwargs):

        self.risk_cost_total = 50000

    def set_risk_cost_total(self, cost):

        self.risk_cost_total = cost

        return self

    def get_cost(self):
        return self.risk_cost_total


if __name__ == "__main__":
    consequence = Consequence()
    print("Consequence - Ok")
