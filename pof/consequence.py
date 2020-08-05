


class Consequence():

    def __init__(self):

        self.risk_cost_total = 50000

    def set_risk_cost_total(self, cost):

        self.risk_cost_total = cost

        return self

    def get_cost(self):
        return self.risk_cost_total