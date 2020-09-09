

class Consequence:
    def __init__(self):

        self.risk_cost_total = 50000

    def load(self, kwargs=None):
        try:
            self._load(**kwargs)
        except:
            print("Error loading Consequence data")

        return self

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
