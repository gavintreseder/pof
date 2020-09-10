

class Consequence:
    def __init__(self, risk_cost_total = 50000):

        self.risk_cost_total = risk_cost_total

    @classmethod
    def from_dict(cls, details=None):
        try:
            csq = cls(**details)
        except:
            print("Error loading Consequence data from dictionary")
        return csq

    @classmethod
    def load(cls, details=None, *args, **kwargs):
        try:
            csq = cls.from_dict(**details)
        except:
            csq = cls()
            print("Error loading Consequence data")

        return csq

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
