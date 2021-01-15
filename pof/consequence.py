from pof.pof_base import PofBase
from config import config

# cf = config.get("Consequence")


class Consequence(PofBase):
    def __init__(
        self, name: str = "consequence", cost=None, group=None, *args, **kwargs
    ):
        super().__init__(name=name, *args, **kwargs)

        self.cost = cost
        self.group = group


if __name__ == "__main__":
    consequence = Consequence()
    print("Consequence - Ok")
