
class State:
    def __init__(self):

        self._initiated = False
        self._detected = False
        self._failed = False

if __name__ == "__main__":
    state = State()
    print("State - Ok")