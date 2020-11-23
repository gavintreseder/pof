import unittest

import testconfig
from pof.decorators import check_arg_positive, coerce_arg_type, check_arg_type


# given, should, actual, expected


class TestDecorators(unittest.TestCase):
    """"""

    # ------------ Class Functions ----------------
    @check_arg_positive("b", "c")
    def arg_positive(self, a, b=2, c=-3, *args, **kwargs):
        return (a, b, c)

    # ------------- Check arg positive ------------

    def test_check_arg_positive(self):

        param_args_kwargs_expected = [
            ((-1, 1, 1, -1), {}, (-1, 1, 1)),
            ((), {"a": -1, "b": 1, "c": 1, "d": -1}, (-1, 1, 1)),
            ((-1,), {"c": 1}, (-1, 2, 1)),
        ]

        for args, kwargs, expected in param_args_kwargs_expected:

            given = "positive inputs"
            should = "execute the function without error"
            actual = self.arg_positive(*args, **kwargs)

            self.assertEqual(actual, expected, msg=f"Given {given}: should {should}")

    def test_check_arg_positive_with_errors(self):

        # args, kwargs
        param_args_kwargs = [
            ((1, -1, -1, 1), {}),
            ((), {"a": 1, "b": -1, "c": -1, "d": 1}),
            ((-1,), {}),
        ]

        for args, kwargs in param_args_kwargs:

            with self.assertRaises(ValueError):
                given = "negative inputs"
                should = "raise an error"
                actual = self.arg_positive(*args, **kwargs)
                expected = ValueError

                self.assertEqual(
                    actual, expected, msg=f"Given {given}: should {should}"
                )

    def test_chain(self):

        # Arrange
        @coerce_arg_type
        @check_arg_positive("c")
        def func(a, b, c: int = 3, *args, **kwargs):
            return (a, b, c)

        args = (1, 2, "3")
        given = "a chain of decorators"
        should = "apply all decorators"
        expected = (1, 2, 3)

        # Act
        actual = func(*args)

        # Assert
        self.assertEqual(actual, expected, msg=f"Given {given}: should {should}")
