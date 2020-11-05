"""
Decoarators that can be used to validate pof class methods
"""

from functools import wraps
import logging
import inspect


def check_arg_type(func):
    """
    Checks the args match the input type annotations

    Usage:
    >>> @check_arg_type
    ... def func(x: int, y: str, z):
    ...     return (x, y, z)

    >>> func(3.0, 2, 1)
    Traceback (most recent call last):
        ...
    TypeError: 3.0 is not of type <class 'int'>

    >>> func(3, 2, 1)
    Traceback (most recent call last):
        ...
    TypeError: 2 is not of type <class 'str'>

    >>> func(3, '2', 1)
    (3, '2', 1)
    """

    @wraps(func)
    def wrapper(*args):
        for index, arg in enumerate(inspect.getfullargspec(func)[0]):
            if arg in func.__annotations__:
                if not isinstance(args[index], func.__annotations__[arg]):
                    raise TypeError(
                        f"{args[index]} is not of type {func.__annotations__[arg]}"
                    )

        return func(*args)

    return wrapper


def coerce_arg_type(func):
    """
    Coerces the args to match the type annotations

    Usage:
    >>> @coerce_arg_type
    ... def func(x: int, y: str, z):
    ...     return (x, y, z)

    >>> func(3.0, 2, 1)
    (3, '2', 1)

    >>> func(3, '2', 1)
    (3, '2', 1)
    """

    @wraps(func)
    def wrapper(*args):
        args = list(args)
        for index, arg in enumerate(inspect.getfullargspec(func)[0]):
            if arg in func.__annotations__:
                args[index] = func.__annotations__[arg](args[index])
        return func(*tuple(args))

    return wrapper


def check_arg_positive(*params):
    """
    Checks the arg value is positive

    Usage:
    >>> @check_arg_positive('value')
    ... def func(value=-10, other=-6):
    ...     return value

    >>> func(10)
    10

    >>> func()
    Traceback (most recent call last):
        ...
    ValueError: -10 is not positive

    >>> func(-8)
    Traceback (most recent call last):
        ...
    ValueError: -8 is not positive
    """

    # TODO rewrite the wrapper so that it wokrs for classes and methods

    def inner(func):
        @wraps(func)
        def wrapper(*args):
            for param in params:
                param_idx = inspect.getfullargspec(func)[0].index(param)

                if args[param_idx] < 0:
                    raise ValueError(f"{args[param_idx]} is not positive")

            func(*args)

        return wrapper

    return inner


# Options

# Raise Errors

# Log Errors and no change

# Log error and use default


def check_in_list(func, valid_list):
    """ Validates a pf_curve"""

    @wraps(func)
    def wrapper(value, *args, **kwargs):

        if value in valid_list:
            return value
        else:
            raise ValueError(f"pf_curve must be from {valid_list}")

    return wrapper


def validate_pf_curve(func):
    """ Validates a pf_curve"""

    @wraps(func)
    def wrapper(self, value):

        if value in self.PF_CURVES:
            return value
        else:
            raise ValueError(
                f"{self.__class__.__name__} - {self.name} - pf_curve must be from {self.PF_CURVES}"
            )

    return wrapper


def deafaults(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except ValueError:
            # Consider using default?

            # TODO
            NotImplemented
            # config.get(self.__class__.__name__, None).get(value)

    return wrapper


def exception_handler(*exceptions):
    """
    Logs exceptions that arne't terminal

    Usage:
    @expection_hanlder(ValueError, KeyError)
    def func(*args):
        raise ValueError

    """

    def handle_exceptions(func):
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except (exceptions) as error:
                logging.warning(error)

        return wrapper

    return handle_exceptions


def accepts(*types):
    """
    Checks data types

    Usage:

    @accepts(int, (int,float))
    def func(arg1, arg2):
        return arg1 * arg2

    func(3, 2) # -> 6
    func('3', 2)

    """

    def check_accepts(func):
        assert len(types) == func.__code__.co_argcount

        def new_func(*args, **kwds):
            for (a, t) in zip(args, types):
                assert isinstance(a, t), f"arg {args} does not match {t}" % (a, t)
            return func(*args, **kwds)

        new_func.__name__ = func.__name__
        return new_func

    return check_accepts


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)  # extraglobs={"dist": Distribution()}
    print("Validators - Ok")
