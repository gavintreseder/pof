"""
Validators that can be used to decorate pof class methods
"""

from functools import wraps
import logging
import inspect


def check_arg_type(func):
    """
    Checks the args match the input

    Usage:
    >>> @check_arg_type
    ... def func(x: int, y: str):
    ...     return (x, y)

    >>> func(10.0, 2)
    Traceback (most recent call last):
        ...
    TypeError: 10.0 is not of type <class 'int'>

    >>> func(10, 2)
    Traceback (most recent call last):
        ...
    TypeError: 2 is not of type <class 'str'>

    >>> func(10, '2')
    (10, '2')
    """

    @wraps(func)
    def wrapper(*args):
        for index, arg in enumerate(inspect.getfullargspec(func)[0]):
            if not isinstance(args[index], func.__annotations__[arg]):
                raise TypeError(
                    f"{args[index]} is not of type {func.__annotations__[arg]}"
                )

        return func(*args)

    return wrapper


def coerce_arg_type(func):
    """
    Checks the args match the input

    Usage:
    >>> @check_arg_type
    ... def func(x: int, y: str):
    ...     return (x, y)

    >>> func(10.0, 2)
    Traceback (most recent call last):
        ...
    TypeError: 10.0 is not of type <class 'int'>

    >>> func(10, 2)
    Traceback (most recent call last):
        ...
    TypeError: 2 is not of type <class 'str'>

    >>> func(10, '2')
    (10, '2')
    """

    def _f(*args):
        new_args = []
        for index, arg in enumerate(inspect.getfullargspec(func)[0]):
            new_args.append(func.__annotations__[arg](args[index]))
        return func(*new_args)

    _f.__doc__ = func.__doc__
    return _f


# Options

# Raise Errors

# Log Errors and no change

# Log error and use default


def check_positive(func):
    """
    Checks the arg is positive

    Usage:
    >>> @check_positive
    ... def func(value):
    ...     return value

    >>> func(10)
    10

    >>> func(-8)
    Traceback (most recent call last):
        ...
    ValueError: -8 is not positive
    """

    @wraps(func)
    def wrapper(value, *args, **kwargs):
        if value >= 0:
            return value
        else:
            raise ValueError(f"{value} is not positive")

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
