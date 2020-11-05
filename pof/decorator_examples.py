#TODO Illyse

#tips

# Step 1 - trial it in a notebook

def func(value):
    
    #do somethign wtih value

    return value

def check_is_negative(func):

    def wrapper(*args, **kwargs):

        # check *args and **wargs are right

        result = func(*args, **kwargs)

        # Chck the result is right

    return wrapper

#inspect.signature
# inspect.getfullargspec

def wrapper(self, value):
    if value < 0:
        raise ValueError
    return func(self, value)


@check_arg_positive('value') #params
def pf_interval(self, value)
def pf_interval(self, pf_interval)


def pf_interval(value)
def pf_interval(self, value)
defpf_interval(self, other_dist, value=-10)

pf_interval() # Uses the default
pf_interval(-10, -10) # args
pf_interval(value=-10) # kwarg
