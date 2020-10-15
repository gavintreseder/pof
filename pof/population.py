# Number of failures in the population


def get_population_failures(comp, population_of_interest, fleet_data):

    df_failures = NotImplemented
    # Get the failure rates

    # Get the population summary

    # Calculate the failure rate

    return df_failures


def get_population_summary(attribute, time_frame=1):

    #

    """

    Challenges categorial v numerical

        Categorical -> list of attributes to keep
        Numerical -> bin size / n_increments
            Where to start it?

    df ->

    df
    Material Type   Count
    CCA             220000
    NR              350000

    Material Type   Age     Count
    CCA             0       1000
                    10      2xxxx
                    20      6000000

    NR


    Only give me timber poles
    Only give me timber poles, but break the summary up by treatment type
    Only give me timber poles, but break the summary up by treatment type and age by decade
    Only give me timber poles, but break the summary up by poles that have condition loss


    """

    col = "Material Type"

    col_values = ["CCA", "Natural Round"]

    attributes = {
        "material_type": ["timber"],
        "treatment_type": ["CCA", "NR"],
        "age": [],
    }

    # df filterning magic

    # Summary

    # TODO ask Gav for futher guidance

    return NotImplemented


"""

Generate a fleet data object, with data that you know

Data set with 

Att 1
    type 1, type 2

Att 2
    var 1, var 2, var 3 

Numerical att
    ss.dist.norm(....)

"""


# Test Data function


def gen_fleet_data(n_assets):

    params = NotImplemented
    return FleetData(params)


# Fleet Data


fd = gen_fleet_data(1000)

fd.field_types

dict(
    material_type="categorical",
    age="numerical",
)

"""Transformer Volatge
11
66
195"""

str, int, float