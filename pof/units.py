
import copy
import logging

valid_units = dict(
    seconds=1 / 3600,
    minutes=1 / 60,
    hours=1,
    days=24,
    weeks=168,
    months=730,  # 24 * 365 / 12
    years=8760,  # 24 * 365
)

def scale_units(df, input_units: str = None, model_units: str = None):

    # expand to include all cols
    time_cols = ["time", "age"]
    unit_cols = list(set(time_cols) & set(df.columns))

    input_factor = valid_units.get(input_units, None)
    model_factor = valid_units.get(model_units, None)

    units = input_units

    if input_factor is None:
        units = model_units
    elif model_factor is None:
        logging.warning("Invalid model units. No scaling completed")
    else:
        ratio = model_factor / input_factor

        # Scale the columns
        df = copy.deepcopy(df)
        df.loc[:, unit_cols] = df[unit_cols] * ratio

        # Scale the index
        if df.index.name in time_cols:
            df.index *= ratio

    return df, units