import inspect
import re
import time
from functools import wraps

import numpy as np
import xarray as xr


def is_expected_array(
    array,
    shape_val: tuple,
    first_last_val: tuple,
    mean_val: float,
    median_val: float,
    sum_val: float,
    nans_val: int,
    decimal: int = 6,
):
    np.testing.assert_array_equal(
        x=array.shape, y=shape_val, err_msg="Shape does not match"
    )
    raveled = array.values.ravel()
    np.testing.assert_array_almost_equal(
        x=raveled[0],
        y=first_last_val[0],
        err_msg="First value does not match",
        decimal=decimal,
    )
    np.testing.assert_array_almost_equal(
        x=raveled[-1],
        y=first_last_val[-1],
        err_msg="Last value does not match",
        decimal=decimal,
    )
    np.testing.assert_array_almost_equal(
        x=array.mean(), y=mean_val, decimal=decimal, err_msg="Mean does not match"
    )
    np.testing.assert_array_almost_equal(
        x=array.median(skipna=True), y=median_val, decimal=decimal, err_msg="Median does not match"
    )
    np.testing.assert_allclose(
        actual=array.sum(), desired=sum_val, rtol=0.05, err_msg="Sum does not match"
    )
    np.testing.assert_array_equal(
        x=array.isnull().sum(), y=nans_val, err_msg="Nans value value does not match"
    )


def print_expected_values(array: xr.DataArray):
    shape_val = array.shape
    print(f"shape_val={shape_val}")

    ravel = array.values.ravel()
    first_last_val = ravel[0], ravel[-1]
    print(f"first_last_val={first_last_val}")

    mean_val = array.mean().item()
    print(f"mean_val={mean_val}")

    median_val = array.median(skipna=True).item()
    print(f"median_val={median_val}")

    sum_val = array.sum().item()
    print(f"sum_val={sum_val}")

    nans_val = array.isnull().sum().item()
    print(f"nans_val={nans_val}")


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        print(f.__name__)

        start = time.time()
        result = f(*args, **kwargs)

        print(f"\t{f.__name__} succeed ({time.time() - start})")
        return result

    return wrapper


def function_has_return(func):
    """Caution: this will return True if the function contains the word 'return'"""
    lines, _ = inspect.getsourcelines(func)
    return any("return" in line for line in lines)


def extract_code_blocks_from_md(
    docstring: str, start_code_block: str = "```python", end_code_block: str = "```"
) -> str:
    return inspect.cleandoc(
        "\n".join(
            re.findall(
                fr"{start_code_block}(.*?){end_code_block}", docstring, re.DOTALL
            )
        )
    )
