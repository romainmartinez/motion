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
    np.testing.assert_array_equal(
        x=raveled[0], y=first_last_val[0], err_msg="First value does not match"
    )
    np.testing.assert_array_equal(
        x=raveled[-1], y=first_last_val[-1], err_msg="Last value does not match"
    )
    np.testing.assert_array_almost_equal(
        x=array.mean(), y=mean_val, decimal=decimal, err_msg="Mean does not match"
    )
    np.testing.assert_array_almost_equal(
        x=array.median(), y=median_val, decimal=decimal, err_msg="Median does not match"
    )
    np.testing.assert_array_almost_equal(
        x=array.sum(), y=sum_val, decimal=decimal, err_msg="Sum does not match"
    )
    np.testing.assert_array_equal(
        x=array.isnull().sum(), y=nans_val, err_msg="Nans value value does not match"
    )


def print_expected_values(array: xr.DataArray):
    shape_val = array.shape
    print(f"{shape_val=}")

    ravel = array.values.ravel()
    first_last_val = ravel[0], ravel[-1]
    print(f"{first_last_val=}")

    mean_val = array.mean().item()
    print(f"{mean_val=}")

    median_val = array.median().item()
    print(f"{median_val=}")

    sum_val = array.sum().item()
    print(f"{sum_val=}")

    nans_val = array.isnull().sum().item()
    print(f"{nans_val=}")
