from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
import xarray as xr

from motion.io.write import write_matlab, write_csv, to_wide_dataframe
from motion.processing import algebra, filter, interp, misc


@xr.register_dataarray_accessor("meca")
class MecaDataArrayAccessor(object):
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    # io ----------------------------------------
    def to_matlab(self, filename: Union[str, Path]):
        """
        Write a matlab file from a xarray.DataArray

        TODO: example with code

        Arguments:
            filename: File path
        """
        write_matlab(self._obj, filename)

    def to_csv(self, filename: Union[str, Path], wide: Optional[bool] = True):
        """
        Write a csv file from a xarray.DataArray

        TODO: example with code

        Arguments:
            filename: File path
            wide: True if you want a wide dataframe (one column for each channel). False if you want a tidy dataframe.
        """
        write_csv(self._obj, filename, wide)

    def to_wide_dataframe(self) -> pd.DataFrame:
        """
        Return a wide xarray.DataArray (one column by channel)

        TODO: example with code

        Returns:
            A wide pandas DataFrame (one column by channel).
                Works only for 2 and 3-dimensional arrays.
                If you want a tidy dataframe type: `array.to_series()`, or `array.to_dataframe()`.
        """
        return to_wide_dataframe(self._obj)

        # algebra -----------------------------------

    def abs(self) -> xr.DataArray:
        """
        Calculate the absolute value element-wise

        TODO: example with code

        Returns:
            A `xarray.DataArray` containing the absolute of each element
        """
        return algebra.abs_(self._obj)

    def matmul(self, other: xr.DataArray) -> xr.DataArray:
        """
        Matrix product of two arrays

        TODO: example with code

        Arguments:
            other: second array to multiply

        Returns:
            A `xarray.DataArray` containing the matrix product of the two arrays
        """
        return algebra.matmul(self._obj, other)

    def square(self, *args, **kwargs) -> xr.DataArray:
        """
        Return the element-wise square of the input

        TODO: example with code

        Arguments:
            args: For other positional arguments,
              see the [numpy docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.square.html)
            kwargs: For other keyword-only arguments,
              see the [numpy docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.square.html)

        Returns:
            A `xarray.DataArray` containing the matrix squared.
        """
        return algebra.square(self._obj, *args, **kwargs)

    def sqrt(self, *args, **kwargs) -> xr.DataArray:
        """
        Return the non-negative square-root of an array, element-wise.

        TODO: example with code

        Arguments:
            args: For other positional arguments,
              see the [numpy docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sqrt.html)
            kwargs: For other keyword-only arguments,
              see the [numpy docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sqrt.html)

        Returns:
            A `xarray.DataArray` containing the square root of the matrix.
        """
        return algebra.sqrt(self._obj, *args, **kwargs)

    def norm(self, dim: Union[str, list], ord: int = None) -> xr.DataArray:
        """
        Return the norm of an array.

        TODO: example with code

        Arguments:
            dim: Name(s) of the data dimension(s)
            ord: Order of the norm

        Returns:
            A `xarray.DataArray` containing the norm of the matrix.
        """
        return algebra.norm(self._obj, dim, ord)

    def rms(self) -> xr.DataArray:
        """
        Return the root-mean-square of an array.

        TODO: example with code

        Returns:
            A `xarray.DataArray` containing the root-mean-square of the matrix.
        """
        return algebra.rms(self._obj)

    def center(
        self, mu: Union[xr.DataArray, np.array, float, int] = None
    ) -> xr.DataArray:
        """
        Center an array (i.e., subtract the mean).

        TODO: example with code

        Arguments:
            mu: mean of the signal to subtract. If not provided, takes the mean along the time_frame axis

        Returns:
            a `xarray.DataArray` containing the root-mean-square of the matrix
        """
        return algebra.center(self._obj, mu)

    def normalize(
        self,
        ref: Union[xr.DataArray, np.array, float, int] = None,
        scale: Union[int, float] = 100,
    ) -> xr.DataArray:
        """
        Normalize a signal against `ref` on a scale of `scale`.

        TODO: example with code

        Arguments:
            ref: Reference value. Could have multiple dimensions. If not provided, takes the mean along the time_frame axis
            scale: Scale on which to express array (e.g. if 100, the signal is normalized from 0 to 100)
        Returns:
            A `xarray.DataArray` containing the normalized signal
        """
        return algebra.normalize(self._obj, ref, scale)

    # interp ------------------------------------
    def time_normalize(
        self,
        time_vector: Union[xr.DataArray, np.array] = None,
        n_frames: int = 100,
        norm_time_frame: bool = False,
    ) -> xr.DataArray:
        """
        Time normalization used for temporal alignment of data

        TODO: example with code

        Arguments:
            time_vector: desired time vector (first to last time_frame with n_frames points by default)
            n_frames: if time_vector is not specified, the length of the desired time vector
            norm_time_frame: Normalize the time_frame dimension from 0 to 100 if True

        Returns:
            A time-normalized `xarray.DataArray`
        """
        return interp.time_normalize(
            self._obj, time_vector, n_frames, norm_time_frame=norm_time_frame
        )

    # filter ------------------------------------
    def low_pass(
        self, freq: Union[int, float], order: int, cutoff: Union[int, float, np.array]
    ) -> xr.DataArray:
        """
        Low-pass Butterworth filter.

        todo: example

        Arguments:
            freq: Sampling frequency
            order: Order of the filter
            cutoff: Cut-off frequency

        Returns:
            A low-pass filtered `xarray.DataArray`
        """
        return filter.low_pass(self._obj, freq, order, cutoff)

    def high_pass(
        self, freq: Union[int, float], order: int, cutoff: Union[int, float, np.array]
    ) -> xr.DataArray:
        """
        High-pass Butterworth filter.

        todo: example

        Arguments:
            freq: Sampling frequency
            order: Order of the filter
            cutoff: Cut-off frequency

        Returns:
            A high-pass filtered `xarray.DataArray`
        """
        return filter.high_pass(self._obj, freq, order, cutoff)

    def band_stop(
        self, freq: Union[int, float], order: int, cutoff: Union[list, tuple, np.array]
    ) -> xr.DataArray:
        """
        Band-stop Butterworth filter.

        todo: example

        Arguments:
            freq: Sampling frequency
            order: Order of the filter
            cutoff: Cut-off frequency such as (lower, upper)

        Returns:
            A band-stop filtered `xarray.DataArray`
        """
        return filter.band_stop(self._obj, freq, order, cutoff)

    def band_pass(
        self, freq: Union[int, float], order: int, cutoff: Union[list, tuple, np.array]
    ) -> xr.DataArray:
        """
        Band-pass Butterworth filter.

        todo: example

        Arguments:
            freq: Sampling frequency
            order: Order of the filter
            cutoff: Cut-off frequency such as (lower, upper)

        Returns:
            A band-pass filtered `xarray.DataArray`
        """
        return filter.band_pass(self._obj, freq, order, cutoff)

    # signal processing misc --------------------
    def fft(self, freq: Union[int, float], only_positive: bool = True) -> xr.DataArray:
        """
       Performs a discrete Fourier Transform and return a DataArray with the corresponding amplitudes and frequencies.

       todo: example

       Arguments:
            freq: Sampling frequency (usually in array.attrs['rate'])
            only_positive: Returns only the positives frequencies if true

        Returns:
            A `xarray.DataArray` with the corresponding amplitudes and frequencies
       """
        return misc.fft(self._obj, freq, only_positive)

    def detect_onset(
        self,
        threshold: Union[float, int],
        n_above: int = 1,
        n_below: int = 0,
        threshold2: int = None,
        n_above2: int = 1,
    ) -> np.array:
        """
        Detects onset based on amplitude threshold

        todo: example

        Arguments:
            threshold: minimum amplitude to detect
            n_above: minimum number of continuous samples >= `threshold` to detect
            n_below: minimum number of continuous samples below `threshold`
              that will be ignored in the detection of `x` >= `threshold`
            threshold2: minimum amplitude of `n_above2` values in `x` to detect
            n_above2: minimum number of samples >= `threshold2` to detect

        Note:
            You might have to tune the parameters according to the signal-to-noise
              characteristic of the data.

        Returns:
            inds: 1D array_like [indi, indf] containing initial and final indexes of the onset events
        """
        return misc.detect_onset(
            self._obj, threshold, n_above, n_below, threshold2, n_above2
        )

    def detect_outliers(self, threshold: int = 3) -> xr.DataArray:
        """
        Detects data points that are `threshold` times the standard deviation from the mean.

        todo: example

        Arguments:
            threshold: Multiple of standard deviation from which data is considered outlier

        Returns:
            A boolean `xarray.DataArray` containing the outliers
        """
        return misc.detect_outliers(self._obj, threshold)
