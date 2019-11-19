from typing import Union

import numpy as np
import xarray as xr

from motion.processing.algebra import (
    _abs,
    matmul,
    square,
    sqrt,
    rms,
    center,
    normalize,
    norm,
)
from motion.processing.filter import low_pass, high_pass, band_pass, band_stop
from motion.processing.interp import time_normalization
from motion.processing.misc import fft


@xr.register_dataarray_accessor("proc")
class _ProcAccessor(object):
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    # algebra -----------------------------------
    def abs(self) -> xr.DataArray:
        """
        Calculate the absolute value element-wise.
        Parameters
        ----------
        self
            input DataArray
        Returns
        -------
        A DataArray containing the absolute value of each element in array.
        """
        return _abs(self._obj)

    def matmul(self, other: xr.DataArray) -> xr.DataArray:
        """
        Matrix product of two arrays.
        Parameters
        ----------
        self
            input DataArray
        other
            second array to multiply
        Returns
        -------
        A DataArray containing the matrix product of the two arrays.
        """
        return matmul(self._obj, other)

    def square(self, *args, **kwargs) -> xr.DataArray:
        """
        Return the element-wise square of the input.
        Parameters
        ----------
        self
            input DataArray
        *args
            For other positional arguments, see the numpy docs
        **kwargs
            For other keyword-only arguments, see the numpy docs
        Returns
        -------
        A DataArray containing the matrix squared.
        """
        return square(self._obj, *args, **kwargs)

    def sqrt(self, *args, **kwargs) -> xr.DataArray:
        """
        Return the non-negative square-root of an array, element-wise.
        Parameters
        ----------
        self
            input DataArray
        *args
            For other positional arguments, see the numpy docs
        **kwargs
            For other keyword-only arguments, see the numpy docs
        Returns
        -------
        A DataArray containing the square root of the matrix.
        """
        return sqrt(self._obj, *args, **kwargs)

    def norm(self, dim: Union[str, list], ord: int = None) -> xr.DataArray:
        """
        Return the norm of an array.
        Parameters
        -------
        dim:
            Name(s) of the data dimension(s)
        ord:
            Order of the norm (see table under Notes). inf means numpy’s inf object
        Returns
        -------
        A DataArray containing the norm of the matrix.
        """
        return norm(self._obj, dim, ord)

    def rms(self) -> xr.DataArray:
        """
        Get root-mean-square values.
        Returns
        -------
        A DataArray containing the root-mean-square of the matrix.
        """
        return rms(self._obj)

    def center(
        self, mu: Union[xr.DataArray, np.array, float, int] = None
    ) -> xr.DataArray:
        """
        Center a DataArray (i.e., subtract the mean).
        Parameters
        ----------
        mu :
            mean of the signal to subtract, optional.
            If not provided, motion takes the mean along the time_frame axis
        Returns
        -------
        A DataArray containing the root-mean-square of the matrix.
        """
        return center(self._obj, mu)

    def normalize(
        self,
        ref: Union[xr.DataArray, np.array, float, int] = None,
        scale: Union[int, float] = 100,
    ) -> xr.DataArray:
        """
        Normalize a signal against `ref` on a scale of `scale`.
        Ref is set to DataArray's max by default.
        Scale is set to 100 by default.
        Parameters
        ----------
        ref :
            Reference value. Could have multiple dimensions. Optional
        scale
            Scale on which to express array. Optional
        Returns
        -------
        A normalized DataArray.
        """
        return normalize(self._obj, ref, scale)

    # interp ------------------------------------
    def time_normalization(
        self, time_vector: Union[xr.DataArray, np.array] = None, n_frames: int = 100
    ) -> xr.DataArray:
        """
        Time normalization used for temporal alignment of data.
        Parameters
        ----------
        time_vector :
            desired time vector (first to last time_frame with n_frames points by default). Optional
        n_frames :
            if time_vector is not specified, the length of the desired time vector. Optional
        Returns
        -------
        A time-normalized DataArray.
        """
        return time_normalization(self._obj, time_vector, n_frames)

    # filter ------------------------------------
    def low_pass(
        self, freq: Union[int, float], order: int, cutoff: Union[int, float, np.array]
    ):
        """
        Low-pass Butterworth filter.
        Parameters
        ----------
        freq:
            Sampling frequency
        order:
            Order of the filter
        cutoff:
            Cut-off frequency
        Returns
        -------
        A low-passed DataArray.
        """
        return low_pass(self._obj, freq, order, cutoff)

    def high_pass(
        self, freq: Union[int, float], order: int, cutoff: Union[int, float, np.array]
    ):
        """
        High-pass Butterworth filter.
        Parameters
        ----------
        freq:
            Sampling frequency
        order:
            Order of the filter
        cutoff:
            Cut-off frequency
        Returns
        -------
        A high-passed DataArray.
        """
        return high_pass(self._obj, freq, order, cutoff)

    def band_stop(
        self, freq: Union[int, float], order: int, cutoff: Union[list, tuple, np.array]
    ):
        """
        Band-stop Butterworth filter.
        Parameters
        ----------
        freq:
            Sampling frequency
        order:
            Order of the filter
        cutoff:
            Cut-off frequency [lower, upper]
        Returns
        -------
        A band-stopped DataArray.
        """
        return band_stop(self._obj, freq, order, cutoff)

    def band_pass(
        self, freq: Union[int, float], order: int, cutoff: Union[list, tuple, np.array]
    ):
        """
        Band-pass Butterworth filter.
        Parameters
        ----------
        freq:
            Sampling frequency
        order:
            Order of the filter
        cutoff:
            Cut-off frequency [lower, upper]
        Returns
        -------
        A band-passed DataArray.
        """
        return band_pass(self._obj, freq, order, cutoff)

    # misc --------------------------------------
    def fft(self, freq: Union[int, float], only_positive: bool = True) -> xr.DataArray:
        """
       Performs a discrete Fourier Transform and return a DataArray with the corresponding amplitudes and frequencies.
       Parameters
       ----------
       freq :
           Sampling frequency
       only_positive
           Returns only the positives frequencies if true (True by default)
       Returns
       -------
       A DataArray with the corresponding amplitudes and frequencies.
       """
        return fft(self._obj, freq, only_positive)

    # def detect_outliers(self, threshold: int = 3) -> xr.DataArray:
    #     """
    #     Detects data that is `threshold` times the standard deviation from the mean.
    #     Parameters
    #     ----------
    #     threshold : int
    #         Multiple of standard deviation from which data is considered outlier
    #     Returns
    #     -------
    #     A boolean DataArray containing the outliers.
    #     """
    #     return detect_outliers(self._obj, threshold)
