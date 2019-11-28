from typing import Union

import numpy as np
import xarray as xr

from motion import rototrans
from motion.processing import algebra
from motion.processing import filter
from motion.processing import interp
from motion.processing import misc


@xr.register_dataarray_accessor("meca")
class MecaDataArrayAccessor(object):
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
        return algebra.abs_(self._obj)

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
        return algebra.matmul(self._obj, other)

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
        return algebra.square(self._obj, *args, **kwargs)

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
        return algebra.sqrt(self._obj, *args, **kwargs)

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
        return algebra.norm(self._obj, dim, ord)

    def rms(self) -> xr.DataArray:
        """
        Get root-mean-square values.
        Returns
        -------
        A DataArray containing the root-mean-square of the matrix.
        """
        return algebra.rms(self._obj)

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
        return algebra.center(self._obj, mu)

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
        return algebra.normalize(self._obj, ref, scale)

    # interp ------------------------------------
    def time_normalize(
        self,
        time_vector: Union[xr.DataArray, np.array] = None,
        n_frames: int = 100,
        norm_time_frame: bool = False,
    ) -> xr.DataArray:
        """
        Time normalization used for temporal alignment of data.
        Parameters
        ----------
        time_vector :
            desired time vector (first to last time_frame with n_frames points by default). Optional
        n_frames :
            if time_vector is not specified, the length of the desired time vector. Optional
        norm_time_frame :
            Normalize the time_frame dimension from 0 to 100 if True
        Returns
        -------
        A time-normalized DataArray.
        """
        return interp.time_normalize(
            self._obj, time_vector, n_frames, norm_time_frame=norm_time_frame
        )

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
        return filter.low_pass(self._obj, freq, order, cutoff)

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
        return filter.high_pass(self._obj, freq, order, cutoff)

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
        return filter.band_stop(self._obj, freq, order, cutoff)

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
        return filter.band_pass(self._obj, freq, order, cutoff)

    # signal processing misc --------------------
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
        Detects onset in data based on amplitude threshold.
        Parameters
        ----------
        threshold : number
            minimum amplitude of `x` to detect.
        n_above : number, optional (default = 1)
            minimum number of continuous samples >= `threshold`
            to detect (but see the parameter `n_below`).
        n_below : number, optional (default = 0)
            minimum number of continuous samples below `threshold` that
            will be ignored in the detection of `x` >= `threshold`.
        threshold2 : number or None, optional (default = None)
            minimum amplitude of `n_above2` values in `x` to detect.
        n_above2 : number, optional (default = 1)
            minimum number of samples >= `threshold2` to detect.
        Returns
        -------
        inds : 1D array_like [indi, indf]
            initial and final indexes of the onset events.
        Notes
        -----
        You might have to tune the parameters according to the signal-to-noise
        characteristic of the data.
        """
        return misc.detect_onset(
            self._obj, threshold, n_above, n_below, threshold2, n_above2
        )

    def detect_outliers(self, threshold: int = 3) -> xr.DataArray:
        """
        Detects data that is `threshold` times the standard deviation from the mean.
        Parameters
        ----------
        threshold : int
            Multiple of standard deviation from which data is considered outlier
        Returns
        -------
        A boolean DataArray containing the outliers.
        """
        return misc.detect_outliers(self._obj, threshold)

    def get_euler_angles(self, angle_sequence: str) -> xr.DataArray:
        """
        Get euler angles with specified angle sequence
        Parameters
        ----------
        angle_sequence
            Euler sequence of angles. Valid values are all permutations of "xyz"
        Returns
        -------
        DataArray with the euler angles associated
        """
        return rototrans.get_euler_angles(self._obj, angle_sequence)
