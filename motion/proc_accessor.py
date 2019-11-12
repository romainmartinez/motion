import xarray as xr

from motion.processing.algebra import _abs, matmul, square, sqrt, rms


@xr.register_dataarray_accessor("proc")
class _ProcAccessor(object):
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

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

    def rms(self) -> xr.DataArray:
        """
        Get root-mean-square values

        Returns
        -------
        A DataArray containing the root-mean-square of the matrix.
        """
        return rms(self._obj)
