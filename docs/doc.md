# Table of Contents

  * [motion](#motion)
  * [motion.dataarray\_accessor](#motion.dataarray_accessor)
    * [MecaDataArrayAccessor](#motion.dataarray_accessor.MecaDataArrayAccessor)
  * [motion.markers](#motion.markers)
    * [Markers](#motion.markers.Markers)
  * [motion.rototrans](#motion.rototrans)
    * [Rototrans](#motion.rototrans.Rototrans)
  * [motion.angles](#motion.angles)
    * [Angles](#motion.angles.Angles)
  * [motion.analogs](#motion.analogs)
    * [Analogs](#motion.analogs.Analogs)

# `motion`


# `motion.dataarray_accessor`


## `MecaDataArrayAccessor` Objects

```python
def __init__(self, xarray_obj: xr.DataArray)
```


### `MecaDataArrayAccessor.__init__()`

```python
def __init__(self, xarray_obj: xr.DataArray)
```


### `MecaDataArrayAccessor.to_matlab()`

```python
def to_matlab(self, filename: Union[str, Path])
```

Write a matlab file from a xarray.DataArray
Parameters
----------
filename
    File path

### `MecaDataArrayAccessor.to_csv()`

```python
def to_csv(self, filename: Union[str, Path], wide: Optional[bool] = True)
```

Write a csv file from a xarray.DataArray
Parameters
----------
filename
    File path
wide
    True if you want a wide dataframe (one column for each channel).
    Set to false if you want a tidy dataframe.

### `MecaDataArrayAccessor.to_wide_dataframe()`

```python
def to_wide_dataframe(self)
```

Returns
-------
A wide pandas DataFrame (one column by channel).
Works only for 2 and 3-dimensional arrays.
If you want a tidy dataframe type: `array.to_series()`, or `array.to_dataframe()`.

### `MecaDataArrayAccessor.abs()`

```python
def abs(self) -> xr.DataArray
```

Calculate the absolute value element-wise.
Parameters
----------
self
    input DataArray
Returns
-------
A DataArray containing the absolute value of each element in array.

### `MecaDataArrayAccessor.matmul()`

```python
def matmul(self, other: xr.DataArray) -> xr.DataArray
```

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

### `MecaDataArrayAccessor.square()`

```python
def square(self, args, *,, ,, kwargs) -> xr.DataArray
```

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

### `MecaDataArrayAccessor.sqrt()`

```python
def sqrt(self, args, *,, ,, kwargs) -> xr.DataArray
```

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

### `MecaDataArrayAccessor.norm()`

```python
def norm(self, dim: Union[str, list], ord: int = None) -> xr.DataArray
```

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

### `MecaDataArrayAccessor.rms()`

```python
def rms(self) -> xr.DataArray
```

Get root-mean-square values.
Returns
-------
A DataArray containing the root-mean-square of the matrix.

### `MecaDataArrayAccessor.center()`

```python
def center(self, mu: Union[xr.DataArray, np.array, float, int] = None) -> xr.DataArray
```

Center a DataArray (i.e., subtract the mean).
Parameters
----------
mu :
    mean of the signal to subtract, optional.
    If not provided, motion takes the mean along the time_frame axis
Returns
-------
A DataArray containing the root-mean-square of the matrix.

### `MecaDataArrayAccessor.normalize()`

```python
def normalize(self, ref: Union[xr.DataArray, np.array, float, int] = None, scale: Union[int, float] = 100) -> xr.DataArray
```

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

### `MecaDataArrayAccessor.time_normalize()`

```python
def time_normalize(self, time_vector: Union[xr.DataArray, np.array] = None, n_frames: int = 100, norm_time_frame: bool = False) -> xr.DataArray
```

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

### `MecaDataArrayAccessor.low_pass()`

```python
def low_pass(self, freq: Union[int, float], order: int, cutoff: Union[int, float, np.array])
```

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

### `MecaDataArrayAccessor.high_pass()`

```python
def high_pass(self, freq: Union[int, float], order: int, cutoff: Union[int, float, np.array])
```

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

### `MecaDataArrayAccessor.band_stop()`

```python
def band_stop(self, freq: Union[int, float], order: int, cutoff: Union[list, tuple, np.array])
```

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

### `MecaDataArrayAccessor.band_pass()`

```python
def band_pass(self, freq: Union[int, float], order: int, cutoff: Union[list, tuple, np.array])
```

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

### `MecaDataArrayAccessor.fft()`

```python
def fft(self, freq: Union[int, float], only_positive: bool = True) -> xr.DataArray
```

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

### `MecaDataArrayAccessor.detect_onset()`

```python
def detect_onset(self, threshold: Union[float, int], n_above: int = 1, n_below: int = 0, threshold2: int = None, n_above2: int = 1) -> np.array
```

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

### `MecaDataArrayAccessor.detect_outliers()`

```python
def detect_outliers(self, threshold: int = 3) -> xr.DataArray
```

Detects data that is `threshold` times the standard deviation from the mean.
Parameters
----------
threshold : int
    Multiple of standard deviation from which data is considered outlier
Returns
-------
A boolean DataArray containing the outliers.

# `motion.markers`


## `Markers` Objects


### `Markers.__new__()`

```python
def __new__(cls, data: Optional[Union[np.array, np.ndarray, xr.DataArray, list]] = None, channels: Optional[list] = None, time_frames: Optional[Union[np.array, list, pd.Series]] = None, args, *,, ,, kwargs) -> xr.DataArray
```

Markers array with `axis`, `channel` and `time_frame` dimensions
Parameters
----------
data
    Array to be passed to xarray.DataArray
channels
    Channel names
args
    Positional argument(s) to be passed to xarray.DataArray
kwargs
    Keyword argument(s) to be passed to xarray.DataArray
Returns
-------
Markers xarray.DataArray

### `Markers.from_csv()`

```python
@classmethod
def from_csv(cls, filename: Union[str, Path], usecols: Optional[List[Union[str, int]]] = None, header: Optional[int] = None, first_row: int = 0, first_column: Optional[Union[str, int]] = None, time_column: Optional[Union[str, int]] = None, last_column_to_remove: Optional[Union[str, int]] = None, prefix_delimiter: Optional[str] = None, suffix_delimiter: Optional[str] = None, skiprows: Optional[List[int]] = None, pandas_kwargs: Optional[dict] = None, attrs: Optional[dict] = None) -> xr.DataArray
```

Read csv data and convert to Markers DataArray
Parameters
----------
filename:
    Any valid string path
usecols:
    All elements must either be positional or strings that correspond to column names.
    For example, a valid list-like usecols parameter would be [0, 1, 2] or ['foo', 'bar', 'baz'].
header:
    Row of the header (0-indexed)
first_row:
    First row of the data (0-indexed)
first_column:
    First column of the data (0-indexed)
time_column:
    Column of the time column. If None, we associate the index
last_column_to_remove:
    If for some reason the csv reads extra columns, how many should be ignored
prefix_delimiter:
    Delimiter that split each column name by its prefix (we keep only the column name)
suffix_delimiter:
    Delimiter that split each column name by its suffix (we keep only the column name)
skiprows:
    Line numbers to skip (0-indexed)
pandas_kwargs:
    Keyword arguments to be passed to pandas.read_csv
attrs:
    attrs to be passed to xr.DataArray. If attrs['rate'] is provided, compute the time_frame accordingly

### `Markers.from_excel()`

```python
@classmethod
def from_excel(cls, filename: Union[str, Path], sheet_name: Union[str, int] = 0, usecols: Optional[List[Union[str, int]]] = None, header: Optional[int] = None, first_row: int = 0, first_column: Optional[Union[str, int]] = None, time_column: Optional[Union[str, int]] = None, last_column_to_remove: Optional[Union[str, int]] = None, prefix_delimiter: Optional[str] = None, suffix_delimiter: Optional[str] = None, skiprows: Optional[List[int]] = None, pandas_kwargs: Optional[dict] = None, attrs: Optional[dict] = None) -> xr.DataArray
```

Read excel data and convert to Markers DataArray
Parameters
----------
filename:
    Any valid string path
sheet_name:
    Strings are used for sheet names. Integers are used in zero-indexed sheet positions
usecols:
    All elements must either be positional or strings that correspond to column names.
    For example, a valid list-like usecols parameter would be [0, 1, 2] or ['foo', 'bar', 'baz'].
header:
    Row of the header (0-indexed)
first_row:
    First row of the data (0-indexed)
first_column:
    First column of the data (0-indexed)
time_column:
    Column of the time column. If None, we associate the index
last_column_to_remove:
    If for some reason the csv reads extra columns, how many should be ignored
prefix_delimiter:
    Delimiter that split each column name by its prefix (we keep only the column name)
suffix_delimiter:
    Delimiter that split each column name by its suffix (we keep only the column name)
skiprows:
    Line numbers to skip (0-indexed)
pandas_kwargs:
    Keyword arguments to be passed to pandas.read_csv
attrs:
    attrs to be passed to xr.DataArray. If attrs['rate'] is provided, compute the time_frame accordingly

### `Markers.from_c3d()`

```python
@classmethod
def from_c3d(cls, filename: Union[str, Path], usecols: Optional[List[Union[str, int]]] = None, prefix_delimiter: Optional[str] = None, suffix_delimiter: Optional[str] = None, attrs: Optional[dict] = None)
```

Read c3d data and convert to Markers DataArray
Parameters
----------
filename
    Any valid string path
usecols
    All elements must either be positional or strings that correspond to column names.
    For example, a valid list-like usecols parameter would be [0, 1, 2] or ['foo', 'bar', 'baz'].
prefix_delimiter:
    Delimiter that split each column name by its prefix (we keep only the column name)
suffix_delimiter:
    Delimiter that split each column name by its suffix (we keep only the column name)
attrs
    attrs to be passed to xr.DataArray
Returns
-------
Markers xarray.DataArray

### `Markers.from_trc()`

```python
@classmethod
def from_trc(cls, filename: Union[str, Path], kwargs)
```

Read a TRC file and return a Markers DataArray
Parameters
----------
filename
    Any valid string path
kwargs
    Keyword arguments to be passed to `from_csv`
Returns
-------
Markers xarray.DataArray

### `Markers.reshape_flat_array()`

```python
@staticmethod
def reshape_flat_array(array: Union[np.array, np.ndarray]) -> xr.DataArray
```

Takes a tabular numpy array (frames x [N * 3]) and return a (3 x N x frames) numpy array
Parameters
----------
array:
    A tabular array (frames x N) with N = 3 x marker

### `Markers.get_requested_channels_from_pandas()`

```python
@staticmethod
def get_requested_channels_from_pandas(columns, header, usecols, prefix_delimiter: str, suffix_delimiter: str) -> Tuple[Optional[list], Optional[list]]
```


# `motion.rototrans`


## `Rototrans` Objects


### `Rototrans.__new__()`

```python
def __new__(cls, data: Optional[Union[np.array, np.ndarray, xr.DataArray, list]] = None, time_frames: Optional[Union[np.array, list, pd.Series]] = None, args, *,, ,, kwargs) -> xr.DataArray
```

Rototrans array with `axis`, `channel` and `time_frame` dimensions
Parameters
----------
data
    Array to be passed to xarray.DataArray
args
    Positional argument(s) to be passed to xarray.DataArray
kwargs
    Keyword argument(s) to be passed to xarray.DataArray
Returns
-------
Rototrans xarray.DataArray

### `Rototrans.from_euler_angles()`

```python
@classmethod
def from_euler_angles(cls, angles: Optional[xr.DataArray] = None, angle_sequence: Optional[str] = None, translations: Optional[xr.DataArray] = None)
```

Get rototrans DataArray from angles/translations
Parameters
----------
angles:
    Euler angles of the rototranslation
angle_sequence:
    Euler sequence of angles; valid values are all permutation of axes (e.g. "xyz", "yzx", ...)
translations
    Translation part of the Rototrans matrix

# `motion.angles`


## `Angles` Objects


### `Angles.__new__()`

```python
def __new__(cls, data: Optional[Union[np.array, np.ndarray, xr.DataArray, list]] = None, time_frames: Optional[Union[np.array, list, pd.Series]] = None, args, *,, ,, kwargs) -> xr.DataArray
```

Angles array with `axis`, `channel` and `time_frame` dimensions
Parameters
----------
data
    Array to be passed to xarray.DataArray
args
    Positional argument(s) to be passed to xarray.DataArray
kwargs
    Keyword argument(s) to be passed to xarray.DataArray
Returns
-------
Angles xarray.DataArray

### `Angles.from_rototrans()`

```python
@classmethod
def from_rototrans(cls, rototrans, angle_sequence)
```

Get Angles DataArray from rototrans and specified angle sequence
Parameters
----------
rototrans
    Rototrans created with motion.create_rototrans()
angle_sequence
    Euler sequence of angles. Valid values are all permutations of "xyz"
Returns
-------
Angles DataArray with the euler angles associated

# `motion.analogs`


## `Analogs` Objects


### `Analogs.__new__()`

```python
def __new__(cls, data: Optional[Union[np.array, np.ndarray, xr.DataArray, list]] = None, channels: Optional[list] = None, time_frames: Optional[Union[np.array, list, pd.Series]] = None, args, *,, ,, kwargs) -> xr.DataArray
```

Analogs array with `channel` and `time_frame` dimensions
Parameters
----------
data
    Array to be passed to xarray.DataArray
args
    Positional argument(s) to be passed to xarray.DataArray
kwargs
    Keyword argument(s) to be passed to xarray.DataArray
Returns
-------
Analogs xarray.DataArray

### `Analogs.from_csv()`

```python
@classmethod
def from_csv(cls, filename: Union[str, Path], usecols: Optional[List[Union[str, int]]] = None, header: Optional[int] = None, first_row: int = 0, first_column: Optional[Union[str, int]] = None, time_column: Optional[Union[str, int]] = None, last_column_to_remove: Optional[Union[str, int]] = None, prefix_delimiter: Optional[str] = None, suffix_delimiter: Optional[str] = None, skiprows: Optional[List[int]] = None, pandas_kwargs: Optional[dict] = None, attrs: Optional[dict] = None) -> xr.DataArray
```

Read csv data and convert to Analogs DataArray
Parameters
----------
filename:
    Any valid string path
usecols:
    All elements must either be positional or strings that correspond to column names.
    For example, a valid list-like usecols parameter would be [0, 1, 2] or ['foo', 'bar', 'baz'].
header:
    Row of the header (0-indexed)
first_row:
    First row of the data (0-indexed)
first_column:
    First column of the data (0-indexed)
time_column:
    Column of the time column. If None, we associate the index
last_column_to_remove:
    If for some reason the csv reads extra columns, how many should be ignored
prefix_delimiter:
    Delimiter that split each column name by its prefix (we keep only the column name)
suffix_delimiter:
    Delimiter that split each column name by its suffix (we keep only the column name)
skiprows:
    Line numbers to skip (0-indexed)
pandas_kwargs:
    Keyword arguments to be passed to pandas.read_csv
attrs:
    attrs to be passed to xr.DataArray. If attrs['rate'] is provided, compute the time_frame accordingly

### `Analogs.from_excel()`

```python
@classmethod
def from_excel(cls, filename: Union[str, Path], sheet_name: Union[str, int] = 0, usecols: Optional[List[Union[str, int]]] = None, header: Optional[int] = None, first_row: int = 0, first_column: Optional[Union[str, int]] = None, time_column: Optional[Union[str, int]] = None, last_column_to_remove: Optional[Union[str, int]] = None, prefix_delimiter: Optional[str] = None, suffix_delimiter: Optional[str] = None, skiprows: Optional[List[int]] = None, pandas_kwargs: Optional[dict] = None, attrs: Optional[dict] = None) -> xr.DataArray
```

Read excel data and convert to Analogs DataArray
Parameters
----------
filename:
    Any valid string path
sheet_name:
    Strings are used for sheet names. Integers are used in zero-indexed sheet positions
usecols:
    All elements must either be positional or strings that correspond to column names.
    For example, a valid list-like usecols parameter would be [0, 1, 2] or ['foo', 'bar', 'baz'].
header:
    Row of the header (0-indexed)
first_row:
    First row of the data (0-indexed)
first_column:
    First column of the data (0-indexed)
time_column:
    Column of the time column. If None, we associate the index
last_column_to_remove:
    If for some reason the csv reads extra columns, how many should be ignored
prefix_delimiter:
    Delimiter that split each column name by its prefix (we keep only the column name)
suffix_delimiter:
    Delimiter that split each column name by its suffix (we keep only the column name)
skiprows:
    Line numbers to skip (0-indexed)
pandas_kwargs:
    Keyword arguments to be passed to pandas.read_csv
attrs:
    attrs to be passed to xr.DataArray. If attrs['rate'] is provided, compute the time_frame accordingly

### `Analogs.from_sto()`

```python
@classmethod
def from_sto(cls, filename: Union[str, Path], end_header: Optional[bool] = None, kwargs)
```

Read a STO file and return an Analogs DataArray
Parameters
----------
filename
    Any valid string path
end_header
    Index where `endheader` appears (0 indexed). If not provided, the index is automatically determined.
kwargs
    Keyword arguments to be passed to `from_csv`
Returns
-------
Analogs xarray.DataArray

### `Analogs.from_mot()`

```python
@classmethod
def from_mot(cls, filename: Union[str, Path], end_header: Optional[bool] = None, kwargs)
```

Read a MOT file and return an Analogs DataArray
Parameters
----------
filename
    Any valid string path
end_header
    Index where `endheader` appears (0 indexed). If not provided, the index is automatically determined.
kwargs
    Keyword arguments to be passed to `from_csv`
Returns
-------
Analogs xarray.DataArray

### `Analogs.from_c3d()`

```python
@classmethod
def from_c3d(cls, filename: Union[str, Path], usecols: Optional[List[Union[str, int]]] = None, prefix_delimiter: Optional[str] = None, suffix_delimiter: Optional[str] = None, attrs: Optional[dict] = None)
```

Read c3d data and convert to Analogs DataArray
Parameters
----------
filename
    Any valid string path
usecols
    All elements must either be positional or strings that correspond to column names.
    For example, a valid list-like usecols parameter would be [0, 1, 2] or ['foo', 'bar', 'baz'].
prefix_delimiter:
    Delimiter that split each column name by its prefix (we keep only the column name)
suffix_delimiter:
    Delimiter that split each column name by its suffix (we keep only the column name)
attrs
    attrs to be passed to xr.DataArray
Returns
-------
Analogs xarray.DataArray

### `Analogs.reshape_flat_array()`

```python
@staticmethod
def reshape_flat_array(array: Union[np.array, np.ndarray]) -> xr.DataArray
```

Takes a tabular numpy array (frames x N) and return a (N x frames) numpy array
Parameters
----------
array:
    A tabular array (frames x N) with N = 3 x marker

### `Analogs.get_requested_channels_from_pandas()`

```python
@staticmethod
def get_requested_channels_from_pandas(columns, header, usecols, prefix_delimiter: str, suffix_delimiter: str) -> Tuple[Optional[list], Optional[list]]
```


