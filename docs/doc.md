# Analogs
```python
Analogs(self, /, *args, **kwargs)
```

## from_csv
```python
Analogs.from_csv(filename: Union[str, pathlib.Path], usecols: Union[List[Union[str, int]], NoneType] = None, header: Union[int, NoneType] = None, first_row: int = 0, first_column: Union[str, int, NoneType] = None, time_column: Union[str, int, NoneType] = None, last_column_to_remove: Union[str, int, NoneType] = None, prefix_delimiter: Union[str, NoneType] = None, suffix_delimiter: Union[str, NoneType] = None, skiprows: Union[List[int], NoneType] = None, pandas_kwargs: Union[dict, NoneType] = None, attrs: Union[dict, NoneType] = None) -> xarray.core.dataarray.DataArray
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

## from_excel
```python
Analogs.from_excel(filename: Union[str, pathlib.Path], sheet_name: Union[str, int] = 0, usecols: Union[List[Union[str, int]], NoneType] = None, header: Union[int, NoneType] = None, first_row: int = 0, first_column: Union[str, int, NoneType] = None, time_column: Union[str, int, NoneType] = None, last_column_to_remove: Union[str, int, NoneType] = None, prefix_delimiter: Union[str, NoneType] = None, suffix_delimiter: Union[str, NoneType] = None, skiprows: Union[List[int], NoneType] = None, pandas_kwargs: Union[dict, NoneType] = None, attrs: Union[dict, NoneType] = None) -> xarray.core.dataarray.DataArray
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

## from_sto
```python
Analogs.from_sto(filename: Union[str, pathlib.Path], end_header: Union[bool, NoneType] = None, **kwargs)
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

## from_mot
```python
Analogs.from_mot(filename: Union[str, pathlib.Path], end_header: Union[bool, NoneType] = None, **kwargs)
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

## from_c3d
```python
Analogs.from_c3d(filename: Union[str, pathlib.Path], usecols: Union[List[Union[str, int]], NoneType] = None, prefix_delimiter: Union[str, NoneType] = None, suffix_delimiter: Union[str, NoneType] = None, attrs: Union[dict, NoneType] = None)
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

## reshape_flat_array
```python
Analogs.reshape_flat_array(array: Union[<built-in function array>, numpy.ndarray]) -> xarray.core.dataarray.DataArray
```

Takes a tabular numpy array (frames x N) and return a (N x frames) numpy array
Parameters
----------
array:
    A tabular array (frames x N) with N = 3 x marker

# Markers
```python
Markers(self, /, *args, **kwargs)
```

## from_csv
```python
Markers.from_csv(filename: Union[str, pathlib.Path], usecols: Union[List[Union[str, int]], NoneType] = None, header: Union[int, NoneType] = None, first_row: int = 0, first_column: Union[str, int, NoneType] = None, time_column: Union[str, int, NoneType] = None, last_column_to_remove: Union[str, int, NoneType] = None, prefix_delimiter: Union[str, NoneType] = None, suffix_delimiter: Union[str, NoneType] = None, skiprows: Union[List[int], NoneType] = None, pandas_kwargs: Union[dict, NoneType] = None, attrs: Union[dict, NoneType] = None) -> xarray.core.dataarray.DataArray
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

## from_excel
```python
Markers.from_excel(filename: Union[str, pathlib.Path], sheet_name: Union[str, int] = 0, usecols: Union[List[Union[str, int]], NoneType] = None, header: Union[int, NoneType] = None, first_row: int = 0, first_column: Union[str, int, NoneType] = None, time_column: Union[str, int, NoneType] = None, last_column_to_remove: Union[str, int, NoneType] = None, prefix_delimiter: Union[str, NoneType] = None, suffix_delimiter: Union[str, NoneType] = None, skiprows: Union[List[int], NoneType] = None, pandas_kwargs: Union[dict, NoneType] = None, attrs: Union[dict, NoneType] = None) -> xarray.core.dataarray.DataArray
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

## from_c3d
```python
Markers.from_c3d(filename: Union[str, pathlib.Path], usecols: Union[List[Union[str, int]], NoneType] = None, prefix_delimiter: Union[str, NoneType] = None, suffix_delimiter: Union[str, NoneType] = None, attrs: Union[dict, NoneType] = None)
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

## from_trc
```python
Markers.from_trc(filename: Union[str, pathlib.Path], **kwargs)
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

## reshape_flat_array
```python
Markers.reshape_flat_array(array: Union[<built-in function array>, numpy.ndarray]) -> xarray.core.dataarray.DataArray
```

Takes a tabular numpy array (frames x [N * 3]) and return a (3 x N x frames) numpy array
Parameters
----------
array:
    A tabular array (frames x N) with N = 3 x marker

