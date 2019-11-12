import xarray as xr


class Analogs(xr.DataArray):
    __slots__ = []

    def __init__(self, *args, **kwargs):
        super(Analogs, self).__init__(*args, **kwargs)

        # TODO verifications


class Markers(xr.DataArray):
    __slots__ = []

    def __init__(self, *args, **kwargs):
        super(Markers, self).__init__(*args, **kwargs)

        # TODO verifications
