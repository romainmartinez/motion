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

    # def __new__(cls, *args, **kwargs):
    #     print("debug")
    #     kwargs.keys()
    #     if "time_frame" not in kwargs["dims"]:
    #         raise ValueError(f"time_frame not in dims. You specified {kwargs['dims']}")
    #     if kwargs["data"].ndim != 3:
    #         raise ValueError(
    #             f'Markers data must have three dimensions. Your data have {kwargs["data"].ndim}'
    #         )
    #     else:
    #         pass
    #     return super(Markers, cls).__new__(cls)
