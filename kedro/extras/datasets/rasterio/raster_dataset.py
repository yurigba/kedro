from ast import Mult
from copy import deepcopy
import copy

import warnings
from kedro.io.core import (
    _CONSISTENCY_WARNING,
    AbstractVersionedDataSet,
    DataSetError,
    Version,
    get_filepath_str,
    get_protocol_and_path
)

import fsspec
import rasterio
from rasterio import windows
from rasterio.enums import Resampling
from rasterio.mask import geometry_mask
from pathlib import PurePath

import numpy as np

import geopandas as gpd
from geopandas.geodataframe import GeoDataFrame
from geopandas.geoseries import GeoSeries
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely import wkt
from pyproj.crs import CRS

from typing import Dict, Any, Optional, TypeVar, Union

_DI = TypeVar("_DI")
_DO = TypeVar("_DO")


def _create_interior_mask_from_raster_and_polygon(raster, geometries):
    
    """
    CREATE INTERIOR MASK - Creates a mask corresponding to the interior of a geometry
    based on a given raster.
    
    This function builds a bool array where is True whenever the pixel is in the interior
    of a geometry, and False in the exterior.
    
    For this to properly work, you need the metadata of a raster, and a polygon-like object
    
    """
    
    bool_mask = dict()
        
    meta = raster["meta"]
    out_shape = (meta["height"], meta["width"])
        
    geometries = geometries.to_crs(meta["crs"])
    
    bool_mask["array"] = (
        geometry_mask(
            geometries,
            out_shape,
            meta["transform"]
            ).astype(bool)
    )
    bool_mask["array"] = (np.logical_not(np.expand_dims(bool_mask["array"], axis=0))).astype("uint8")
    meta.update({"count": 1, "dtype": "uint8", "nodata": 0})
    bool_mask["meta"] = meta
    bool_mask["banda"] = "interior_mask"
    
    return bool_mask
    

class GDALRasterDataSet(AbstractVersionedDataSet):
    
    
    """
    
    
    """
    
    DEFAULT_LOAD_ARGS = {}  # type: Dict[str, Any]
    DEFAULT_SAVE_ARGS = {}  # type: Dict[str, Any]
    
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
        version: Version = None,
        credentials: Dict[str,Any] = None,
        fs_args: Dict[str, Any] = None,
    ) -> None:

        _fs_args = copy.deepcopy(fs_args) or {}
        _fs_open_args_load = _fs_args.pop("open_args_load", {})
        _fs_open_args_save = _fs_args.pop("open_args_save", {})
        _credentials = copy.deepcopy(credentials) or {}
        protocol, path = get_protocol_and_path(filepath, version)
        self._protocol = protocol
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._fs = fsspec.filesystem(self._protocol, **_credentials, **_fs_args)

        super().__init__(
            filepath=PurePath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )
        
        self._load_args = copy.deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)
            
        _fs_open_args_save.setdefault("mode", "wb")
        self._fs_open_args_load = _fs_open_args_load
        self._fs_open_args_save = _fs_open_args_save
        
    def load(self, **kwargs) -> _DO:
        """Loads data by delegation to the provided load method.
        
        This is a modified version that accepts kwargs on the load method,
        such that it is possible to pass a polygon to the method and get the
        crop of the specified polygon region.

        Returns:
            Data returned by the provided load method.

        Raises:
            DataSetError: When underlying load method raises error.

        """

        self._logger.debug("Loading %s", str(self))

        try:
            return self._load(**kwargs)
        except DataSetError:
            raise
        except Exception as exc:
            # This exception handling is by design as the composed data sets
            # can throw any type of exception.
            message = "Failed while loading data from data set {}.\n{}".format(
                str(self), str(exc)
            )
            raise DataSetError(message) from exc
        
    def save(self, data: _DI, **kwargs) -> None:
        
        """
        The base save class from AbstractVersionedDataSet, but with
        the possibility of using **kwargs to save
        """
        
        self._version_cache.clear()
        save_version = self.resolve_save_version()  # Make sure last save version is set
        try:
            
            # The base save method from AbstractDataSet, but with kwargs
            
            if data is None:
                raise DataSetError("Saving 'None' to a 'DataSet' is not allowed")

            try:
                self._logger.debug("Saving %s", str(self))
                self._save(data, **kwargs)
            except DataSetError:
                raise
            except (FileNotFoundError, NotADirectoryError):
                raise
            except Exception as exc:
                message = f"Failed while saving data to data set {str(self)}.\n{str(exc)}"
                raise DataSetError(message) from exc
            
        except (FileNotFoundError, NotADirectoryError) as err:
            # FileNotFoundError raised in Win, NotADirectoryError raised in Unix
            _default_version = "YYYY-MM-DDThh.mm.ss.sssZ"
            raise DataSetError(
                f"Cannot save versioned dataset '{self._filepath.name}' to "
                f"'{self._filepath.parent.as_posix()}' because a file with the same "
                f"name already exists in the directory. This is likely because "
                f"versioning was enabled on a dataset already saved previously. Either "
                f"remove '{self._filepath.name}' from the directory or manually "
                f"convert it into a versioned dataset by placing it in a versioned "
                f"directory (e.g. with default versioning format "
                f"'{self._filepath.as_posix()}/{_default_version}/{self._filepath.name}"
                f"')."
            ) from err

        load_version = self.resolve_load_version()
        if load_version != save_version:
            warnings.warn(
                _CONSISTENCY_WARNING.format(save_version, load_version, str(self))
            )
    
    def _load(
        self,
        geometry: Optional[Union[GeoDataFrame, GeoSeries, Polygon, MultiPolygon, str]] = None,
        geometry_crs: Optional[str] = None,
        scale_factor = 1,
        resampling = Resampling.nearest,
        envelope: bool = False
        ) -> dict:
        
        """
        Loads the raster into memory by passing a geometry object 
        (or None for the whole raster)
        
        Describe parameters for _load here
        """

        out_raster = dict.fromkeys(["array", "meta"])
        
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        
        # fsspec conflicts with GDAL virtual datasets
        
        # class WrapperContext:

        #     def __init__(self, filepath):
        #         if filepath.parts[0] == 's3:':
        #             vsipath = "/vsis3/" + "/".join(filepath.parts[1:])
        #             self.real_context = rasterio.open(vsipath)
        #         else:
        #             self.real_context = ContextB()

        #     def __enter__(self):
        #         return self.real_context.__enter__()

        #     def __exit__(self):
        #         return self.real_context.__exit__()
        
        if self._protocol == 's3':
            
            load_path = "/vsis3/"+load_path
            
        # with self._fs.open(load_path, **self._fs_open_args_load) as fs_file:
        with rasterio.open(load_path) as raster_src:
            
            # When geometry is None, read the whole raster
            # TODO: Load test case #1
            if geometry is None:
                
                out_raster["meta"] = raster_src.meta
                out_raster["array"] = raster_src.read(
                    out_shape=( # TODO: test if shapes are compatible after resampled read
                        raster_src.count,
                        int(raster_src.height * scale_factor),
                        int(raster_src.width * scale_factor)
                        ),
                    resampling=resampling
                    )
                # Keep width, heigth and transform compatible with resampling
                out_raster["meta"]["transform"] = (
                    raster_src.transform * raster_src.transform.scale(
                        (raster_src.width / out_raster["array"].shape[-1]),
                        (raster_src.height / out_raster["array"].shape[-2])
                        )
                )
                out_raster["meta"].update({
                    "width": out_raster["meta"]["width"]*scale_factor,
                    "height": out_raster["meta"]["height"]*scale_factor
                    })
                return out_raster
            
            # You can pass a GeoDataFrame of a GeoSeries as a geometry
            elif isinstance(geometry, (GeoDataFrame, GeoSeries)):
                
                assert len(geometry) == 1, "GeoDataFrame and GeoSeries must be made of a single geometry"
                # Checks if both geometry_crs and geometry.crs are None
                # If True, then it will assign the crs of the raster
                if (geometry.crs is None) and (geometry_crs is None):
                    # TODO: Load test case #2, #3
                    geometry = geometry.set_crs(raster_src.crs)
                    
                elif (geometry.crs is None) and (geometry_crs is not None):
                    # TODO: Load test case #4, #5
                    geometry = geometry.set_crs(geometry_crs)
                    geometry = geometry.to_crs(raster_src.crs)
                
                elif (geometry.crs is not None) and (geometry_crs is None):
                    # TODO: Load test case #6, #7
                    geometry = geometry.to_crs(raster_src.crs)
                    
                elif (geometry.crs is not None) and (geometry_crs is not None):
                    # TODO: Load test case #8, #9
                    raise ValueError("If geometry.crs is not None, geometry_crs should be None")
                
                if isinstance(geometry, GeoDataFrame):
                    # Converting to a GeoSeries
                    geometry = geometry["geometry"]
                
            elif isinstance(geometry, str): 
                # Checks if geometry_crs is None
                # If True, then it will assign the crs of the raster
                geometry = wkt.loads(geometry)
                if geometry_crs is None:
                    # TODO: Load test case #10
                    geometry = GeoSeries([geometry], crs = raster_src.crs)
                    
                elif geometry_crs is not None:
                    # TODO: Load test case #11
                    geometry = GeoSeries([geometry], crs = geometry_crs)
                    geometry = geometry.to_crs(raster_src.crs)
                
            # You can pass a shapely.Polygon or MultiPolygon
            elif isinstance(geometry, (Polygon, MultiPolygon)):
                # Checks if geometry_crs is None
                # If True, then it will assign the crs of the raster
                if geometry_crs is None:
                    # TODO: Load test case #12
                    geometry = GeoSeries([geometry], crs = raster_src.crs)
                    
                elif geometry_crs is not None:
                    # TODO: Load test case #13
                    geometry = GeoSeries([geometry], crs = geometry_crs)
                    geometry = geometry.to_crs(raster_src.crs)
            
            if scale_factor != 1:
                # TODO: fix this implementation
                print("WARN - scale_factor is ignored when a geometry argument is passed (NOT IMPLEMENTED)")
                
            # Assumes that geometry and src are same CRS
            bounds = geometry.iloc[0].bounds
            crop_window = windows.from_bounds(*bounds, transform = raster_src.transform)
            
            # Gets the windows raster and the corresponding transform
            
            out_image = raster_src.read(window=crop_window)
            out_transform = raster_src.window_transform(crop_window)
            
            # Updates metadata
            
            out_meta = raster_src.meta 
            out_raster["meta"] = out_meta
            out_raster["meta"]["transform"] = out_transform
            out_raster["meta"]["width"] = out_image.shape[2]
            out_raster["meta"]["height"] = out_image.shape[1]
            
            # Envelope means to return the whole crop image (not burning the zone outside of the Polygon as NODATA)
            
            if envelope:
                
                out_raster["array"] = out_image
                
            else:
                # Burns the region outside polygon as NODATA
                out_raster["array"] = out_image
                interior_mask = _create_interior_mask_from_raster_and_polygon(out_raster, geometries=geometry)
                out_raster["array"] = out_image*interior_mask["array"]
            
            return out_raster
        
    def _save(
        self,
        raster_in: dict,
        compress = "DEFLATE"
        ) -> None:

        # Path("/".join(self._filepath.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        
        raster_in["meta"].update(
            {
                "count": raster_in["array"].shape[0],
                "width": raster_in["array"].shape[2],
                "height": raster_in["array"].shape[1],
                "dtype": str(raster_in["array"].dtype),
                "compress": compress
                }
            )
        
        if 'not_exists' not in raster_in['meta']:
            
            with self._fs.open(save_path, **self._fs_open_args_save) as fs_file:
                with rasterio.open(fs_file, 'w', **raster_in["meta"]) as dst:
                    
                    dst.write(
                        raster_in["array"]
                        # .astype(rasterio.int16)
                        )
                    
                # del raster_in
                    
                # except KeyboardInterrupt:
                #     if os.path.exists(self._filepath):
                #         print(f"Deleting raster in {self._filepath}...")
                #         os.remove(self._filepath)
                #     raise
        self._invalidate_cache()

    def _exists(self) -> bool:
        # return Path(self._filepath).exists()
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
        except DataSetError:
            return False
        
        return self._fs.exists(load_path)

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
            load_args=self._load_args,
            save_args=self._save_args,
            version=self._version,
        )
        
    def _release(self) -> None:
        super()._release()
        self._version_cache.clear()
    
    # def _invalidate_cache(self) -> None:
    #     """Invalidate underlying filesystem caches."""
    #     # filepath = get_filepath_str(self._filepath, self._protocol)
    #     filepath = get_filepath_str(self._filepath, self._protocol)
    #     self._fs.invalidate_cache(filepath)
    
        
        
        
