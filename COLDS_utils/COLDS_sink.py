# =====================================================================================================================
# Filename:     COLDS_sink.py
# Written by:   Keith Cusson                Date: Dec 2025
# Description:  This script contains the identify sinkholes function for the Cusson Open-source LiDAR Depression Scanner
# License:      MIT License (c) 2025 Keith Cusson
# =====================================================================================================================
# ---------------------------------------------------------------------------------------------------------------------
# Import packages
# ---------------------------------------------------------------------------------------------------------------------
import geopandas as gpd
from multiprocessing import Queue
import numpy as np
import os
from osgeo import gdal, ogr
import pandas as pd
from pathlib import Path
from qgis.core import *
from time import time


def identify_sinkholes(
        proj_path: str,
        queue: Queue
):
    """
    Function that takes the DEMs generated from the generate_dem script, and analyzes it to detect depressions, and
    quantify them with a sinkhole score. This function is run in a multiprocessing process to prevent the GUI from
    freezing during execution.

    :param proj_path: Path to the QGIS project.
    :type proj_path:  str

    :param queue:     Queue to post intermediate results of the function.
    :type queue:      multiprocessing.Queue

    :return:          None return, results are put on queue when complete.
    """
    # -----------------------------------------------------------------------------------------------------------------
    # Child Environment Setup
    # -----------------------------------------------------------------------------------------------------------------
    # Allow GDAL exceptions in the child process
    gdal.UseExceptions()

    # Create an instance of QGIS within the child process
    QgsApplication.setPrefixPath(prefixPath=os.getenv('QGIS_PREFIX_PATH'),
                                 useDefaultPaths=True)
    qgs: QgsApplication = QgsApplication([], False)
    qgs.initQgis()

    # Set up the processing framework once the child QGIS process has been initialized
    import processing
    from processing.core.Processing import Processing
    Processing.initialize()

    # -----------------------------------------------------------------------------------------------------------------
    # GDAL setup
    # -----------------------------------------------------------------------------------------------------------------
    # Create the GDAL drivers to be used in this process.
    drv_mem: gdal.Driver = gdal.GetDriverByName('MEM')
    drv_gtiff: gdal.Driver = gdal.GetDriverByName('GTiff')

    # Retrieve the paths required for the function
    path_home: Path = Path(proj_path).parent
    path_vec: Path = Path(proj_path).with_suffix('.gpkg')
    path_temp_in: Path = path_home / 'rasters/temp_in.tif'
    path_temp_out: Path = path_home / 'rasters/temp_out.tif'
    path_temp_pcurve: Path = path_home / 'rasters/temp_pcurve.tif'
    path_temp_tcurve: Path = path_home / 'rasters/temp_tcurve.tif'

    # -----------------------------------------------------------------------------------------------------------------
    # Clipping Vector Layer Setup
    # -----------------------------------------------------------------------------------------------------------------
    # Set up the AOI and water feature layers for use to clip raster data
    qlyr_clip: QgsVectorLayer = QgsVectorLayer(f"{str(path_vec)}|layername=AOI",
                                               'AOI',
                                               "ogr")

    tmp_poly: QgsVectorLayer = QgsVectorLayer(f'{str(path_vec)}|layername=Water_Features',
                                              'Water Features',
                                              'ogr')

    # If there are water features, remove the area they cover from the area of interest.
    if tmp_poly.featureCount() > 0:
        # Ensure that the water features geometry has no geometric errors that would hamper processing
        arg_params = {'INPUT': tmp_poly,
                      'METHOD': 1,
                      'OUTPUT': 'TEMPORARY_OUTPUT'}
        temp_poly = processing.run('native:fixgeometries', arg_params)['OUTPUT']

        # Use the difference function to remove the water features from the AOI
        arg_params = {'INPUT': qlyr_clip,
                      'OVERLAY': temp_poly,
                      'OUTPUT': 'TEMPORARY_OUTPUT'}

        qlyr_clip = processing.run('qgis:difference', arg_params)['OUTPUT']

    # Create a list that will hold all depression geodataframes to be returned to the main application as the result
    depression_list: list[gpd.GeoDataFrame] = []

    # Send signals for the top progress bar for processing
    path_rast_list: list[Path] = [rast_path for rast_path in path_home.glob('rasters/*.tif')]
    queue.put({
        'pbar_size': (len(path_rast_list), 0),
        'disp_perc': 0
    })

    # Iterate through each raster resolution and identify the depressions within their DEM
    for i, path_rast in enumerate(path_rast_list):
        # Emit the signals required to prepare the second progress bar
        queue.put({'desc': (f'Identifying sinkholes in {path_rast.stem}', 0)})

        # Fill Depressions
        # ----------------
        queue.put({
            'desc': (f'Filling Depressions', 1),
            'pbar_size': (7, 1),
            'disp_perc': 1
        })

        # Create a raster with depressions filled using GRASS function r.fill.dir
        start_time = time()
        arg_params = {'input': str(path_rast),
                      'format': 1,
                      'output': str(path_temp_out),
                      'direction': 'TEMPORARY_OUTPUT',
                      'areas': 'TEMPORARY_OUTPUT'}
        processing.run("grass:r.fill.dir", arg_params)

        # Emit progress signals
        queue.put({
            'msg': f'Fill raster generated in {time() - start_time:.3f} seconds',
            'progress': (1, 1)
        })

        # Compute Fill Difference
        # -----------------------
        """ Open the DEM raster. To add additional bands, a memory copy of the DEM will be created, and new bands will 
        be added as the function progresses. Upon completion of the loop, the memory dataset will be saved to the
        original file. """
        start_time = time()
        queue.put({'desc': ('Computing fill difference', 1)})

        with gdal.OpenShared(str(path_rast)) as ds:
            ds_dem: gdal.Dataset = drv_mem.CreateCopy('DEM Memory',
                                                      ds,
                                                      0)

        # Retrieve the numpy arrays of the DEM and fill rasters.
        arr_dem: np.ndarray = ds_dem.ReadAsArray()
        with gdal.OpenEx(str(path_temp_out)) as ds_fill:
            arr_fill: np.ndarray = ds_fill.ReadAsArray()

        # Write the filled DEM data to the DEM file
        ds_dem.AddBand(gdal.GDT_Float64)
        ds_dem.GetRasterBand(ds_dem.RasterCount).WriteArray(arr_fill)

        # Convert nodata values to np.nan to prevent unintended computation results
        arr_dem = np.where(arr_dem == -9999., np.nan, arr_dem)
        arr_fill = np.where(arr_fill == -9999., np.nan, arr_fill)

        # Subtract the DEM from the filled DEM to identify areas that were filled
        arr_fill -= arr_dem

        # Replace nan values with -9999 for nodata values, and write the array to the DEM file.
        arr_fill = np.where(np.isnan(arr_fill), -9999., arr_fill)
        ds_dem.AddBand(gdal.GDT_Float64)
        ds_dem.GetRasterBand(ds_dem.RasterCount).WriteArray(arr_fill)

        # Emit progress signals
        queue.put({
            'msg': f'Fill difference computed in {time() - start_time:.3f} seconds',
            'progress': (2, 1)
        })

        # Compute Slope/Aspect of raster difference
        # -----------------------------------------
        start_time = time()
        queue.put({'desc': ('Computing slope and curvature', 1)})

        # Create a temporary single band raster of the fill difference raster to be used for slope and aspect analysis
        xs = ds_dem.RasterXSize
        ys = ds_dem.RasterYSize

        with drv_gtiff.Create(str(path_temp_in), xs, ys, 1, gdal.GDT_Float64) as ds_temp:
            ds_temp.SetSpatialRef(ds_dem.GetSpatialRef())
            ds_temp.SetGeoTransform(ds_dem.GetGeoTransform())
            ds_temp.WriteArray(arr_fill)

        # The slope/aspect tool will measure the slope of the raster at every pixel, as well as the vertical and
        # tangential curvature.
        arg_params = {'elevation': str(path_temp_in),
                      'slope': str(path_temp_out),
                      'pcurvature': str(path_temp_pcurve),
                      'tcurvature': str(path_temp_tcurve)}
        processing.run('grass:r.slope.aspect', arg_params)

        # Save the results of the slope processing back to the DEM raster as 3 bands.
        for temp_path in [path_temp_out, path_temp_pcurve, path_temp_tcurve]:
            ds_dem.AddBand(gdal.GDT_Float64)
            with gdal.OpenEx(str(temp_path)) as ds_slope:
                ds_dem.GetRasterBand(ds_dem.RasterCount).WriteArray(ds_slope.ReadAsArray())

        # Emit progress signals
        queue.put({
            'msg': f'Slope and curvature rasters generated in {time() - start_time:.3f} seconds',
            'progress': (3, 1)
        })

        # Clip Raster Difference
        # ----------------------
        # Clip the fill raster with the
        arg_params = {'INPUT': str(path_temp_in),
                      'MASK': qlyr_clip,
                      'SOURCE_CRS': qlyr_clip.crs(),
                      'TARGET_CRS': qlyr_clip.crs(),
                      'CROP_TO_CUTLINE': False,
                      'KEEP_RESOLUTION': True,
                      'OUTPUT': f'TEMPORARY_OUTPUT'}
        temp_rast = processing.run('gdal:cliprasterbymasklayer', arg_params)['OUTPUT']

        # Emit progress signals
        queue.put({
            'msg': f'Fill difference clipped in {time() - start_time:.3f} seconds',
            'progress': (4, 1)
        })

        # Raster Difference Low Band Pass Filter
        # --------------------------------------
        start_time = time()
        queue.put({'desc': ('Applying low band pass filter', 1)})

        # Using the GRASS r.neighbours operator on the clipped raster produces the same result as the ArcGIS Pro
        # Spatial Analyst Filter tool set to low band pass
        arg_params = {'input': temp_rast,
                      'method': 0,
                      'size': 3,
                      'output': str(path_temp_out)}
        processing.run('grass:r.neighbors', arg_params)

        # Add the filtered fill difference raster back to the main raster dataset
        with gdal.OpenEx(str(path_temp_out)) as ds:
            arr_fill = ds.ReadAsArray()

        ds_dem.AddBand(gdal.GDT_Float64)
        filt_band: gdal.Band = ds_dem.GetRasterBand(ds_dem.RasterCount)
        filt_band.WriteArray(arr_fill)

        # Save the DEM dataset to file to save the data for further analysis.
        path_out: Path = path_rast.with_stem(f'{path_rast.stem}_data')
        ds: gdal.Dataset = drv_gtiff.CreateCopy(str(path_out),
                                                ds_dem,
                                                0)
        ds = None

        # Emit progress signals
        queue.put({
            'msg': f'Low Band Pass filter applied to fill difference in {time() - start_time:.3f} seconds',
            'progress': (5, 1)
        })

        # Polygonize Raster Difference
        # ----------------------------
        """ Reclassify the raster into a binary raster. The vertical accuracy of the LIDAR data is 0.15m, therefore 
        cells in the filtered raster that are deeper than that will be given a value of 1, and all other cells will be 
        given a value of 0 """
        start_time = time()
        queue.put({'desc': ('Converting depressions to polygons', 1)})

        # Classify the cells, and write the array to a new band in the memory raster
        arr_class: np.ndarray = np.where(arr_fill >= 0.15, 1, 0)

        # Create a new memory raster band, and write this array to the band
        ds_dem.AddBand(gdal.GDT_Int64)
        band_class: gdal.Band = ds_dem.GetRasterBand(ds_dem.RasterCount)
        band_class.WriteArray(arr_class)

        # Open the vector file as a datasource, and create a new layer for the polygonize function
        with gdal.OpenEx(str(path_vec), nOpenFlags=gdal.OF_UPDATE) as ds_vec:
            ds_vec: gdal.Dataset
            # Create the layer that will hold the identified depressions
            lyr_name = path_rast.stem.replace('DEM', 'Depressions')
            lyr_depressions: ogr.Layer = ds_vec.CreateLayer(lyr_name,
                                                            srs=ds_dem.GetSpatialRef(),
                                                            geom_type=ogr.wkbPolygon)

            # Create a field to hold the values from the classified raster.
            fld: ogr.FieldDefn = ogr.FieldDefn('Value', ogr.OFTInteger)
            lyr_depressions.CreateField(fld)
            dep_fld = lyr_depressions.GetLayerDefn().GetFieldIndex('Value')

            # Polygonize the layer using the binary classified raster as both the data and a mask. This will generate
            # only those polygons that are valid depressions, and ignore the remainder of the area.
            gdal.Polygonize(band_class,
                            band_class,
                            lyr_depressions,
                            dep_fld,
                            [],
                            callback=None)

        # Emit progress signals
        queue.put({
            'msg': f'Depressions converted to vector polygons in {time() - start_time:.3f} seconds',
            'progress': (6, 1)
        })

        # Depression Analysis
        # -------------------
        start_time = time()
        queue.put({'desc': ('Analyzing Depressions', 1)})

        """ -----------------------------------------------------------------------------------------------------------
        Smoothing 
        The resulting polygons have a jagged appearance inherited from the pixelation of the Raster. To reduce the
        effects of pixelation, the polygons will be smoothed using the smooth geometry processing algorithm in QGIS. 
        ------------------------------------------------------------------------------------------------------------ """
        # Create a QGIS vector layer from the depressions layer
        qlyr_dep: QgsVectorLayer = QgsVectorLayer(str(path_vec) + "|layername=" + lyr_name,
                                                  lyr_name,
                                                  "ogr")

        arg_params = {'INPUT': qlyr_dep,
                      'OFFSET': 0.5,
                      'OUTPUT': f'TEMPORARY_OUTPUT'}
        tmp_poly = processing.run('native:smoothgeometry', arg_params)['OUTPUT']

        # Once the features have been smoothed, they can be saved back to the original layer
        arg_params = {'INPUT': tmp_poly,
                      'OUTPUT': str(path_vec),
                      'LAYER_NAME': lyr_name,
                      'ACTION_ON_EXISTING_FILE': 1}
        processing.run("native:savefeatures", arg_params)

        """ -----------------------------------------------------------------------------------------------------------
        The polygons can now be manipulated in a GeoDataFrame to analyze the results, and make inferences based on the
        data computed. When the analysis is complete, the data can be resaved back to the polygon layer in the vector
        geopackage.
        ------------------------------------------------------------------------------------------------------------ """
        # Open the depression layer as GeoDataFrame.
        gdf_depressions: gpd.GeoDataFrame = gpd.read_file(str(path_vec),
                                                          layer=lyr_name)

        """ -----------------------------------------------------------------------------------------------------------
        Shape / Size 
        To determine the aspect ratio of each shape, the major and minor axis of the shapes need to be computed. By
        computing the minimum oriented rectangle, the closest fit of a rectangle oriented along the major axis and 
        spanning across the minor axis is produced, providing the length of both sides. While not exact, it should 
        provide a close approximation of the length of both axes.
        ------------------------------------------------------------------------------------------------------------ """
        # Compute the minimum bounding rectangle
        gs_min_rect: gpd.GeoSeries = gdf_depressions['geometry'].minimum_rotated_rectangle()

        # Convert the polygons to a dataframe of vertex coordinates
        df_min_verts: pd.DataFrame = gs_min_rect.get_coordinates(index_parts=True)

        # Bring the polygon id and vertex number out of the index and into the dataframe
        df_min_verts.reset_index(inplace=True,
                                 names=['PolyNo', 'VertNo'])

        # Only the first 3 vertices of each bounding box is required to compute the length and width of the boxes.
        df_min_verts = df_min_verts[df_min_verts['VertNo'] < 3]

        # Create points for each vertex, and compute the distance between adjacent vertices
        gdf_min_verts: gpd.GeoDataFrame = gpd.GeoDataFrame(
            data=df_min_verts,
            geometry=gpd.points_from_xy(
                x=df_min_verts['x'],
                y=df_min_verts['y'],
                crs=gdf_depressions.crs
            )
        )
        first: pd.Series = gdf_min_verts['VertNo'] < 2
        last: pd.Series = gdf_min_verts['VertNo'] > 0
        gdf_min_verts.loc[first, 'distance'] = gdf_min_verts[first].distance(gdf_min_verts[last], align=False)

        # Use aggregate functions to find the min and max distances for each polygon, and assign them back to the
        # depression GeoDataFrame
        gdf_depressions[['Width', 'Height']] = (
            gdf_min_verts[['PolyNo', 'distance']].groupby(['PolyNo']).agg(func=['min', 'max']))

        # Compute the area and perimeter of the shape to avoid redundant calculation
        gdf_depressions['Area'] = gdf_depressions.area
        gdf_depressions['Length'] = gdf_depressions.length

        # Compare the width to the length to compute the aspect ratio of the shape
        gdf_depressions['Aspect'] = gdf_depressions['Width'] / gdf_depressions['Height']

        # Compute the roundness of the shape which is defined as 4 * pi * area / perimeter^2
        gdf_depressions['Roundness'] = 4 * np.pi * gdf_depressions.Area / np.square(gdf_depressions.Length)

        # Comparing the area of a shape to the area of its convex hull provides a measure of how convex the shape is.
        gdf_depressions['Convex'] = gdf_depressions.Area / gdf_depressions.geometry.convex_hull.area

        """ -----------------------------------------------------------------------------------------------------------
        Sinkhole Score 
        From these values a Sinkhole Score can be computed. By multiplying multiple values that are less than 1, a
        value is returned that is within the range of [0, 1] providing a confidence value of the polygon. To weight
        specific criteria from the result, raising them to a higher power increases their influence on the score.
        ------------------------------------------------------------------------------------------------------------ """
        gdf_depressions['Score'] = (gdf_depressions['Convex'] ** 2 *
                                    gdf_depressions['Aspect'] ** 1.5 *
                                    gdf_depressions['Roundness'])

        # Save the depressions back to the GeoPackage.
        gdf_depressions.to_file(str(path_vec),
                                layer=lyr_name,
                                crs=gdf_depressions.crs,
                                engine='fiona')

        """ -----------------------------------------------------------------------------------------------------------
        Zonal Statistics
        While not used for assigning a sinkhole score, associating the aspect and curvature of each depression with
        their respective polygon may provide additional insight. Using the zonal statistics function in QGIS allows this
        analysis to be conducted on the resultant data.
        ------------------------------------------------------------------------------------------------------------ """
        # Create a QGIS vector layer from the depressions layer
        qlyr_dep: QgsVectorLayer = QgsVectorLayer(str(path_vec) + "|layername=" + lyr_name,
                                                  lyr_name,
                                                  "ogr")

        # Using zonal statistics, the slope and curvature information can be added to the depression shapefile.
        arg_params = {'INPUT': qlyr_dep,
                      'INPUT_RASTER': str(path_out),
                      'RASTER_BAND': 4,
                      'COLUMN_PREFIX': 'S_',
                      'STATISTICS': [2, 3, 5, 6],
                      'OUTPUT': 'TEMPORARY_OUTPUT'}
        tmp_poly = processing.run('native:zonalstatisticsfb', arg_params)['OUTPUT']

        arg_params = {'INPUT': tmp_poly,
                      'INPUT_RASTER': str(path_out),
                      'RASTER_BAND': 5,
                      'COLUMN_PREFIX': 'P_',
                      'STATISTICS': [2, 3, 5, 6],
                      'OUTPUT': 'TEMPORARY_OUTPUT'}

        tmp_poly = processing.run('native:zonalstatisticsfb', arg_params)['OUTPUT']

        arg_params = {'INPUT': tmp_poly,
                      'INPUT_RASTER': str(path_out),
                      'RASTER_BAND': 6,
                      'COLUMN_PREFIX': 'T_',
                      'STATISTICS': [2, 3, 5, 6],
                      'OUTPUT': 'TEMPORARY_OUTPUT'}

        tmp_poly = processing.run('native:zonalstatisticsfb', arg_params)['OUTPUT']

        # Once the zonal statistics have been computed, they can be saved back to the original geopackage layer
        arg_params = {'INPUT': tmp_poly,
                      'OUTPUT': str(path_vec),
                      'LAYER_NAME': lyr_name,
                      'ACTION_ON_EXISTING_FILE': 1}
        processing.run("native:savefeatures", arg_params)

        # Reopen the depressions layer as a geodataframe
        gdf_depressions = gpd.read_file(str(path_vec),
                                        layer=lyr_name)

        # Add an indicator for the resolution, and add the depression geodataframe to the main list
        gdf_depressions['Resolution'] = path_rast.stem.replace('DEM_', '')
        depression_list.append(gdf_depressions)

        # Emit progress signals
        queue.put({
            'msg': f'{len(gdf_depressions)} Depressions analyzed in {time() - start_time:.3f} seconds',
            'progress': (7, 1)
        })
        queue.put({'progress': (i + 1, 0)})

    # Delete temporary files
    for tmp_path in (path_home / 'rasters').glob('temp_*'):
        tmp_path.unlink()

    # Emit the finished signal to signify the process has completed.
    queue.put({
        'result': (pd.concat(depression_list, ignore_index=True), None)
    })

    # Upon completion, exit the child instance of QGIS
    qgs.exitQgis()
