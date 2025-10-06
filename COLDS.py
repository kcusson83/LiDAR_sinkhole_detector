# =====================================================================================================================
# Filename:     COLDS.py
# Written by:   Keith Cusson                Date: Aug 2025
# Description:  This script contains the primary application for the Cusson Open-source LiDAR Depression Scanner
# License:      MIT License (c) 2025 Keith Cusson
# =====================================================================================================================
# ---------------------------------------------------------------------------------------------------------------------
# Import packages
# ---------------------------------------------------------------------------------------------------------------------
from COLDS_utils import COLDS_gui as gui
import gc
import geopandas as gpd
from multiprocessing import Manager, Queue, pool, Process, set_start_method
import numpy as np
import os
from osgeo import gdal, ogr
import pandas as pd
from pathlib import Path
import pdal
import pyproj
from qgis.core import *
import re
from shapely import Polygon, box
from time import time, sleep


# =====================================================================================================================
# CLASSES
# =====================================================================================================================
class Colds(gui.MainWindow):
    """
    Subclass of the COLDS_gui.MainWindow that provides additional functionality relating to QGIS integration.
    """

    # Gui window constructor
    def __init__(
            self,
            app: QgsApplication = None):
        # Initalize the methods of the main window
        super().__init__(app)
        self.qgs_proj: QgsProject | None = None
        self.progress_queue: Queue = Queue()

    def update_data(
            self,
            qgs_proj: QgsProject = None,
            data: gui.ProjectData = None
    ):
        """
        Updates the window's project data. To be used as a slot for a threaded emitter.

        :param qgs_proj: QGIS Project instance to be updated. [Default: None]
        :type qgs_proj:  qgis.core.QgsProject

        :param data:     Project Data object to be updated.
        :type data:      gui.ProjectData

        :return:         No return, updates the window variables in place.
        """
        if qgs_proj is not None:
            self.qgs_proj = qgs_proj

        if data is not None:
            self.data = data

    # ------------------------------------------------------------------------------------------------------------------
    # Project Menu
    def create_project(
            self,
            name: str,
            proj_path: str = None
    ):
        """
        This method creates a new project file

        :param name:      Name of the project.
        :type name:       str

        :param proj_path: Path where the project file will be stored.
        :type proj_path:  str

        :return:          None
        """
        if proj_path is None:
            return

        # Create a new QGIS project instance.
        qgs_proj: QgsProject = QgsProject.instance()

        # Set the project identication values
        qgs_proj.setTitle(name)
        qgs_proj.setPresetHomePath(proj_path)
        qgs_proj.setCustomVariables(self.data.export_project_state())

        # Add the link to this application to the project links.
        colds_link: QgsAbstractMetadataBase.Link = QgsProjectMetadata.Link(
            name='Cusson Open-source LiDAR Depression Scanner',
            type='GIT',
            url='https://github.com/CunningStuntK/LiDAR_sinkhole_detector'
        )
        colds_link.description = 'Source code for the program that generated this file.'
        qgs_proj.metadata().addLink(colds_link)

        # Create the folder structure for the project
        folders = ['las_files',
                   'rasters/tiles']
        for f in folders:
            folder = Path(proj_path) / f
            folder.mkdir(parents=True,
                         exist_ok=True)

        # Save the project to file
        out_path: Path = Path(proj_path) / f'{valid_filename(name)}.qgz'
        qgs_proj.setFileName(str(out_path))
        qgs_proj.write()

        # Save the project handle to the application, and prepare the Main Window for use.
        self.qgs_proj = qgs_proj
        self.menu_bar.toggle_add_menu(True)
        self.setWindowTitle(f'{self.program_name} - {self.qgs_proj.title()}')
        self.menu_bar.update_recent(str(out_path))

    def open_project(
            self,
            filename: str = None
    ) -> None:
        """
        This method opens an existing project from a file.

        :param filename: Path to project file to be opened.
        :type filename:  str

        :return: None return
        """
        if filename is None:
            return
        self.qgs_proj: QgsProject = QgsProject.instance()
        result = self.qgs_proj.read(filename=filename)
        if result:
            self.qgs_proj.pathResolver()
        else:
            self.dlg_error('File Open',
                           'Invalid project file, please try again.')
            self.qgs_proj = None
            return
        self.setWindowTitle(f'{self.program_name} - {self.qgs_proj.title()}')

        self.menu_bar.update_recent(filename)

        # Check if the project geopackage exists
        gpkg_path: Path = Path(filename).with_suffix('.gpkg')
        if gpkg_path.exists():
            self.open_project_geopackage(str(gpkg_path))

        # Read the current project state
        self.data.update_proj_state(self.qgs_proj.customVariables())

        self.menu_bar.toggle_add_menu(self.data.state == 0)
        self.wgt_buttons.enable_step_btn(self.data.export_project_state())
        if self.data.state > 0:
            self.wgt_table.enable_selection(0)
            if len(self.data.gdf_vec_md) > 0:
                self.wgt_table.enable_selection(1)
        if self.data.state > 1:
            self.wgt_table.enable_selection(2)

    def open_project_geopackage(
            self,
            filename: str
    ):
        """
        This method opens up an existing project geopackage, and sets the project variables as applicable

        :return: None, sets application project data based on file.
        """
        df_lyrs: pd.DataFrame = gpd.list_layers(filename)

        if 'AOI' in df_lyrs.values:
            self.data.gdf_aoi = gpd.read_file(filename,
                                              layer='AOI')

        if 'Water_Features' in df_lyrs.values:
            self.data.gdf_water = gpd.read_file(filename,
                                                layer='Water_Features')

        if 'Point_Clouds' in df_lyrs.values:
            self.data.gdf_pc_md = gpd.read_file(filename,
                                                layer='Point_Clouds')

        if 'Vector_Metadata' in df_lyrs.values:
            self.data.gdf_vec_md = gpd.read_file(filename,
                                                 layer='Vector_Metadata')

    # ------------------------------------------------------------------------------------------------------------------
    # Add Menu
    def add_cloud(
            self,
            cloud_type: str,
            filename: list[str]
    ):
        """
        This method adds any point clouds loaded by selecting point clouds from the Add menu

        :param cloud_type: Cloud type label for loaded files.
        :type cloud_type:  str

        :param filename:   Path(s) to the list of point cloud(s) to be uploaded
        :type filename:    list[str]

        :return:           None
        """
        # Retrieve the metadata from the file(s) as a list of dictionaries
        md_list = []
        for las_file in filename:
            # Prior to reading the file, determine if it has already been added to the geodataframe.
            if las_file in self.data.gdf_pc_md.values:
                self.dlg_warning('File Exists',
                                 f'This file has already been loaded:\n{las_file}')
                continue
            md = read_input_las_metadata(las_file, cloud_type)
            md_list.append(md)

        if len(md_list) < 1:
            return

        # Generate a geodataframe from the remaining files, and concatenate it with the existing frame.
        gdf_las: gpd.GeoDataFrame = gpd.GeoDataFrame(
            data=md_list,
            geometry=gpd.GeoSeries.from_wkt([md['extent'] for md in md_list]))

        gdf_las.drop('extent',
                     axis=1,
                     inplace=True)

        self.data.gdf_pc_md = pd.concat([self.data.gdf_pc_md, gdf_las],
                                        ignore_index=True)
        self.wgt_message.print_message(f'{len(gdf_las)} point cloud(s) added to the project.')
        self.wgt_table.btn_accept.setVisible(True)

        # Ensure the data type has been activated in the combo box
        self.wgt_table.enable_selection(0)

        # If no selection has been made in the combo box, select Input Cloud
        if self.wgt_table.combo_data_type.currentIndex() == -1:
            self.wgt_table.combo_data_type.setCurrentIndex(0)

        # If Input Cloud is the current selection, update the table.
        if self.wgt_table.combo_data_type.currentIndex() == 0:
            self.change_type()

    def add_vector(
            self,
            lyr_type: str,
            filename: list[str]
    ) -> None:
        """
        This function provides all the functionality for adding vector files, including adding them to the vector
        metadata DataFrame, enabling the selector combo box, and updating the statistics table area.

        :param lyr_type: String representing the type of vector layer loaded.
        :type lyr_type:  str

        :param filename: List of paths to files to be added to the project.
        :type filename:  list[str]

        :return:         None
        """
        # Create a list to hold all loaded dataframes.
        vec_df_list: list[pd.DataFrame] = [self.data.gdf_vec_md]

        # Load the vector file(s) selected by the user as geodataframes.
        for vec_file in filename:
            file_name = Path(vec_file).name
            vec_layers: pd.DataFrame = gpd.list_layers(vec_file)
            # Add only the layers within the file that the user selects.
            if len(vec_layers) > 1:
                selected_layers = self.dlg_list(
                    title='Select Layers',
                    message=f'Which layers would you like to load for file:\n{file_name}',
                    vals=vec_layers.name.tolist()
                )
                if selected_layers is None:
                    continue
                # Remove all unselected layers from the dataframe.
                vec_layers = vec_layers[vec_layers['name'].isin(selected_layers)]

            # Add the additional metadata required for later processing.
            vec_layers['filename'] = vec_file
            vec_layers['layer_type'] = lyr_type

            # Determine if any of the selected layers already exist, and notify the user there can be no duplicates.
            if vec_file in self.data.gdf_vec_md.values:
                # If the file has already been loaded, check to see if any of its layers have been loaded using a merge.
                df_all: pd.DataFrame = vec_layers.merge(self.data.gdf_vec_md[['name', 'filename']],
                                                        on=['name', 'filename'],
                                                        how='left',
                                                        indicator=True)
                # If the layer is already in the metadata dataframe, notifu the user and do not add it.
                if 'both' in df_all['_merge'].values:
                    dup_lyrs = df_all.loc[df_all['_merge'] == 'both', 'name'].tolist()
                    warn_msg = f'The following layers in **{file_name}** have already been added:\n'
                    for lyr in dup_lyrs:
                        warn_msg += f'  + {lyr}\n'
                    self.dlg_warning('Duplicate Layer',
                                     warn_msg)

                    vec_layers = vec_layers[df_all['_merge'] != 'both']

            # If no layers have been identified, move to the next file
            if len(vec_layers) == 0:
                continue

            # Retrieve the spatial reference for each layer
            ds_vec: gdal.Dataset = ogr.Open(vec_file)
            vec_layers['srs'] = vec_layers.apply(
                lambda x: ds_vec.GetLayer(x['name']).GetSpatialRef(),
                axis=1
            )

            # Compute the EPSG code(s) for layer(s) that have a spatial reference.
            filt: np.ndarray[bool] = vec_layers.srs.isnull()
            vec_layers.loc[~filt, ['Horizontal EPSG', 'Vertical EPSG']] = vec_layers[~filt].apply(
                lambda x: epsg_codes_from_wkt(x.srs.ExportToWkt()),
                axis=1,
                result_type='expand'
            )

            # Convert instances where no spatial reference was detected to empty strings
            vec_layers.loc[filt, ['Horizontal EPSG', 'Vertical EPSG']] = ''

            # Store the compound wkt description of the spatial reference for future processing.
            vec_layers.loc[:, 'wkt'] = vec_layers.apply(
                lambda x: combine_epsg_codes(x['Horizontal EPSG'], x['Vertical EPSG']),
                axis=1
            )

            vec_layers['geometry'] = None
            vec_layers = gpd.GeoDataFrame(data=vec_layers,
                                          geometry='geometry')
            vec_layers.drop(columns=['srs'],
                            inplace=True)

            vec_df_list.append(vec_layers)

        # If the vector list only contains the original dataframe, do not make any changes
        if len(vec_df_list) == 1:
            return

        # Add the data to the vector metadata DataFrame.
        self.data.gdf_vec_md = pd.concat(vec_df_list,
                                         ignore_index=True)
        self.wgt_message.print_message(f'{len(vec_df_list) - 1} {lyr_type} vector layer(s) added to the project.')

        # Ensure the data type has been activated in the combo box
        self.wgt_table.enable_selection(1)

        # If no selection has been made in the combo box, select Vector Data
        if self.wgt_table.combo_data_type.currentIndex() == -1:
            self.wgt_table.combo_data_type.setCurrentIndex(1)

        # If Vector Data is the current selection, update the table.
        if self.wgt_table.combo_data_type.currentIndex() == 1:
            self.change_type()

    # ------------------------------------------------------------------------------------------------------------------
    # Table View
    def change_type(self):
        """
        This function updates the display table if a change in the display selector is detected

        :return: None return, updates the table.
        """
        # Determine the state of the selector
        match self.wgt_table.combo_data_type.currentIndex():
            # Input Cloud
            case 0:
                # Filter the point cloud dataframe for input clouds
                filt: pd.Series = self.data.gdf_pc_md.cloud_type == 'Input Cloud'
                columns = ['filename',
                           'name',
                           'Size (bytes)',
                           '# Points',
                           'min x',
                           'max x',
                           'min y',
                           'max y',
                           'Classified',
                           'Horizontal EPSG',
                           'Vertical EPSG',
                           'Read Time (s)']

                # Create the dataframe to be set as the model.
                df_model = self.data.gdf_pc_md.loc[filt, columns]

            # Vector File
            case 1:
                columns = ['filename',
                           'name',
                           'layer_type',
                           'geometry_type',
                           'Horizontal EPSG',
                           'Vertical EPSG']
                df_model = self.data.gdf_vec_md.loc[:, columns]

            # Tile Cloud
            case 2:
                # Filter the point cloud data for tile clouds
                filt: pd.Series = self.data.gdf_pc_md.cloud_type == 'Tile Cloud'
                columns = ['filename',
                           'name',
                           'Size (bytes)',
                           '# Points',
                           'min x',
                           'max x',
                           'min y',
                           'max y',
                           'Classified',
                           'Horizontal EPSG',
                           'Vertical EPSG',
                           'Read Time (s)']

                # Create the dataframe to be set as the model.
                df_model = self.data.gdf_pc_md.loc[filt, columns]

            case _:
                return
        self.wgt_table.table.clearSelection()
        self.wgt_table.model.update_data(df_model)
        self.wgt_table.table.resizeColumnsToContents()

    def remove_file(
            self,
            index: int
    ):
        """
        Method that removes an input file selected from the stats table.

        :param index: Row index for the file to be removed.
        :type index:  int

        :return:      None. Removes the file from the appropriate project dataframe and refreshes the stats table view.
        """
        # Determine what type of file you intend to remove
        if self.wgt_table.combo_data_type.currentIndex() == 0:
            self.wgt_message.print_message(
                f'Removed point cloud: {self.data.gdf_pc_md.loc[index, "filename"]}'
            )
            self.data.gdf_pc_md.drop([index],
                                     inplace=True)
            self.data.gdf_pc_md.reset_index(inplace=True,
                                            drop=True)
            if len(self.data.gdf_pc_md) == 0:
                self.wgt_table.btn_accept.setVisible(False)
        elif self.wgt_table.combo_data_type.currentIndex() == 1:
            self.wgt_message.print_message(
                f'Removed {self.data.gdf_vec_md.loc[index, "layer_type"]} vector layer: '
                f'{self.data.gdf_vec_md.loc[index, "filename"]}'
            )
            self.data.gdf_vec_md.drop([index],
                                      inplace=True)
            self.data.gdf_pc_md.reset_index(inplace=True,
                                            drop=True)
        else:
            return

        self.change_type()

    # ------------------------------------------------------------------------------------------------------------------
    # Input File finalization
    def finalize_start(self):
        """
        Method that finalizes input files, and adds them to project.

        :return: None. Updates ProjectData and qgs_proj, saves data to file. Final processing is done in a thread.
        """
        # Get confirmation that all required files have been added.
        msg = f'Are these all the files required for the project?'
        if len(self.data.gdf_vec_md) > 0:
            df_vec: pd.DataFrame = self.data.gdf_vec_md.drop(columns=['geometry'])
        else:
            df_vec: pd.DataFrame = pd.DataFrame()
        response = self.dlg_input_tree('Finalize inputs',
                                       'Are these all the files required for the project?',
                                       df_vec,
                                       self.data.gdf_pc_md.drop(columns=['geometry']))

        # If the user responds no, uncheck the button, and exit the function.
        if not response:
            self.wgt_table.btn_accept.setChecked(False)
            return

        # Determine the Horizontal and Vertical CRS for the project
        df_epsg: pd.DataFrame = pd.concat([self.data.gdf_pc_md, self.data.gdf_vec_md],
                                          ignore_index=True)
        if len(df_epsg.drop_duplicates(subset=['Horizontal EPSG', 'Vertical EPSG'])) != 1:
            df_epsg.loc[~(df_epsg['Horizontal EPSG'] == ''), 'H_Desc'] = (
                df_epsg[~(df_epsg['Horizontal EPSG'] == '')].apply(
                    lambda x: pyproj.CRS.from_epsg(x['Horizontal EPSG']).name,
                    axis=1
                ))
            df_epsg.loc[~(df_epsg['Vertical EPSG'] == ''), 'V_Desc'] = (
                df_epsg[~(df_epsg['Vertical EPSG'] == '')].apply(
                    lambda x: pyproj.CRS.from_epsg(x['Vertical EPSG']).name,
                    axis=1
                ))
            df_epsg.loc[df_epsg.cloud_type == 'Input Cloud', 'layer_type'] = 'Point Cloud'

            epsg_list: list[str] = self.dlg_epsg_table('Select Coordinate System',
                                                       'Given the following coordinate reference systems from the '
                                                       'input files, select a horizontal and vertical coordinate '
                                                       'system for the project.<br>'
                                                       'Ideally, select the relevant coordinate systems from the point '
                                                       'cloud inputs to generate the most accurate results.',
                                                       df_epsg[['filename',
                                                                'H_Desc',
                                                                'V_Desc',
                                                                'layer_type',
                                                                'name',
                                                                'Horizontal EPSG',
                                                                'Vertical EPSG']])
        else:
            columns = ['Horizontal EPSG', 'Vertical EPSG']
            epsg_list: np.ndarray = df_epsg.drop_duplicates(subset=columns)[columns].to_numpy()[0]
            epsg_list: list[str] = epsg_list.tolist()

        # Prepare the application for displaying progress bars and running the thread
        self.wgt_progress.show_bars(2)
        self.stack.setCurrentIndex(1)
        self.wgt_progress.update_desc('Finalizing Inputs')
        self.wgt_table.btn_accept.setVisible(False)
        self.menu_bar.toggle_add_menu(False)

        # Create the EPSG for the project from the provided inputs
        if not epsg_list[0] == '':
            crs_h: QgsCoordinateReferenceSystem = QgsCoordinateReferenceSystem(f'EPSG:{epsg_list[0]}')
            self.qgs_proj.setCrs(crs_h)
            self.wgt_message.print_message(f'Horizontal Spatial Reference System set to EPSG:{epsg_list[0]}')
        if not epsg_list[1] == '':
            crs_v: QgsCoordinateReferenceSystem = QgsCoordinateReferenceSystem(f'EPSG:{epsg_list[1]}')
            self.qgs_proj.setVerticalCrs(crs_v)
            self.wgt_message.print_message(f'Vertical Spatial Reference System set to EPSG:{epsg_list[1]}')
        proj_wkt = combine_epsg_codes(epsg_list[0], epsg_list[1])

        # Update the progress bar.
        self.wgt_progress.update_bar(10)

        # Update the project state
        self.data.state = 1
        process: Process = Process(target=finalize_inputs,
                                   args=(Path(self.qgs_proj.fileName()),
                                         self.data,
                                         proj_wkt,
                                         self.progress_queue))

        self.worker_signals.result.connect(self.update_data)
        self.worker_signals.finished.connect(self.finalize_end)
        self.worker_signals.finished.connect(self.timer.stop)
        self.worker_signals.finished.connect(process.join)
        self.timer.start(200)
        process.start()

    def finalize_end(self):
        """
        Method that adds the input layers as layers to the QGIS project once the project's geopackage has been created.

        :return: None. Updates qgs_proj in place
        """
        # Add the layers to the QGIS project
        self.wgt_progress.update_desc('Saving layers to QGIS Project', 1)
        self.wgt_progress.reset_bar(3, 1)
        out_file: Path = Path(self.qgs_proj.fileName()).with_suffix('.gpkg')

        # Add water features to the QGIS project if they exist
        if len(self.data.gdf_water) > 0:
            qlyr = QgsVectorLayer(f'{str(out_file)}|layername=Water_Features',
                                  'Water Features',
                                  'ogr')
            qlyr_wf: QgsVectorLayer = self.qgs_proj.addMapLayer(qlyr)

            # Set the styling for the Water Feature layer
            qstyle = QgsStyle.defaultStyle().symbol('topo water')
            qrend = QgsSingleSymbolRenderer(qstyle)
            qlyr_wf.setRenderer(qrend)
            qlyr_wf.triggerRepaint()

            # Print a completion message
            self.wgt_message.print_message('Water features layer added to project')

        # Update the progress bars
        self.wgt_progress.update_bar(1, 1)
        self.wgt_progress.update_bar(90, 0)

        # Add the point cloud extents to the QGIS project file
        qlyr = QgsVectorLayer(f'{str(out_file)}|layername=Point_Clouds',
                              'Point Cloud Extents',
                              'ogr')
        qlyr_pc: QgsVectorLayer = self.qgs_proj.addMapLayer(qlyr)

        # Set the styling for the point cloud extents
        qstyle: QgsSymbol = QgsStyle.defaultStyle().symbol('outline green')
        qstyle.symbolLayer(0).setWidth(0.5)
        qrend: QgsFeatureRenderer = QgsSingleSymbolRenderer(qstyle)
        qlyr_pc.setRenderer(qrend)
        qlyr_pc.triggerRepaint()
        qlyr_pc.setOpacity(0.3)

        # Report function progress
        self.wgt_progress.update_bar(2, 1)
        self.wgt_progress.update_bar(95, 0)
        self.wgt_message.print_message('Point cloud extents layer added to project')

        # Add the area of interest layer to the QGIS project file
        qlyr: QgsVectorLayer = QgsVectorLayer(f'{str(out_file)}|layername=AOI',
                                              'AOI',
                                              'ogr')
        qlyr_aoi: QgsVectorLayer = self.qgs_proj.addMapLayer(qlyr)

        # Set the styling for the AOI layer
        qstyle: QgsSymbol = QgsStyle.defaultStyle().symbol('outline red')
        qstyle.symbolLayer(0).setWidth(0.5)
        qrend: QgsFeatureRenderer = QgsSingleSymbolRenderer(qstyle)
        qlyr_aoi.setRenderer(qrend)
        qlyr_aoi.triggerRepaint()

        # Report function process
        self.wgt_progress.update_bar(3, 1)
        self.wgt_progress.update_bar(99, 0)
        self.wgt_message.print_message('Area of interest layer added to project')

        # Change the state of the program, set the default view, and save the project file to disk.
        self.qgs_proj.setCustomVariables(self.data.export_project_state())
        view_settings: QgsProjectViewSettings = self.qgs_proj.viewSettings()
        extent: QgsReferencedRectangle = QgsReferencedRectangle(rectangle=qlyr_aoi.extent().buffered(50),
                                                                crs=self.qgs_proj.crs())
        view_settings.setDefaultViewExtent(extent)
        self.qgs_proj.write()

        # Save the vector metadata GeoDataFrame to the output file if there is any.
        if len(self.data.gdf_vec_md) > 0:
            self.data.gdf_vec_md.to_file(str(out_file),
                                         layer='Vector_Metadata')

        # Reset window use
        self.stack.setCurrentIndex(0)
        self.set_buttons()

    def click_tile(self):
        """
        Method to merge input point clouds and split them into equal sized tiles.

        :return: None. Performs the described actions, and saves the new point clouds to the las_tiles folder.
        """
        # Approximate the number of points in the area of interest by computing the percentage overlap between each
        # point cloud with the area of interest and using that to approximate the number of points from that point
        # cloud that are in the area of interest. Increase the result by 25% to account for point clouds that do not
        # cover their full bounding box extents
        aoi: Polygon = self.data.gdf_aoi.geometry.union_all()
        coverage: pd.Series = self.data.gdf_pc_md.intersection(aoi).area / self.data.gdf_pc_md.area

        approx_points = (coverage * self.data.gdf_pc_md['# Points']).sum() * 1.25

        tiles = 1000 * approx_points / gdal.GetUsablePhysicalRAM()

        # A tiling scheme needs to be derived to subdivide the area of interest. The ideal tiling scheme will create
        # equal size squares to cover the entirety of the bounding box. To compensate for boundary issues, there should
        # be a 5 metre overlap on each side of the box

        # Divide the area of the area of interest by the number of tiles to determine the side length for each tile.
        max_tile_area = aoi.area / tiles
        max_tile_length = np.sqrt(max_tile_area) - 10  # Subtracting 10 accounts for the buffer on both sides

        # Ensure the tile length is no larger than the size of the largest side of the bounding box of the area of
        # interest
        minx, miny, maxx, maxy = aoi.bounds
        max_tile_length = min(max_tile_length, max(maxx - minx, maxy - miny))

        # Create a grid of tiles with the max tile length, and overlay it onto the area of interest.
        tile_grid = [int(np.ceil((maxx - minx) / max_tile_length)),
                     int(np.ceil((maxy - miny) / max_tile_length))]

        # Warn the user that the process can be incredibly time-consuming, and confirm they wish to start.
        response = self.dlg_confirm('Merging/Tiling Point clouds',
                                    f'This process will convert the {len(self.data.gdf_pc_md)} loaded input '
                                    f'cloud(s) into {tile_grid[0] * tile_grid[1]:,} tile(s). This process may take '
                                    f'several hours depending on computer performance. If it is interrupted, the '
                                    f'process will need to be restarted. Would you like to continue?')

        if response == 65536:
            self.wgt_buttons.button_tile.setChecked(False)
            return

        # Run the merge/split thread.
        self.wgt_message.print_message('Point cloud merge & tile process started.')
        # Prepare the application for displaying progress bars and running the thread
        self.wgt_progress.show_bars(2)
        self.stack.setCurrentIndex(1)
        self.wgt_progress.update_desc('Merging and splitting point clouds')
        self.wgt_progress.reset_bar(len(self.data.gdf_pc_md) * 2)
        self.wgt_progress.set_perc(0)
        self.wgt_progress.update_desc('Creating grid', 1)
        self.wgt_progress.reset_bar(1, 1)
        self.wgt_buttons.disable_all()

        # Update the project state
        self.data.state = 2

        # Retrieve the project's spatial reference data
        epsg_list = [self.qgs_proj.crs().authid()[5:],
                     self.qgs_proj.verticalCrs().authid()[5:]]

        # Run the thread
        process: Process = Process(target=tile_and_merge,
                                   args=(self.data.gdf_pc_md,
                                         aoi,
                                         max_tile_length,
                                         Path(self.qgs_proj.fileName()),
                                         epsg_list,
                                         self.no_cpus,
                                         self.progress_queue))

        self.worker_signals.error.connect(print)
        self.worker_signals.error.connect(breakpoint)
        self.worker_signals.result.connect(self.data.update_gdf_pc_md)
        self.worker_signals.finished.connect(self.click_tile_cleanup)
        self.worker_signals.finished.connect(process.join)
        self.timer.start(50)
        process.start()

    def click_tile_cleanup(self):
        """
        This method reverts the display to where it should be after the tile_and_merge process has finished.

        :return: None
        """
        # Set the gui state and stop the buttons
        self.timer.stop()
        self.stack.setCurrentIndex(0)
        self.wgt_table.enable_selection(2)
        self.set_buttons()

        # Update the QGIS project, and save it to file.
        self.qgs_proj.setCustomVariables(self.data.export_project_state())
        self.qgs_proj.write()

    # TODO: REMOVE ALL TEST FUNCTIONS
    def test1(self):
        breakpoint()


# =====================================================================================================================
# FUNCTIONS
# =====================================================================================================================
def valid_filename(txt: str) -> str:
    """
    This function takes an input string and removes spaces and invalid characters to make it safe to be used as a
    filename.

    :param txt: Text to be modified

    :return:    String containing the valid filename
    """
    txt = re.sub(r'[/\\:*?"<>|]', '_', txt)
    txt = txt.strip()
    txt = txt.strip('.')
    txt = txt.replace(' ', '_')
    return txt


def epsg_codes_from_wkt(wkt: str) -> dict[str, str]:
    """
    This function returns the EPSG code from a Well-Known Text representation of a Coordinate Reference System. It is
    separated into horizontal and vertical components if applicable.

    :param wkt: Well-Known Text to be decoded.
    :type wkt:  str

    :return:    String representation of the EPSG code.
    """
    # Create the epsg dictionary that will be used to return the result.
    epsg = {'Horizontal EPSG': '',
            'Vertical EPSG': ''}

    # Determine if the well known text is in the correct format
    if wkt is None:
        return epsg

    if not pyproj.crs.is_wkt(wkt):
        return epsg

    crs: pyproj.CRS = pyproj.CRS.from_wkt(wkt)
    crs_list: list[pyproj.CRS] = crs.sub_crs_list
    if not crs.is_compound:
        crs_list.append(crs)

    for comp in crs_list:
        comp: pyproj.CRS
        if comp.is_vertical:
            epsg['Vertical EPSG'] = str(comp.to_epsg(25))
        else:
            epsg['Horizontal EPSG'] = str(comp.to_epsg(25))

    return epsg


def combine_epsg_codes(
        horizontal: str = '',
        vertical: str = ''
) -> str:
    """
    Given the EPSG codes for horizontal and vertical coordinate systems, return the well known text for a combined
    coordinate system. If only one coordinate system is provided, return that coordinate reference system.

    :param horizontal: EPSG for the horizontal coordinate reference system
    :type horizontal:  str

    :param vertical:   EPSG for the vertical coordinate reference system
    :type vertical:    str

    :return:           String well known text representation of the output coordinate reference system.
    """
    # Create the input coordinate reference systems
    if horizontal != '':
        crs_h: pyproj.CRS = pyproj.CRS.from_epsg(horizontal)
    else:
        crs_h = None

    if vertical != '':
        crs_v: pyproj.CRS = pyproj.CRS.from_epsg(vertical)
    else:
        crs_v = None

    # If both coordinate reference systems have been provided, combine them.
    if crs_h and crs_v:
        out_name: str = f'{crs_h.name} + {crs_v.name}'
        out: pyproj.CRS = pyproj.crs.CompoundCRS(name=out_name,
                                                 components=[crs_h, crs_v])
    # If only one CRS has been provided, set it to be the output CRS
    elif crs_h and not crs_v:
        out = crs_h
    elif crs_v and not crs_h:
        out = crs_v
    # If no coordinate reference systems have been provided, return an empty string.
    else:
        return ''

    # Return the well known text of the output CRS
    return out.to_wkt()


def classify_ground_pts(
        pts_arr: np.ndarray,
        queue: Queue
):
    """
    Using the Cloth Surface Filtering algorithm in PDAL, classify points as being ground/not-ground and return the
    results of the operation to the provided queue. This function is intended to be used as a multiprocessing Process.

    :param pts_arr: Point cloud points to be classified.
    :type pts_arr:  numpy.ndarray

    :param queue:   Queue to place the results when complete.
    :type queue:    multiprocessing.Queue

    :return:        None. Results of the operation are placed on the queue.
    """
    # Construct the pipeline that will be used to classify ground points and execute it.
    pl: pdal.Pipeline = pdal.Filter.csf(resolution=0.1,
                                        threshold=0.1).pipeline(pts_arr)
    _ = pl.execute()

    queue.put(pl.arrays[0][['X', 'Y', 'Z', 'Classification']])


def read_input_las_metadata(
        file_name: str,
        cloud_type: str
) -> dict:
    """
    This function reads in a LAS/LAZ file to retrieve the file's metadata. Reading a single point speeds up read time.

    Sample Dictionary::

        {
            name:             'file.las',
            filename:         'C:/path/to/file.las',
            Size (bytes):     12458734
            cloud_type:       'Input Cloud',
            # Points:         123456789,
            Classified:      'Unknown',
            min x:            415000.0,
            max x:            415999.9,
            min y:            4980000.0,
            max y:            4980999.9
            Horizontal EPSG: '2961',
            Vertical EPSG:   '6647',
            wkt:             GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",...,
            read_time:       0.27,
            extent:          'POLYGON ((415000.0 4980000.0,
                                        415999.9 4980000.0,
                                        415999.9 4980999.9,
                                        415000.0 4980999.9,
                                        415000.0 4980000.0))',
        }

    :param file_name:     Path to the LAS/LAZ file to be read.
    :type file_name:      str

    :param cloud_type:    Type of cloud metadata to be stored. Either Input Cloud or Tile Cloud.
    :type cloud_type:     str

    :return:              dict containing relevant metadata for the file read in.
    """
    # Create a pipeline to read the las file.
    pl: pdal.Pipeline = pdal.Reader(filename=file_name,
                                    count=1).pipeline()

    # Execute the pipeline and record how long it takes to read it.
    t_start = time()
    _ = pl.execute()
    md: dict = pl.metadata['metadata']['readers.las']

    out_dict = {
        'name': Path(file_name).name,
        'filename': file_name,
        'Size (bytes)': Path(file_name).stat().st_size,
        'cloud_type': cloud_type,
        '# Points': md['count'],
        'Classified': 'Unknown',
        'min x': float(md['minx']),
        'max x': float(md['maxx']),
        'min y': float(md['miny']),
        'max y': float(md['maxy'])
    }

    # Determine the SRS of the input las file.
    if 'compoundwkt' in md['srs']:
        out_dict |= epsg_codes_from_wkt(md['srs']['compoundwkt'])
        out_dict['wkt'] = md['srs']['compoundwkt']
    else:
        if 'horizontal' in md['srs']:
            out_dict['Horizontal EPSG'] = epsg_codes_from_wkt(md['srs']['horizontal'])['Horizontal EPSG']
            out_dict['wkt'] = md['srs']['horizontal']
        else:
            out_dict['Horizontal EPSG'] = ''
        if 'vertical' in md['srs']:
            out_dict['Vertical EPSG'] = epsg_codes_from_wkt(md['srs']['vertical'])['Vertical EPSG']
        else:
            out_dict['Vertical EPSG'] = ''

    out_dict['wkt'] = combine_epsg_codes(out_dict['Horizontal EPSG'],
                                         out_dict['Vertical EPSG'])

    out_dict['Read Time (s)'] = time() - t_start
    out_dict['extent'] = (f'POLYGON (({md["minx"]} {md["miny"]}, '
                          f'{md["maxx"]} {md["miny"]}, '
                          f'{md["maxx"]} {md["maxy"]}, '
                          f'{md["minx"]} {md["maxy"]}, '
                          f'{md["minx"]} {md["miny"]}))')

    # Return the metadata dictionary and the relevant data from the points array.
    return out_dict


# ----------------------------------------------------------------------------------------------------------------------
# Threading Functions
def finalize_inputs(
        proj_file: Path,
        data: gui.ProjectData,
        proj_wkt: str,
        queue: Queue
) -> None:
    """
    Function intended to be used in a multiprocessing process to create the geopackage that will be used to store vector
    data for the project. Creates the file and adds the layers Point_Clouds, AOI, and Water_features.

    :param proj_file: Path to the QGIS project file
    :type proj_file:  pathlib.Path

    :param data:     Project data object containing the data to be processed.
    :type data:      COLDS_util.COLDS_gui.ProjectData

    :param proj_wkt: Well-known text description of the project's spatial reference system.
    :type proj_wkt:  str

    :param queue:    Queue to be used to pass results back to main thread.
    :type queue:     multiprocessing.Queue

    :return:         None. Submits one-tuple of a ProjectData object to the queue upon completion.
    """

    # Set the Spatial Reference System for the point cloud metadata geodataframe.
    pc_wkt_list = data.gdf_pc_md.drop_duplicates(subset=['wkt'])['wkt'].to_numpy()

    # Create the path for the output geopackage
    out_file: Path = proj_file.with_suffix('.gpkg')

    # Reset the partial progress bar for the point cloud extents processing
    queue.put({
        'desc': ('Creating point cloud extents', 1),
        'pbar_size': (len(data.gdf_pc_md), 1)
    })
    processed = 0
    pc_gdf_list: list[gpd.GeoDataFrame] = []

    # Iterate through the different wkt descriptions of the spatial reference systems in the point cloud dataframe,
    # and set the SRS for the elements that match it.
    for wkt in pc_wkt_list:
        pc_gdf_list.append(data.gdf_pc_md[data.gdf_pc_md.wkt == wkt].copy())
        pc_gdf_list[-1].set_crs(crs=wkt)

        # If a subset of the dataframe has an SRS that does not match the project's SRS, reproject the data to the
        # correct SRS
        if wkt != proj_wkt:
            pc_gdf_list[-1].to_crs(crs=proj_wkt,
                                   inplace=True)

        # Emit the progress signals
        processed += len(pc_gdf_list[-1])
        queue.put({'progress': (processed, 1)})
        queue.put({'progress': (10 + int(20 * processed / (len(data.gdf_pc_md) + 1)), 0)})

    # Concatenate the results, and emit a signal indicating this step is done
    data.gdf_pc_md = pd.concat(pc_gdf_list, axis=0)
    queue.put({
        'progress': (30, 0),
        'message': 'Point cloud extents created'
    })

    # Save the point cloud extents to a geopackage that will hold all project layers.
    queue.put({'desc': ('Opening area of interest layers', 1)})
    data.gdf_pc_md.to_file(str(out_file),
                           layer='Point_Clouds',
                           crs=proj_wkt,
                           engine='fiona')
    queue.put({
        'msg': 'Point cloud extents saved to GeoPackage',
        'progress': (35, 0)
    })

    # Generate the Area of interest for the project.
    if 'AOI' not in data.gdf_vec_md.values:
        # If no Area of Interest vector layers have been provided, use the extents of the input point clouds.
        queue.put({
            'msg': '<b>WARNING: </b>No area of interest has been provided. Point cloud extents will be used as the area'
                   ' of interest.',
            'pbar_size': (1, 1)
        })

        data.gdf_aoi = data.gdf_pc_md.copy()

        # Rename the cloud type field to the layer type field for standardization of outputs
        data.gdf_aoi.rename(columns={'cloud_type': 'layer_type'},
                            inplace=True)
        queue.put({'progress': (50, 0)})
    else:
        # Prepare the partial progress bar for new signals
        filt: np.ndarray[bool] = data.gdf_vec_md.layer_type == 'AOI'
        queue.put({'pbar_size': (np.count_nonzero(filt), 1)})

        for i, aoi in data.gdf_vec_md[data.gdf_vec_md.layer_type == 'AOI'].iterrows():
            gdf_aoi: gpd.GeoDataFrame = gpd.read_file(aoi['filename'],
                                                      layer=aoi['name'])
            # Check the SRS of the input file, and set or reproject the data if required.
            file_crs: pyproj.CRS = gdf_aoi.crs
            if file_crs is None:
                queue.put({
                    'msg': f'<b>WARNING: </b>Layer {aoi["name"]} in file {aoi.filename} has no spatial reference system'
                           f' defined. Its SRS will be set to the current project SRS.'
                })
                gdf_aoi.set_crs(crs=proj_wkt,
                                inplace=True)
            elif file_crs.to_wkt() != proj_wkt:
                gdf_aoi.to_crs(crs=proj_wkt,
                               inplace=True)

            # Save the relevant information for the layers being added
            gdf_aoi['layer_type'] = aoi['layer_type']
            gdf_aoi = gdf_aoi[['layer_type', 'geometry']]

            # Concatenate the loaded file with the AOI GeoDataFrame and emit progress signals
            data.gdf_aoi = pd.concat([data.gdf_aoi, gdf_aoi], axis=0)
            queue.put({'progress': (i + 1, 1)})
            queue.put({'progress': (35 + int(15 * (i + 1) / np.count_nonzero(filt)), 0)})

    # Dissolve the AOI into a single shape to be used
    data.gdf_aoi = data.gdf_aoi.dissolve(by=['layer_type'])

    queue.put({
        'progress': (55, 0),
        'msg': 'Area of interest defined.'
    })

    # Save the area of interest layer to the project geopackage
    data.gdf_aoi.to_file(str(out_file),
                         layer='AOI',
                         crs=proj_wkt,
                         engine='fiona')
    queue.put({
        'progress': (60, 0),
        'msg': 'Area of interest saved to Geopackage.',
        'desc': ('Opening water feature layers.', 1)
    })

    # Load the water features for the project if they exist.
    if 'Water Feature' in data.gdf_vec_md.values:
        # Prepare the partial progress bar for new signals
        filt: np.ndarray[bool] = data.gdf_vec_md.layer_type == 'Water Feature'
        queue.put({'pbar_size': (np.count_nonzero(filt), 1)})

        for i, wf in data.gdf_vec_md[filt].iterrows():
            # Read the file. Using the fiona engine will compensate for SRS mismatches between the water feature
            # geometry and the area of interest bounding box.
            gdf_wf: gpd.GeoDataFrame = gpd.read_file(wf['filename'],
                                                     layer=wf['name'],
                                                     bbox=data.gdf_aoi.geometry.boundary,
                                                     engine='fiona')

            # Check the SRS of the input file, and set or reproject the data if required.
            file_crs: pyproj.CRS = gdf_wf.crs
            if file_crs is None:
                queue.put({
                    'msg': f'<b>WARING: </b>Layer {wf["name"]} in file {wf.filename} has no spatial reference system '
                           f'defined. Its SRS will be set to the current project SRS.'
                })
                gdf_wf.set_crs(crs=proj_wkt,
                               inplace=True)
            elif file_crs.to_wkt() != proj_wkt:
                gdf_wf.to_crs(crs=proj_wkt,
                              inplace=True)

            # If the water feature layer's geometry type is linestring or linestring z, buffer the features to
            # represent small bodies of water.
            if 'linestring' in wf['geometry_type'].lower():
                gdf_wf.geometry = gdf_wf.geometry.buffer(distance=2)

            # Add source information to the layer
            gdf_wf['filename'] = wf['filename']
            gdf_wf['source_layer'] = wf['name']

            # Concatenate the file with the Water Feature GeoDataFrame, and emit progress signals
            data.gdf_water = pd.concat([data.gdf_water, gdf_wf])
            queue.put({'progress': (i + 1, 1)})
            queue.put({'progress': (60 + int(20 * (i + 1) / np.count_nonzero(filt)), 0)})

        # Send final emits for the process
        queue.put({
            'progress': (80, 0),
            'msg': 'Water features identified.'
        })

        # Save the water features to the project geopackage
        if len(data.gdf_water) > 0:
            data.gdf_water.to_file(str(out_file),
                                   layer='Water_Features',
                                   crs=proj_wkt,
                                   engine='fiona')
            queue.put({'msg': 'Water features saved to Geopackage'})
        else:
            queue.put({'msg': 'No water features found in area of interest'})
    else:
        # If there are no water features found, emit required signals
        queue.put({
            'pbar_size': (1, 1),
            'msg': 'No water features found. Continuing.'
        })

    # Send the process progress emission
    queue.put({'progress': (85, 0)})

    # Save the vector metadata GeoDataFrame to the output file if there is any.
    if len(data.gdf_vec_md) > 0:
        data.gdf_vec_md.to_file(str(out_file),
                                layer='Vector_Metadata')

    queue.put({'result': (None, data)})


def read_point_cloud_chunk(
        filename: str,
        start_point: int,
        no_points: int,
        wkt_in: str,
        wkt_out: str
) -> np.ndarray:
    """
    Function that reads a selection of points from a las/laz point cloud file. Intended to be used within a
    multiprocessing pool. This script will also reproject points to a different spatial reference system if required.

    :param filename:    Path to the point cloud to be read.
    :type filename:     str

    :param start_point: Index of the first point in the file to be read.
    :type start_point:  int

    :param no_points:   Number of points to be read from the file.
    :type no_points:    int

    :param wkt_in:      Well-known text representation of the spatial reference system for the input file.
    :type wkt_in:       str

    :param wkt_out:     Well-known text representation of the desired output spatial reference system
    :type wkt_out:      str

    :return:            Numpy structured array of the selected points
    """
    # Construct the pipeline to read in a chunk of points.
    pl: pdal.Pipeline = pdal.Reader(filename=filename,
                                    start=start_point,
                                    count=no_points).pipeline()

    # If the input spatial reference system does not match the output spatial reference system, reproject the data.
    if wkt_in != wkt_out:
        pl |= pdal.Filter.reproject(out_srs=wkt_out)

    # Run the pipeline, and return the points array.
    _ = pl.execute()

    return pl.arrays[0]


def read_las_points_in_chunks(
        filename: str,
        no_points: int,
        chunk_size: int,
        wkt_in: str,
        wkt_out: str,
        bar_pos: int,
        no_cpus: int,
        queue: Queue
) -> np.ndarray:
    """
    Function that reads a las/laz point cloud file in chunks using a multiprocessing pool, and reports its results back
    to a multiprocessing queue.

    :param filename:   Path to las/laz point cloud file to be read.
    :type filename:    str

    :param no_points:  Number of points in the file to be read.
    :type no_points:   int

    :param chunk_size: Number of points in each chunk to be read.
    :type no_points:   int

    :param wkt_in:     Well-known text representation of the spatial reference system for the input file.
    :type wkt_in:      str

    :param wkt_out:    Well-known text representation of the desired output spatial reference system
    :type wkt_out:     str

    :param bar_pos:    Position of the progress bar in the progress bar widget to be updated.
    :type bar_pos:     int

    :param no_cpus:    Number of CPUs to be used for pool processing
    :type no_cpus:     int

    :param queue:      Multiprocessing queue in which to leave progress updates.
    :type queue:       multiprocessing.Queue

    :return:           Numpy structured array of the points in the file.
    """
    # Create a pool of chunks
    pool_queue = [i for i in range(0, no_points, chunk_size)]
    n_chunks = len(pool_queue)

    queue.put({
        'pbar_size': (n_chunks, bar_pos),
        'disp_perc': bar_pos,
        'desc': (f'Opening file {Path(filename).name}', bar_pos)
    })

    # Create the pool
    with pool.Pool(no_cpus) as p, Manager() as manager:
        # Create the counter that will be used to report progress
        chunk_counter = manager.Value('i', 0)

        # Create a list of the queued results
        queued_results: list[pool.AsyncResult] = [
            p.apply_async(
                read_point_cloud_chunk,
                args=(filename, start, chunk_size, wkt_in, wkt_out),
                callback=pool_progress(chunk_counter, bar_pos, queue)) for start in pool_queue]

        # Wait for all results to be completed
        for r in queued_results:
            r.wait()

        # Collect the results, and return them as a concatenated array
        results: list[np.ndarray] = [r.get()[['X', 'Y', 'Z', 'Classification']] for r in queued_results]

        return np.concatenate(results)


def write_las_process(
        pts_out: np.ndarray,
        filename: str,
        wkt_out: str,
        queue: Queue
):
    """
    Function that writes a collection of points to a las/laz file. Intended to be run in a Multiprocessing Process

    :param pts_out:  Points to be written to file.
    :type pts_out:   numpy.ndarray

    :param filename: Path to the file to be written.
    :type filename:  str

    :param wkt_out:  Well-known text representation of the spatial reference system for the output file.
    :type wkt_out:   str

    :param queue:    Multiprocessing queue in which to post the results of the write operation.
    :type queue:     multiprocessing.Queue

    :return:         None. Write time in seconds is placed on the queue.
    """
    # Record the time it takes to write the file
    start_time = time()
    # Write the points to file, re-projecting the data if necessary
    pl: pdal.Pipeline = pdal.Writer(filename=filename,
                                    a_srs=wkt_out,
                                    compression=True).pipeline(pts_out)

    # Execute the pipeline.
    _ = pl.execute_streaming()

    # Report completion to the queue.
    queue.put(time() - start_time)


def pool_progress(
        counter: Manager,
        bar_pos: int,
        queue: Queue
):
    """
    Convenience function for multiprocessing progress updating.

    :param counter: Multiprocessing manager containing a value.
    :type counter:  multiprocessing manager.

    :param bar_pos: Position of the progress bar to be updated.
    :type bar_pos:  int

    :param queue:   Multiprocessing queue to emit progress signals.
    :type queue:    multiprocessing.Queue

    :return:        None, updates the counter value in place, and emits appropriate signals.
    """
    counter.value += 1

    queue.put({'progress': (counter.value, bar_pos)})


def tile_and_merge(
        df: gpd.GeoDataFrame,
        aoi: Polygon,
        tile_size: float,
        proj_name: Path,
        epsg_list: list[str],
        no_cpus: int,
        queue: Queue
):
    """
    Function that splits the project's input point clouds into gridded tiles that encompass the area of interest.
    This function is run in a multiprocessing process to prevent the GUI from freezing during completion.

    :param df:        Dataframe containing the metadata for the input point cloud files.
    :type df:         geopandas.GeoDataFrame

    :param aoi:       Polygon representing the project's area of interest.
    :type aoi:        shapely.Polygon

    :param tile_size: Size of each tile to be created in metres.
    :type tile_size:  float

    :param proj_name: Path to the project file.
    :type proj_name:  pathlib.Path

    :param epsg_list: List of horizontal and vertical spatial reference systems for the project.
    :type epsg_list:  list[str]

    :param no_cpus:   Number of CPUs to be used for multiprocessing pools.
    :type no_cpus:    int

    :param queue:     Queue in which to put progress updates for the main GUI.
    :type queue:      multiprocessing.Queue

    :return:          None. Puts an updated point cloud dataframe on the queue for retrieval from the main GUI upon
                      completion.
    """
    # Create the wkt string for the output point clouds from the epsg list
    wkt_out = combine_epsg_codes(*epsg_list)

    # Use the bounding box extents of the area of interest, subdivide the area of interest into a grid.
    minx, miny, maxx, maxy = aoi.bounds

    # Starting from the northwest corner, tile the area moving east to west, and north to south
    grid: list[Polygon] = []
    y = maxy

    # Iterate over each dimension with a while condition.
    while y > miny:
        x = minx

        # Limit the tile size in the y direction when it extends past the area of interest
        y_tile_size = min(tile_size, (y - miny))

        while x < maxx:
            # Limit the tile size in the x direction when it extends past the area of interest
            x_tile_size = min(tile_size, (maxx - x))

            # Set the bounds of the grid based on the current (x, y) position and tile size
            x0 = x - 5
            y0 = y - y_tile_size - 5
            x1 = x + x_tile_size + 5
            y1 = y + 5

            # Create the box polygon given the bounds and add it to the list if it intersects with the original AOI.
            sq: Polygon = box(x0, y0, x1, y1)
            if sq.intersects(aoi):
                grid.append(sq)
            x += tile_size
        y -= tile_size

    # Determine the folder in which outputs will be saved
    proj_folder = proj_name.parent

    # Create a dataframe of the grid tiles
    gdf_grid: gpd.GeoDataFrame = gpd.GeoDataFrame(geometry=grid,
                                                  crs=df.crs)

    # Add the relevant metadata for the tile geodataframe.
    gdf_grid['Size (bytes)'] = 0
    gdf_grid['Read Time (s)'] = 0.0
    gdf_grid['cloud_type'] = 'Tile Cloud'
    gdf_grid['# Points'] = 0
    gdf_grid['min x'] = 0.0
    gdf_grid['max x'] = 0.0
    gdf_grid['min y'] = 0.0
    gdf_grid['max y'] = 0.0
    gdf_grid['Classified'] = 'False'
    gdf_grid['Horizontal EPSG'] = epsg_list[0]
    gdf_grid['Vertical EPSG'] = epsg_list[1]
    gdf_grid['wkt'] = wkt_out
    gdf_grid['name'] = 'Tile_' + (gdf_grid.index + 1).astype(str).str.zfill(3) + '.laz'

    # Retrieve the project folder from the project name
    gdf_grid['filename'] = f'{proj_folder.as_posix()}/las_files/' + gdf_grid['name']

    queue.put({
        'progress': (1, 1),
        'msg': 'Grid created'
    })

    # Iterate through the input clouds, separating them into the grid tiles.
    for i, pc in df.iterrows():
        # Keep track how long it takes to open the file.
        start_time = time()

        # Retrieve the points from the file
        pc_pts: np.ndarray = read_las_points_in_chunks(filename=pc['filename'],
                                                       no_points=pc['# Points'],
                                                       chunk_size=1_000_000,
                                                       wkt_in=pc['wkt'],
                                                       wkt_out=wkt_out,
                                                       bar_pos=1,
                                                       no_cpus=no_cpus,
                                                       queue=queue)

        df.loc[i, 'Read Time (s)'] += time() - start_time

        # Emit the overall process signal.
        queue.put({'progress': (2 * i + 1, 0)})

        # Determine which grid squares intersect with the current point cloud.
        tile_filt: pd.Series = gdf_grid.intersects(pc.geometry)
        gdf_intersect: gpd.GeoDataFrame = gdf_grid[tile_filt].reset_index()

        queue.put({
            'pbar_size': (len(gdf_intersect), 1),
            'desc': (f'Tiling file {pc["name"]}', 1)
        })

        # Iterate over the intersecting tiles, and write their points to file.
        for j, gd in gdf_intersect.iterrows():
            # Keep track of how long it takes to process the tile.
            start_time = time()

            # Determine which points lie within the tile
            minx, miny, maxx, maxy = gd.geometry.bounds
            inliers: np.ndarray = np.logical_and(
                np.logical_and(pc_pts['X'] >= minx, pc_pts['X'] <= maxx),
                np.logical_and(pc_pts['Y'] >= miny, pc_pts['Y'] <= maxy),
            )

            pts_out: np.ndarray = pc_pts[inliers]

            # If there are no points in the grid tile, emit signals and move on to the next
            if len(pts_out) == 0:
                queue.put({'progress': (j + 1, 1)})
                continue

            # If some tile points have already been written to file, read them into memory, and add them to pts_out
            out_file: Path = Path(gd['filename'])
            if out_file.is_file():
                # Read the points into memory, and add them to the output points
                existing_pts: np.ndarray = read_las_points_in_chunks(filename=gd['filename'],
                                                                     no_points=gd['# Points'],
                                                                     chunk_size=1_000_000,
                                                                     wkt_in=wkt_out,
                                                                     wkt_out=wkt_out,
                                                                     bar_pos=2,
                                                                     no_cpus=no_cpus,
                                                                     queue=queue)
                pts_out = np.concatenate([pts_out, existing_pts])

            # Write the points to file in a separate process, using a child queue to monitor progress.
            child_queue: Queue = Queue()
            write_process: Process = Process(target=write_las_process,
                                             args=(pts_out,
                                                   gd['filename'],
                                                   gd['wkt'],
                                                   child_queue))

            # Start the process
            write_process.start()

            # While the process is running collect statistics from the point cloud
            cur_cloud: np.ndarray = gdf_grid["name"] == gd["name"]
            gdf_grid.loc[cur_cloud, '# Points'] = len(pts_out)
            gdf_grid.loc[cur_cloud, 'min x'] = np.min(pts_out['X'])
            gdf_grid.loc[cur_cloud, 'max x'] = np.max(pts_out['X'])
            gdf_grid.loc[cur_cloud, 'min y'] = np.min(pts_out['Y'])
            gdf_grid.loc[cur_cloud, 'max y'] = np.max(pts_out['Y'])
            if 'Classification' in pts_out.dtype.names:
                gdf_grid.loc[cur_cloud, 'Classified'] = str(np.any(pts_out['Classification'] == 2))

            # Create a while loop that checks the queue and ensures the thread is still running.
            loop = True
            while loop:
                sleep(1)
                if not child_queue.empty():
                    break

                if not write_process.is_alive():
                    loop = False

            if not loop:
                queue.put({'error': f'Process to merge {gd["filename"]} crashed without providing a result.'})
                return

            # Get the time to split the input cloud, and add that information to the dataframe time
            _ = child_queue.get()
            gdf_grid.loc[cur_cloud, 'Read Time (s)'] += time() - start_time

            # Compute the file's size, and add it to the dataframe data
            gdf_grid.loc[cur_cloud, 'Size (bytes)'] = Path(gd['filename']).stat().st_size

            # Emit the signals for completion of the current grid square.
            queue.put({
                'progress': (j + 1, 1),
            })

            # Terminate the write process if it is still running.
            write_process.terminate()

            # Remove the points from memory before moving to the next tile.
            del pts_out
            gc.collect(2)

        # Remove the file points from memory and emit signals before moving to next file.
        del pc_pts
        gc.collect(2)

        queue.put({
            'progress': (2 * i + 2, 0),
            'msg': f'Completed tiling file: {pc["name"]}'
        })

    # Recompute the geometry of the tiles from their min and max values
    gdf_grid['geometry'] = gdf_grid.apply(lambda g: box(g['min x'], g['min y'], g['max x'], g['max y']), axis=1)

    # Concatenate the tile geodataframe to the original geodataframe, and submit the geodataframe to the queue.
    df = pd.concat([df, gdf_grid],
                   axis=0,
                   ignore_index=True)

    queue.put({'result': (df, None)})

    # Save the dataframe to file.
    vec_file_out: Path = proj_name.with_suffix('.gpkg')
    df.to_file(str(vec_file_out),
               layer='Point_Clouds',
               crs=wkt_out,
               engine='fiona')


def main():
    """
    Main program loop for COLDS.py

    :return: None
    """
    # Initialize the QGIS environment and the QApplication
    QgsApplication.setPrefixPath(prefixPath=os.getenv('QGIS_PREFIX_PATH'),
                                 useDefaultPaths=True)  # Defines the location of QGIS
    qgs = QgsApplication([], False)  # Start the application
    qgs.initQgis()  # Initialize QGIS
    qgs.setStyle('Fusion')  # Set the application style

    # Create and show the main window
    colds_window = Colds(qgs)
    colds_window.show()

    # Run the application
    qgs.exec()

    # Upon completion, exit QGIS
    qgs.exitQgis()


if __name__ == '__main__':
    ogr.UseExceptions()               # Can be removed when GDAL 4.0 is released.
    set_start_method('spawn')         # Explicitly defines multiprocessing process creation method.

    main()
