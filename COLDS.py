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
from COLDS_utils.COLDS_gui import ProjectData
import geopandas as gpd
import numpy as np
import os
from osgeo import gdal, ogr
import pandas as pd
from pathlib import Path
import pdal
import pyproj
from qgis.core import *                                       # Import QGIS functionality
import re
from time import time


# =====================================================================================================================
# CLASSES
# =====================================================================================================================
class ProjectSetup(gui.Worker):
    """
    A worker class to be used to finalize project inputs and used in a thread to prevent an unresponsive program.
    """
    def __init__(
            self,
            qgs_proj: QgsProject,
            data: ProjectData,
            epsg: list[str],
            parent: gui.MainWindow = None
    ):
        """
        Method that initializes the input finalization script

        :param qgs_proj: QGIS project instance to be modified.
        :type qgs_proj:  qgis.core.QgsProject

        :param data:     Project data object containing the data to be processed.
        :type data:      ProjectData

        :param epsg:     List of EPSG codes for the project.
        :type epsg:      list[str]

        :param parent:   Parent window for the process.
        :type parent:    COLDS_gui.MainWindow

        :return:         None, updates parent's qgs_proj and data variables at the end of the process.
        """
        super().__init__(parent=parent)
        self.qgs_proj: QgsProject = qgs_proj
        self.data: ProjectData = data
        self.epsg: list[str] = epsg

    def run(self):
        """
        Method that runs the project input finalization script, and emits progress signals.

        :return: None, updates parent's qgs_proj and data variables at the end of the process.
        """
        # Set the project Spatial Reference System with the loaded data.
        if not self.epsg[0] == '':
            crs_h: QgsCoordinateReferenceSystem = QgsCoordinateReferenceSystem(f'EPSG:{self.epsg[0]}')
            self.qgs_proj.setCrs(crs_h)
            self.message.emit(f'Horizontal Spatial Reference System set to EPSG:{self.epsg[0]}')
        if not self.epsg[1] == '':
            crs_v: QgsCoordinateReferenceSystem = QgsCoordinateReferenceSystem(f'EPSG:{self.epsg[1]}')
            self.qgs_proj.setVerticalCrs(crs_v)
            self.message.emit(f'Vertical Spatial Reference System set to EPSG:{self.epsg[1]}')
        proj_wkt = combine_epsg_codes(self.epsg[0], self.epsg[1])
        self.overall.emit(10)

        # Set the Spatial Reference System for the point cloud metadata geodataframe.
        pc_wkt_list = self.data.gdf_pc_md.drop_duplicates(subset=['wkt'])['wkt'].to_numpy()

        # Reset the partial progress bar for the point cloud extents processing
        self.desc.emit('Creating point cloud extents')
        self.partial_size.emit(len(self.data.gdf_pc_md))
        processed = 0
        pc_gdf_list: list[gpd.GeoDataFrame] = []

        # Iterate through the different wkt descriptions of the spatial reference systems in the point cloud dataframe,
        # and set the SRS for the elements that match it.
        for wkt in pc_wkt_list:
            pc_gdf_list.append(self.data.gdf_pc_md[self.data.gdf_pc_md.wkt == wkt].copy())
            pc_gdf_list[-1].set_crs(crs=wkt)

            # If a subset of the dataframe has an SRS that does not match the project's SRS, reproject the data to the
            # correct SRS
            if wkt != proj_wkt:
                pc_gdf_list[-1].to_crs(crs=proj_wkt,
                                       inplace=True)

            # Emit the progress signals
            processed += len(pc_gdf_list[-1])
            self.partial.emit(processed)
            self.overall.emit(10 + int(20 * processed / (len(self.data.gdf_pc_md) + 1)))

        # Concatenate the results, and emit a signal indicating this step is done
        self.data.gdf_pc_md = pd.concat(pc_gdf_list, axis=0)
        self.overall.emit(30)
        self.message.emit('Point cloud extents created')

        # Save the point cloud extents to a geopackage that will hold all project layers.
        out_file_name: Path = Path(self.qgs_proj.fileName()).with_suffix('.gpkg')

        self.desc.emit('Opening area of interest layers')
        self.data.gdf_pc_md.to_file(str(out_file_name),
                                    layer='Point_Clouds',
                                    crs=proj_wkt,
                                    engine='fiona')
        self.message.emit('Point cloud extents saved to GeoPackage')
        self.overall.emit(35)

        # Generate the Area of interest for the project.
        if 'AOI' not in self.data.gdf_vec_md.values:
            # If no Area of Interest vector layers have been provided, use the extents of the input point clouds.
            self.message.emit('<b>WARNING: </b>No area of interest has been provided. Point cloud extents will be used '
                              'as the area of interest.')
            self.partial_size.emit(1)
            self.data.gdf_aoi = self.data.gdf_pc_md.copy()

            # Rename the cloud type field to the layer type field for standardization of outputs
            self.data.gdf_aoi.rename(columns={'cloud_type': 'layer_type'},
                                     inplace=True)
            self.overall.emit(50)
        else:
            # Prepare the partial progress bar for new signals
            filt: np.ndarray[bool] = self.data.gdf_vec_md.layer_type == 'AOI'
            self.partial_size.emit(np.count_nonzero(filt))

            for i, aoi in self.data.gdf_vec_md[self.data.gdf_vec_md.layer_type == 'AOI'].iterrows():
                gdf_aoi: gpd.GeoDataFrame = gpd.read_file(aoi['filename'],
                                                          layer=aoi['name'])
                # Check the SRS of the input file, and set or reproject the data if required.
                file_crs: pyproj.CRS = gdf_aoi.crs
                if file_crs is None:
                    self.message.emit(f'<b>WARNING: </b>Layer {aoi["name"]} in file {aoi.filename} has no spatial '
                                      f'reference system defined. Its SRS will be set to the current project SRS.')
                    gdf_aoi.set_crs(crs=proj_wkt,
                                    inplace=True)
                elif file_crs.to_wkt() != proj_wkt:
                    gdf_aoi.to_crs(crs=proj_wkt,
                                   inplace=True)

                # Save the relevant information for the layers being added
                gdf_aoi['layer_type'] = aoi['layer_type']
                gdf_aoi = gdf_aoi[['layer_type', 'geometry']]

                # Concatenate the loaded file with the AOI GeoDataFrame and emit progress signals
                self.data.gdf_aoi = pd.concat([self.data.gdf_aoi, gdf_aoi], axis=0)
                self.partial.emit(i + 1)
                self.overall.emit(35 + int(15 * (i + 1) / np.count_nonzero(filt)))

        # Dissolve the AOI into a single shape to be used
        self.data.gdf_aoi = self.data.gdf_aoi.dissolve(by=['layer_type'])
        self.overall.emit(55)
        self.message.emit('Area of interest defined.')

        # Save the area of interest layer to the project geopackage
        self.data.gdf_aoi.to_file(str(out_file_name),
                                  layer='AOI',
                                  crs=proj_wkt,
                                  engine='fiona')
        self.message.emit('Area of interest saved to Geopackage')
        self.overall.emit(60)

        self.desc.emit('Opening water feature layers')
        # Load the water features for the project if they exist.
        if 'Water Feature' in self.data.gdf_vec_md.values:
            # Prepare the partial progress bar for new signals
            filt: np.ndarray[bool] = self.data.gdf_vec_md.layer_type == 'Water Feature'
            self.partial_size.emit(np.count_nonzero(filt))

            for i, wf in self.data.gdf_vec_md[filt].iterrows():
                # Read the file. Using the fiona engine will compensate for SRS mismatches between the water feature
                # geometry and the area of interest bounding box.
                gdf_wf: gpd.GeoDataFrame = gpd.read_file(wf['filename'],
                                                         layer=wf['name'],
                                                         bbox=self.data.gdf_aoi.geometry.boundary,
                                                         engine='fiona')

                # Check the SRS of the input file, and set or reproject the data if required.
                file_crs: pyproj.CRS = gdf_wf.crs
                if file_crs is None:
                    self.message.emit(f'<b>WARING: </b>Layer {wf["name"]} in file {wf.filename} has no spatial '
                                      f'reference system defined. Its SRS will be set to the current project SRS.')
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
                self.data.gdf_water = pd.concat([self.data.gdf_water, gdf_wf])
                self.partial.emit(i + 1)
                self.overall.emit(60 + int(20 * (i + 1) / np.count_nonzero(filt)))

            # Send final emits for the process
            self.overall.emit(80)
            self.message.emit('Water features identified.')

            # Save the water features to the project geopackage
            if len(self.data.gdf_water) > 0:
                self.data.gdf_water.to_file(str(out_file_name),
                                            layer='Water_Features',
                                            crs=proj_wkt,
                                            engine='fiona')
                self.message.emit('Water features saved to Geopackage')
            else:
                self.message.emit('No water features found in area of interest')
            self.overall.emit(85)
        else:
            # If there are no water features found, emit required signals
            self.partial_size.emit(1)
            self.message.emit('No water features found. Continuing.')
            self.overall.emit(85)

        # Add the layers to the QGIS project
        self.desc.emit('Saving layers to QGIS project')
        self.partial_size.emit(3)

        # Add water features to the QGIS project if they exist
        if len(self.data.gdf_water) > 0:
            qlyr = QgsVectorLayer(f'{str(out_file_name)}|layername=Water_Features',
                                  'Water Features',
                                  'ogr')
            qlyr_wf: QgsVectorLayer = self.qgs_proj.addMapLayer(qlyr)

            # Set the styling for the Water Feature layer
            qstyle = QgsStyle.defaultStyle().symbol('topo water')
            qrend = QgsSingleSymbolRenderer(qstyle)
            qlyr_wf.setRenderer(qrend)
            qlyr_wf.triggerRepaint()

            # Emit completion message
            self.message.emit('Water features layer added to project')

        # Emit relevant signals
        self.partial.emit(1)
        self.overall.emit(90)

        # Add the point cloud extents to the QGIS project file
        qlyr = QgsVectorLayer(f'{str(out_file_name)}|layername=Point_Clouds',
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

        # Emit relevant signals
        self.message.emit('Point cloud extents layer added to project')
        self.partial.emit(2)
        self.overall.emit(95)

        # Add the area of interest layer to the QGIS project file
        qlyr: QgsVectorLayer = QgsVectorLayer(f'{str(out_file_name)}|layername=AOI',
                                              'AOI',
                                              'ogr')
        qlyr_aoi: QgsVectorLayer = self.qgs_proj.addMapLayer(qlyr)

        # Set the styling for the AOI layer
        qstyle: QgsSymbol = QgsStyle.defaultStyle().symbol('outline red')
        qstyle.symbolLayer(0).setWidth(0.5)
        qrend: QgsFeatureRenderer = QgsSingleSymbolRenderer(qstyle)
        qlyr_aoi.setRenderer(qrend)
        qlyr_aoi.triggerRepaint()

        # Emit relevant signals
        self.message.emit('Area of interest layer added to project')
        self.partial.emit(3)
        self.overall.emit(99)

        # Change the state of the program, set the default view, and save the project file to disk.
        self.qgs_proj.setCustomVariables(self.data.export_project_state())
        view_settings: QgsProjectViewSettings = self.qgs_proj.viewSettings()
        extent: QgsReferencedRectangle = QgsReferencedRectangle(rectangle=qlyr_aoi.extent().buffered(50),
                                                                crs=self.qgs_proj.crs())
        view_settings.setDefaultViewExtent(extent)
        self.qgs_proj.write()

        # Save the vector metadata GeoDataFrame to the output file if there is any.
        if len(self.data.gdf_vec_md) > 0:
            self.data.gdf_vec_md.to_file(str(out_file_name),
                                         layer='Vector_Metadata')

        self.parent().data = self.data
        self.parent().qgs_proj = self.qgs_proj
        self.overall.emit(100)
        self.finished.emit()


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
                   'rasters']
        for f in folders:
            folder = Path(proj_path) / f / 'tiles'
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
            md_list.append(read_input_las_metadata(las_file, cloud_type))

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
                vec_layers = vec_layers[vec_layers.name.isin(selected_layers)]

            # If no layers have been identified, move to the next file
            if len(vec_layers) == 0:
                continue

            # Add the additional metadata required for later processing.
            vec_layers['filename'] = vec_file
            vec_layers['layer_type'] = lyr_type

            # Retrieve the spatial reference for each layer
            ds_vec: gdal.Dataset = ogr.Open(vec_file)
            vec_layers['srs'] = vec_layers.apply(
                lambda x: ds_vec.GetLayer(x['name']).GetSpatialRef(),
                axis=1
            )

            # Compute the EPSG code(s) for layer(s) that have a spatial reference.
            filt: np.ndarray[bool] = vec_layers.srs.isnull()
            vec_layers.loc[~filt, ['Horizontal EPSG', 'Vertical EPSG']] = vec_layers[~filt].apply(
                lambda x: epsg_code_wkt(x.srs.ExportToWkt()),
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

    def lock(self):
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

            results: list[str] = self.dlg_epsg_table('Select Coordinate System',
                                                     'Given the following coordinate reference systems from the input '
                                                     'files, select a horizontal and vertical coordinate system for '
                                                     'the project.<br>'
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
            results: np.ndarray = df_epsg.drop_duplicates(subset=columns)[columns].to_numpy()[0]
            results: list[str] = results.tolist()

        # Prepare the application for displaying progress bars and running the thread
        self.stack.setCurrentIndex(2)
        self.progress2.pbar_list[0].desc.setText('Finalizing Inputs')
        self.wgt_table.btn_accept.setVisible(False)
        self.menu_bar.toggle_add_menu(False)

        # Update the project state
        self.data.state = 1

        # Run the thread
        setup: ProjectSetup = ProjectSetup(self.qgs_proj,
                                           self.data,
                                           results,
                                           self)
        setup.finished.connect(
            lambda: self.wgt_buttons.enable_step_btn(self.data.export_project_state())
        )
        self.double_thread(setup)

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


def epsg_code_wkt(wkt: str) -> dict[str, str]:
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


def read_input_las_metadata(
        file_name: str,
        cloud_type: str
) -> dict:
    """
    This function reads in a LAS/LAZ file to retrieve the file's metadata. Read time is improved by reading only a
    single point.

    Sample Dictionary::

        {
            cloud_name:   'file.las',
            filename:     'D:/path/to/file.las',
            Size (bytes): 12458734
            cloud_type:   'Input Cloud',
            # Points:     123456789,
            min x:        415000.0,
            max x:        415999.9,
            min y:        4980000.0,
            max y:        4980999.9
            epsg_h:       '2961',
            epsg_v:       '6647',
            read_time:    0.27,
            extent:       'POLYGON ((415000.0 4980000.0,
                                     415999.9 4980000.0,
                                     415999.9 4980999.9,
                                     415000.0 4980999.9,
                                     415000.0 4980000.0))',
        }

    :param file_name:  Path to the LAS/LAZ file to be read.
    :type file_name:   str

    :param cloud_type: Type of cloud metadata to be stored. Either Input Cloud or Tile Cloud.
    :type cloud_type:  str

    :return:           Dictionary of relevant metadata.
    """
    # Create a pipeline to read the las file.
    pl: pdal.Pipeline = pdal.Reader(filename=file_name,
                                    count=1).pipeline()

    # Execute the pipeline and record how long it takes to read it.
    t_start = time()
    pts = pl.execute()
    md: dict = pl.metadata['metadata']['readers.las']

    out_dict = {
        'name': Path(file_name).name,
        'filename': file_name,
        'Size (bytes)': Path(file_name).stat().st_size,
        'cloud_type': cloud_type,
        '# Points': md['count'],
        'min x': float(md['minx']),
        'max x': float(md['maxx']),
        'min y': float(md['miny']),
        'max y': float(md['maxy'])
    }

    # Determine the SRS of the input las file.
    if 'compoundwkt' in md['srs']:
        out_dict |= epsg_code_wkt(md['srs']['compoundwkt'])
        out_dict['wkt'] = md['srs']['compoundwkt']
    else:
        if 'horizontal' in md['srs']:
            out_dict['Horizontal EPSG'] = epsg_code_wkt(md['srs']['horizontal'])['Horizontal EPSG']
            out_dict['wkt'] = md['srs']['horizontal']
        else:
            out_dict['Horizontal EPSG'] = ''
        if 'vertical' in md['srs']:
            out_dict['Vertical EPSG'] = epsg_code_wkt(md['srs']['vertical'])['Vertical EPSG']
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

    del pl

    return out_dict


def main():
    # Initialize the QGIS environment and the QApplication
    QgsApplication.setPrefixPath(prefixPath=os.getenv('QGIS_PREFIX_PATH'),
                                 useDefaultPaths=True)                                  # Defines the location of QGIS
    qgs = QgsApplication([], False)                                    # Start the application
    qgs.initQgis()                                                                      # Initialize QGIS
    qgs.setStyle('Fusion')                                                              # Set the application style

    # Create and show the main window
    colds_window = Colds(qgs)
    colds_window.show()

    # Run the application
    qgs.exec()

    # Upon completion, exit QGIS
    qgs.exitQgis()


if __name__ == '__main__':
    ogr.UseExceptions()  # Can be removed when GDAL 4.0 is released.

    main()
