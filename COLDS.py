# =====================================================================================================================
# Filename:     COLDS.py
# Written by:   Keith Cusson                Date: Aug 2025
# Description:  This script contains the primary application for the Cusson Open-source LiDAR Depression Scanner
# License:      MIT License (c) 2025 Keith Cusson
# =====================================================================================================================
# ---------------------------------------------------------------------------------------------------------------------
# Import packages
# ---------------------------------------------------------------------------------------------------------------------
from COLDS_utils import COLDS_gui as gui, COLDS_func as CF
import geopandas as gpd
import numpy as np
import os
from osgeo import gdal, ogr
import pandas as pd
from pathlib import Path
import pdal
import pyproj
from qgis.core import *                                       # Import QGIS functionality
from qgis.PyQt.QtWidgets import QApplication
import re
from time import time


# =====================================================================================================================
# CLASSES
# =====================================================================================================================
# This class corresponds to project data
class ProjectData(object):
    """
    This class contains the project data for the COLDS application, and methods corresponding to its components.
    """
    # Project object constructor
    def __init__(self):
        # Create the geospatial variables for the project
        self.gdf_aoi: gpd.GeoDataFrame = None                 # GeoDataFrame for the project's Area of Interest
        self.gdf_water: gpd.GeoDataFrame = None               # GeoDataFrame for all project water features
        self.gdf_pc_md: gpd.GeoDataFrame = None               # GeoDataFrame for all point cloud metadata and extents
        self.df_vec_md: pd.DataFrame = None                   # DataFrame containing relevant metadata for vector inputs


# This class creates an instance of the gui with additional functionality
class Colds(gui.MainWindow):
    # Gui window constructor
    def __init__(
            self,
            app: QApplication = None):
        # Initalize the methods of the main window
        super().__init__(app)
        self.qgs_proj: QgsProject | None = None
        self.data: ProjectData = ProjectData()

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
        qgs_proj.setCustomVariables({'state': 0,
                                     'classified': False})

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
        self.qgs_proj = QgsProject.instance()
        result = self.qgs_proj.read(filename=filename)
        if result:
            self.qgs_proj.pathResolver()
        else:
            self.dlg_error('File Open',
                           'Invalid project file, please try again.')
            self.qgs_proj = None
            return
        self.menu_bar.toggle_add_menu(True)
        self.setWindowTitle(f'{self.program_name} - {self.qgs_proj.title()}')

        self.menu_bar.update_recent(filename)

    # ------------------------------------------------------------------------------------------------------------------
    # Add Menu
    def add_cloud(
            self,
            filename: list[str]
    ):
        """
        This method adds any point clouds loaded by selecting point clouds from the Add menu

        :param filename: Path to the list of files to be uploaded
        :type filename:  list[str]

        :return:         None
        """
        # Retrieve the metadata from the file(s) as a list of dictionaries
        md_list = []
        for las_file in filename:
            # Prior to reading the file, determine if it has already been added to the geodataframe.
            if self.data.gdf_pc_md is not None:
                if las_file in self.data.gdf_pc_md.values:
                    self.dlg_warning('File Exists',
                                     f'This file has already been loaded:\n{las_file}')
                    continue
            md_list.append(read_input_las_metadata(las_file))

        # Generate a geodataframe from the remaining files, and concatenate it with the existing frame.
        gdf_las: gpd.GeoDataFrame = gpd.GeoDataFrame(
            data=md_list,
            geometry=gpd.GeoSeries.from_wkt([md['extent'] for md in md_list]))

        if len(gdf_las) < 1:
            return

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
        vec_df_list: list[pd.DataFrame] = [self.data.df_vec_md]

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
            # breakpoint()
            vec_layers.loc[~filt, ['Horizontal EPSG', 'Vertical EPSG']] = vec_layers[~filt].apply(
                lambda x: epsg_code_wkt(x.srs.ExportToWkt()),
                axis=1,
                result_type='expand'
            )

            vec_df_list.append(vec_layers)

        # If the vector list only contains the original dataframe, do not make any changes
        if len(vec_df_list) == 1:
            return

        # Add the data to the vector metadata DataFrame.
        self.data.df_vec_md = pd.concat(vec_df_list,
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
                df_model = self.data.df_vec_md.loc[:, columns]

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
        # Determine what type of file you intend to remove
        if self.wgt_table.combo_data_type.currentIndex() == 0:
            self.data.gdf_pc_md.drop([index],
                                     inplace=True)
            self.data.gdf_pc_md.reset_index(inplace=True,
                                            drop=True)
            if len(self.data.gdf_pc_md) == 0:
                self.wgt_table.btn_accept.setVisible(False)
        elif self.wgt_table.combo_data_type.currentIndex() == 1:
            self.data.df_vec_md.drop([index],
                                     inplace=True)
            self.data.gdf_pc_md.reset_index(inplace=True,
                                            drop=True)
        else:
            return

        self.change_type()

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


if __name__ == '__main__':
    ogr.UseExceptions()  # Can be removed when GDAL 4.0 is released.

    # Initialize the QGIS environment and initialize processing
    QgsApplication.setPrefixPath(os.getenv('QGIS_PREFIX_PATH'), True)  # Defines the location of QGIS
    qgs = QgsApplication([], False)  # Start the application
    qgs.initQgis()  # Initialize QGIS

    # Activate the GUI and set the style for the application
    colds_app: QApplication = QApplication([])
    colds_app.setStyle('Fusion')

    # Create and show the main window
    colds_window = Colds(colds_app)
    colds_window.on_close.append(qgs.exitQgis)
    colds_window.show()

    # Run the application
    colds_app.exec()


def epsg_code_wkt(wkt: str) -> dict[str, str]:
    """
    This function returns the EPSG code from a Well-Known Text representation of a Coordinate Reference System. It is
    separated into horizontal and vertical components if applicable.

    :param wkt: Well-Known Text to be decoded.
    :type wkt:  str

    :return:    String representation of the EPSG code or None.
    """
    # Create the epsg dictionary that will be used to return the result.
    epsg = {'Horizontal EPSG': None,
            'Vertical EPSG': None}

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


def read_input_las_metadata(file_name: str) -> dict:
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

    :param file_name: Path to the LAS/LAZ file to be read.
    :type file_name:  str

    :return:          Dictionary of relevant metadata.
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
        'cloud_type': 'Input Cloud',
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
