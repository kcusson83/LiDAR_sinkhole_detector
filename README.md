# Cusson Open-source LiDAR Depression Scanner (COLDS)
A tool created to take an input of an airborne LiDAR dataset of an area, and automatically detect potential sinkholes in that geographic area using QGIS, GRASS, and PDAL.

# Installation Instructions
## Installing OSGEO Applications (QGIS/GRASS)
1. Download the OSGeo4W network installer from the website: https://trac.osgeo.org/osgeo4w/
2. Run the installer.

![Install_1](https://user-images.githubusercontent.com/95769776/235305337-8d5b46a2-b346-4e1c-844b-d0390180e029.jpg)

3. Select **`Advanced Install`**, and click **`Next`**.
4. Select **`Install from internet`**, and click **`Next`**.
5. Select your installation directory. It is recommended that you select the default path, but this path will be referred to as `{root directory}`. Click **`Next`**.
6. Select the location where temporary installation files will be installed, and the name of the Start Menu folder to place program links. Click **`Next`**.
7. Select your internet connection settings, and click **`Next`**.
8. Select https://download.osgeo.org as the download site, and click **`Next`**.
9. On the select packages screen, ensure that **`Curr`** is selected (for current).

![Install_2](https://user-images.githubusercontent.com/95769776/235305684-27732a78-b0e9-47ec-a6f9-3635766d893f.jpg)

10. **Do not** deselect any default packages that are selected. To select a package, click the arrows under the header **New** until the most recent version appears:
    - Under **Commandline_Utilities**, ensure the following packages are selected:
      - **gdal**: The GDAL/OGR library and commandline tools
      - **gdal312-runtime**: The GDAL/OGR 3.12 runtime library
      - **python3-core**: Python core interpreter and runtime
      - **python3-tools**: Python tools
      - **qt5-tools**: Qt5 tools (Development)
      - **setup**: OSGeo4W Installer/Updater
    - Under **Desktop**, ensure the following packages are selected:
      - **grass**: GRASS GIS 8.4 (required for GRASS processing functions)
      - **qgis-ltr**: QGIS Desktop (long term release)
      - **qgis-ltr-full**: QGIS Desktop Full (meta package; long term release)
      - **qgis-ltr-full-free**: QGIS Desktop (meta package; long term release)
      - **qt5-tools**: Qt5 tools (Development)
      - **saga**: SAGA (System for Automated Geographical Analyses)
    - Under **Libs**, ensure the following packages are selected:
      - **gdal**: The GDAL/OGR library and commandline tools
      - **pdal**: PDAL: Point Data Abstraction Library (Executable)
      - **pdal-libs**: PDAL: Point Data Abstraction Library (Runtime)
      - **python3-fiona**: Fiona reads and writes spatial data files
      - **python3-gdal**: The GDAL/OGR Python3 Bindings and Scripts
      - **python3-geopandas**: Geographic pandas extensions
      - **python3-numpy**: Fundamental package for array computing in Python
      - **python3-pandas**: Powerful data structures for data analysis, time series, and statistics
      - **python3-pdal**: Point cloud data processing Python API
      - **python3-pip**: The PyPA recommended tool for installing Python packages
      - **python3-pyogorio**: Vectorized spatial vector file format I/O using GDAL/OGR
      - **python3-pyproj**: Python interface to PROJ (cartographic projections and coordinate transformations library)
      - **python3-pyqt5**: Python bindigns for the Qt cross platform application toolkit
      - **python3-pyqt5-sip**: The sip module support for PyQt5
      - **python3-scipy**: Fundamental algorithms for scientific computing in Python
      - **qgis-ltr-common**: QGIS (common; long term release)
      - **qgis-ltr-grass-plugin**: GRASS plugin for QGIS (long term release)
11. Select **`Next`**.
12. Agree with any required license terms, and agree to install any required dependencies.
13. If any script errors are indicated, click **`Next`**. Once installation is complete, click **`Finish`**.

## Configuring Python
14. Download the repository files to a folder of your choosing. The batch script to run the program is set up for cloning the entire GitHub repository into the following path:
```
{root directory}\apps\qgis-ltr\python\plugins
```
15. If the repository is saved to a different location, modify the final line of **`python-COLDS.bat`** to include the path to the **`COLDS.py`** file as follows:
```
python path\to\COLDS.py%*
```
16. In file browser, copy both **`*.bat`** files to `{root directory}\bin`:
    - ***Notes:***
      - If you intend to run the code from an IDE (i.e. PyCharm), use the **python-qgis-ltr.bat** file as the python interpreter for your IDE. Geoprocessing tools may not work properly within your IDE.
      - *Anytime* a new package is installed with the OSGeo4W installer, the **python-qgis-ltr.bat** file from this package must be placed in the bin folder again.
17. To run the program, run **python-COLDS.bat**
---

# Sample Data

To run the process on the same data that was used to develop and test this product, the following datasets can be used within the program.

## Area of Interest

For the project AOI, use the shapefile provided in the [sample data folder](/SampleData).

## Water Features

For project water features, use the **Nova Scotia Topographic Database - Water Features** available on the [Nova Scotia Geographic Data Directory](https://nsgi.novascotia.ca/gdd/). Specifically, use the Shapefile(SHP) dataset stored in `NAD83(CSRS)v6, UTM Zone 20, CGVD2013`. Within the zip file, the following two sources are relevant:
- `WA_LINE_10K`
- `WA_POLY_10K`

## LiDAR Point Clouds

For project LiDAR point Clouds, use the **[Nova Scotia DataLocator - Elevation Explorer](https://nsgi.novascotia.ca/datalocator/elevation/)**. Ensure that in the layers tab on the left, only **LiDAR Tiles** are selected, to make navigation easier.

![Website](/img/Elevation%20Explorer.png)

- On the menu on the left side, click the **Search** tab and search by *Index*. Search for one of the following adjacent tiles:
  - `415_4980`
  - `416_4980`
  - `417_4980`
- On the menu on the left side, click on the **Selection Results** tab
- For each tile, click on the tile. The selection results will display the available LiDAR datasets. Select the data from 2018, adding it to the cart.
- Once all three tiles have been selected, click on the **Downloads** tab. Download each of the selected point clouds.
- 