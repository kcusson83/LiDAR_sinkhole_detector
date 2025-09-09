# =====================================================================================================================
# Filename:     COLDS_gui.py
# Written by:   Keith Cusson                Date: Aug 2025
# Description:  This script contains the graphical interface for the Cusson Open-source LiDAR Depression Scanner
# License:      MIT License (c) 2025 Keith Cusson
# =====================================================================================================================
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt5.QtGui import QCloseEvent, QColor, QContextMenuEvent, QFont, QPalette
from PyQt5.QtWidgets import *
from time import strftime

""" --------------------------------------------------------------------------------------------------------------------
Style Settings
 - Change as desired
---------------------------------------------------------------------------------------------------------------------"""
# Create the dark mode palette
dark_mode: QPalette = QPalette()
dark_mode.setColor(QPalette.Window, QColor(53, 53, 53))
dark_mode.setColor(QPalette.WindowText, Qt.white)
dark_mode.setColor(QPalette.Base, QColor(25, 25, 25))
dark_mode.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
dark_mode.setColor(QPalette.ToolTipBase, Qt.black)
dark_mode.setColor(QPalette.ToolTipText, Qt.white)
dark_mode.setColor(QPalette.Text, Qt.white)
dark_mode.setColor(QPalette.Button, QColor(53, 53, 53))
dark_mode.setColor(QPalette.ButtonText, Qt.white)
dark_mode.setColor(QPalette.BrightText, Qt.red)
dark_mode.setColor(QPalette.Link, QColor(42, 130, 218))
dark_mode.setColor(QPalette.Highlight, QColor(42, 130, 218))
dark_mode.setColor(QPalette.HighlightedText, Qt.black)

# Create font styles for required elements.
header_font: QFont = QFont('Arial', 14)
header_font.setBold(True)

button_font: QFont = QFont('Arial', 12)
button_font.setWeight(57)

table_data_font: QFont = QFont('Consolas', 11)
table_head_font: QFont = QFont('Arial', 11)
table_head_font.setWeight(63)

# Create the variables to be used to size the various elements of the application.
control_width = 225
# ----------------------------------------------------------------------------------------------------------------------


class ActionButtons(QWidget):
    """
    A widget containing the main group of action buttons for the application.
    """
    def __init__(
            self,
            parent: QMainWindow = None
    ):
        super().__init__(parent=parent)
        # Create the layout for the widget
        layout_buttons: QVBoxLayout = QVBoxLayout()

        # Add an invisible visual element that aligns the action buttons with the table to the left
        space_button: QPushButton = QPushButton(" ",
                                                enabled=False,
                                                font=button_font,
                                                flat=True)

        # Create the individual buttons
        self.button_tile: QPushButton = QPushButton(" Merge/Split Point Clouds ",
                                                    enabled=False,
                                                    font=button_font,
                                                    checkable=True,
                                                    clicked=parent.click_tile)

        self.button_classify: QPushButton = QPushButton(" Classify Point Clouds ",
                                                        enabled=False,
                                                        font=button_font,
                                                        checkable=True,
                                                        clicked=parent.click_classify)

        self.button_dem: QPushButton = QPushButton(" Generate DEM ",
                                                   enabled=False,
                                                   font=button_font,
                                                   checkable=True,
                                                   clicked=parent.click_dem)

        self.button_sinkholes: QPushButton = QPushButton(" Find Sinkholes ",
                                                         enabled=False,
                                                         font=button_font,
                                                         checkable=True,
                                                         clicked=parent.click_sinkholes)

        self.button_view: QPushButton = QPushButton(" View Results ",
                                                    enabled=False,
                                                    font=button_font,
                                                    clicked=parent.click_view)

        # Add the buttons to the layout
        layout_buttons.addWidget(space_button)
        layout_buttons.addWidget(self.button_tile)
        layout_buttons.addWidget(self.button_classify)
        layout_buttons.addWidget(self.button_dem)
        layout_buttons.addWidget(self.button_sinkholes)
        layout_buttons.addWidget(self.button_view)

        # Align the buttons with the top of the widget, and add the layout to the widget.
        layout_buttons.setAlignment(Qt.AlignTop)
        self.setLayout(layout_buttons)
        self.setMaximumWidth(control_width)


class ListSelectorDialog(QDialog):
    """
    This class creates a custom dialog box with a listbox selector.
    """
    def __init__(
            self,
            title: str,
            message_txt: str,
            vals: list[str],
            parent: QMainWindow = None
    ):
        """
        Initializes the List Selector Dialog

        :param title:       Title to be displayed on the window
        :type title:        str

        :param message_txt: Message to be displayed to the user.
        :type message_txt:  str

        :param vals:        Values to be displayed in the listbox selector
        :type vals:         list[str]

        :param parent:      Modal parent window for the dialog box
        :type parent:       PyQt5.QtWidgets.QMainWindow
        """
        super().__init__(parent)

        # Set the window title.
        self.setWindowTitle(title)

        # Create the elements that make up the dialog box.
        lbl_message: QLabel = QLabel(message_txt)

        self.list_box: QListWidget = QListWidget(self)
        self.list_box.setSelectionMode(QAbstractItemView.MultiSelection)
        self.list_box.addItems(vals)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.button_box: QDialogButtonBox = QDialogButtonBox(buttons)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Set the layout for the window
        layout: QVBoxLayout = QVBoxLayout()
        layout.addWidget(lbl_message)
        layout.addWidget(self.list_box)
        layout.addWidget(self.button_box)
        self.setLayout(layout)


class MessageFrame(QWidget):
    """
    A widget containing the message area for the main application.
    """
    def __init__(
            self,
            parent: QMainWindow = None
    ):
        super().__init__(parent=parent)
        layout_messages: QVBoxLayout = QVBoxLayout()

        # # Create the label for the text box
        lbl_msg: QLabel = QLabel('Messages')
        lbl_msg.setFont(header_font)
        lbl_msg.setAlignment(Qt.AlignHCenter)
        layout_messages.addWidget(lbl_msg)

        # Create the message box display area
        self.msg_box: QTextBrowser = QTextBrowser(self)
        self.msg_box.setAcceptRichText(True)
        self.msg_box.setUndoRedoEnabled(True)
        layout_messages.addWidget(self.msg_box)

        # Add the layout to the message
        self.setLayout(layout_messages)

    def print_message(
            self,
            msg
    ):
        """
        This method prints a message to the message box with a time stamp.

        :param msg: Message to be printed
        :type msg:  Any

        :return:    No return, prints a message to the message box.
        """
        msg_time = strftime('%Y-%m-%d %H:%M:%S')
        self.msg_box.append(f'<i>[{msg_time}]</i> - {str(msg)}')

    def update_last_line(
            self,
            msg
    ):
        """
        This method changes the last line in the message box.

        :param msg: New message to be printed

        :return:    No return, prints a message to the message box.
        """
        # Create a cursor to move to the beginning of the last line
        self.msg_box.undo()
        self.print_message(msg)


class MenuBar(QMenuBar):
    """
    This class create the application's menu bar
    """
    def __init__(
            self,
            parent: QMainWindow = None
    ):
        super().__init__(parent=parent)

        """ ------------------------------------------------------------------------------------------------------------
        File Menu
        ------------------------------------------------------------------------------------------------------------ """
        # --- Actions ---
        act_about: QAction = QAction('About',
                                     self,
                                     triggered=self.about)

        self.act_style: QAction = QAction('Dark Mode',
                                          self,
                                          triggered=self.change_palette)

        self.recent_menu: QMenu = QMenu('Recent Files',
                                        self)

        act_exit: QAction = QAction('Exit',
                                    self,
                                    triggered=parent.close)

        # --- Menus ---
        file_menu: QMenu = self.addMenu('&File')

        file_menu.addActions([act_about,
                              self.act_style])

        file_menu.addSeparator()

        file_menu.addMenu(self.recent_menu)

        file_menu.addSeparator()

        file_menu.addAction(act_exit)

        """ ------------------------------------------------------------------------------------------------------------
        Project Menu
        -------------------------------------------------------------------------------------------------------------"""
        # --- Actions ---
        act_new: QAction = QAction('New Project',
                                   self,
                                   triggered=parent.new)
        act_open: QAction = QAction('Open Project',
                                    self,
                                    triggered=parent.open)

        # --- Menus ---
        proj_menu: QMenu = self.addMenu('&Project')

        proj_menu.addActions([act_new,
                              act_open])

        """ ------------------------------------------------------------------------------------------------------------
        Add Menu
        ------------------------------------------------------------------------------------------------------------ """
        # --- Actions ---
        self.act_aoi: QAction = QAction('Area of Interest',
                                        self,
                                        triggered=parent.open_aoi,
                                        enabled=False)
        self.act_cloud: QAction = QAction('Point Cloud',
                                          self,
                                          triggered=parent.open_cloud,
                                          enabled=False)
        self.act_water: QAction = QAction('Water Features',
                                          self,
                                          triggered=parent.open_water,
                                          enabled=False)

        # --- Menus ---
        add_menu: QMenu = self.addMenu('&Add')

        add_menu.addActions([self.act_aoi,
                             self.act_cloud,
                             self.act_water])

        if parent.mode == 'Dark Mode':
            self.change_palette()
        self.update_recent()

        # TODO: Remove Test functionality
        bp: QAction = QAction('Breakpoint',
                                   self,
                                   triggered=parent.test1)
        test2: QAction = QAction('Test2',
                                      self,
                                      triggered=parent.test2)

        test_menu: QMenu = self.addMenu('&Test')
        test_menu.addActions([bp,
                              test2])

    def about(self):
        """
        This method displays details about the appliction.

        :return: None. Opens the "about" dialog box.
        """
        # Produce the formatted message to be displayed.
        about_msg = ('<b>Cusson Open-source LiDAR Depression Scanner (COLDS)</b><br>'
                     'Created by Keith Cusson<br>'
                     f'Version <b>{self.parent().__version__}</b><br>'
                     'MIT License &copy; 2025')

        # Create the dialog box and apply its settings
        msg_box: QMessageBox = QMessageBox()
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)

        # Apply the custom text to the window.
        msg_box.setWindowTitle('About')
        msg_box.setText(about_msg)

        # Run the dialog box
        msg_box.exec_()

    def change_palette(self):
        # Get the direction for the change
        win: MainWindow = self.parent()
        win.mode = self.act_style.text()

        # Set the menu text to the opposite selection
        menu_text = list({'Light Mode', 'Dark Mode'} - {win.mode})[0]

        # Apply the changes to the menu and the palette
        self.act_style.setText(menu_text)
        win.app.setPalette(win._palette[win.mode])

    def toggle_add_menu(
            self,
            enabled: bool
    ):
        """
        This method enables or disables all items in the add menu.

        :param enabled: Flag to enable/disable menu items.
        :type enabled:  bool

        :return:        None. Enables/Disables add menu actions
        """
        self.act_aoi.setEnabled(enabled)
        self.act_water.setEnabled(enabled)
        self.act_cloud.setEnabled(enabled)

    def update_recent(
            self,
            filename: str = None
    ):
        win: MainWindow = self.parent()
        # Update the recent_files list with the new filename if it exists:
        if filename is not None:
            if filename in win.recent_files:
                win.recent_files.pop(win.recent_files.index(filename))
            win.recent_files.insert(0, filename)

        # Clear the current files in the recent menu
        self.recent_menu.clear()

        # Iterate through the recent files, and add those actions to the recent menu.
        for file in win.recent_files[:3]:
            act_recent: QAction = QAction(file,
                                          self,
                                          triggered=win.open)
            self.recent_menu.addAction(act_recent)


class PandasTableModel(QAbstractTableModel):
    """
    This class provides a custom table model to instantiate the application's table viewer.
    """
    def __init__(
            self,
            df: pd.DataFrame
    ):
        super().__init__()
        self._data = df

    def data(
            self,
            index: QModelIndex,
            role: int
    ):
        """
        This method places data values in the cells of the table.

        :param index: Index of the cell to be filled.
        :type index:  PyQt5.QtCore.Qt.QModelIndex

        :param role:  Indicator for the intended use of the input data.
        :type role:   int

        :return:      Any data type to the cell value.
        """
        # Set the font for the data being displayed
        if role == Qt.FontRole:
            return table_data_font

        # Format the value being displayed
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if isinstance(value, np.integer):
                value = f'{value:>18,}'
            elif isinstance(value, np.floating):
                value = f'{value:>14,.3f}'
            return value
        elif role == Qt.ToolTipRole and index.column() == 1:
            return self._data.iloc[index.row(), 0]

    def rowCount(
            self,
            index: QModelIndex
    ) -> int:
        return self._data.shape[0]

    def columnCount(
            self,
            index: QModelIndex
    ) -> int:
        return  self._data.shape[1]

    def headerData(
            self,
            section: int,
            orientation: int,
            role: int
    ) -> str:
        """
        This method applies the headers to the table model rows and columns.

        :param section:     Index for the corresponding row/column.
        :type section:      int

        :param orientation: Indicator for row/column.
        :type orientation:  int

        :param role:        Indicator for intended use of the input data.
        :type role:         int

        :return:            str representing the title for the row/column
        """
        # Set the font for the header being displayed
        if role == Qt.FontRole:
            return table_head_font

        # Section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])

    def update_data(
            self,
            new_df: pd.DataFrame
    ) -> None:
        """
        This method refreshes the view with data from a new dataframe.

        :param new_df: New Data to be placed into the table.
        :type new_df:  pandas.DataFrame

        :return:       None. Updates table model.
        """
        self.beginResetModel()
        self._data = new_df
        self.endResetModel()


class StatsTable(QWidget):
    """
    This widget displays a table containing the statistics of loaded files with a combo box to select different types
    of files.
    """
    def __init__(
            self,
            parent: QMainWindow = None
    ):
        """
        This method initializes the widget containing the pandas table with metadata from loaded files.
        """

        super().__init__(parent=parent)
        layout_table: QGridLayout = QGridLayout()

        # Create the cloud type selector combo box
        self.combo_data_type: QComboBox = QComboBox(editable=False,
                                                    font=button_font,
                                                    activated=parent.change_type)
        self.combo_data_type.setMaximumWidth(control_width)
        layout_table.addWidget(self.combo_data_type, 0, 0)

        # Create a button that allows a user to accept all inputs
        self.btn_accept: QPushButton = QPushButton(' Finalize Inputs ',
                                                   font=button_font,
                                                   visible=False,
                                                   checkable=True,
                                                   clicked=parent.lock)
        self.btn_accept.setMaximumWidth(control_width)
        layout_table.addWidget(self.btn_accept, 0, 2)

        # Add the elements to the combobox, and disable them
        self.combo_items = ['Input Cloud', 'Vector File', 'Tile Cloud']
        self.combo_data_type.addItems(self.combo_items)
        self.combo_data_type.setCurrentIndex(-1)

        for i in range(0, len(self.combo_items)):
            self.combo_data_type.model().item(i).setEnabled(False)

        # Create the table that will hold the pandas data selected by the combo box
        self.table: QTableView = QTableView()
        self.table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.model: PandasTableModel = PandasTableModel(pd.DataFrame())
        self.table.setModel(self.model)
        self.model.modelReset.connect(self.handle_model_reset)

        # Add the table to the layout.
        layout_table.addWidget(self.table, 1, 0, 1, 3)

        self.setLayout(layout_table)

    def contextMenuEvent(
            self,
            a0: QContextMenuEvent
    ):
        # Determine the position of the click event
        table_vert_pos = self.table.pos().y()
        table_head_pos = self.table.horizontalHeader().size().height()
        index: QModelIndex = self.table.rowAt(a0.pos().y() - table_vert_pos - table_head_pos)

        # If the context action does not intersect with an item, end the event
        if index not in self.model._data.index or self.combo_data_type.currentIndex() not in range(2):
            return

        # Create the menu for the removal action
        menu: QMenu = QMenu()
        remove_act: QAction = menu.addAction('Remove')
        result = menu.exec_(a0.globalPos())

        # If the remove action is not selected, end the event
        if result != remove_act:
            return

        # Retrieve the main window handle, as well as the data for the selected row
        win: MainWindow = self.parent().parent()
        data: pd.Series = self.model._data.iloc[index, :]

        if self.combo_data_type.currentIndex() == 0:
            title = 'Remove Point Cloud'
            message = (f'Would you like to remove this point cloud from the project?<br>'
                       f'<b>File:</b><br>{data["filename"]}')
        elif self.combo_data_type.currentIndex() == 1:
            title = f'Remove {data["layer_type"]} Layer'
            message = (f'Would you like to remove this {data["layer_type"]} layer from the project?<br>'
                       f'<b>File:</b><br> {data["filename"]}<br>'
                       f'<b>Layer:</b><br> {data["name"]}')
        else:
            return

        reply: int = win.dlg_confirm(title,
                                     message)
        if reply == QMessageBox.Yes:
            win.remove_file(index)

    def enable_selection(
            self,
            index: int
    ) -> None:
        """
        This method enables an entry in the combo selector

        :param index: Entry to be enabled.
        :type index:  str

        :return:      No return, enables the entry
        """
        if index in range(len(self.combo_items)):
            self.combo_data_type.model().item(index).setEnabled(True)

    def handle_model_reset(self):
        """
        This method indicates the actions to be undertaken after the model has been reset.

        :return: None
        """
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setColumnHidden(0, True)


class MainWindow(QMainWindow):
    """
    This class represents the window making up the main interface for the COLDS graphical interface.
    """
    __version__ = '2.0a'
    program_name = 'Cusson Open-source LiDAR Depression Scanner'

    def __init__(
            self,
            app: QApplication = None
    ):
        super().__init__()
        # Set variables that are accessible to subclass methods of the MainWindow implementation.
        self.app = app
        self.on_close = []

        # Create the palettes for the various styles
        self._palette: dict[str, QPalette] = {'Light Mode': QPalette(),
                                              'Dark Mode': dark_mode}
        self.mode = 'Light Mode'
        self.recent_files = []

        # Check for configuration settings
        if Path('colds.cfg').exists():
            with open('colds.cfg') as f:
                config = f.readlines()
                self.mode = config[0].strip()
                self.recent_files = [file.strip() for file in config[1:]]

        # Set the Window title
        self.setWindowTitle(self.program_name)

        # Add the window's menu bar
        self.menu_bar: MenuBar = MenuBar(self)
        self.setMenuBar(self.menu_bar)

        # Create the widgets
        self.wgt_table: StatsTable = StatsTable(self)                # Displays all the statistics for the project files
        self.wgt_message: MessageFrame = MessageFrame()              # Displays system notices
        self.wgt_buttons: ActionButtons = ActionButtons(self)        # Displays function buttons

        # Set the window layout
        layout_page: QLayout = QGridLayout()

        layout_page.addWidget(self.wgt_table, 0, 0)
        layout_page.addWidget(self.wgt_message, 1, 0, 1, 2)
        layout_page.addWidget(self.wgt_buttons, 0, 1)

        # Create the container
        container: QWidget = QWidget()
        container.setLayout(layout_page)

        # Add the layout to the window, and prevent window resizing
        self.setCentralWidget(container)
        self.setMinimumSize(800, 600)

    """ ----------------------------------------------------------------------------------------------------------------
    SUPERSEDING METHODS
    """
    def closeEvent(
            self,
            a0: QCloseEvent
    ):
        # Set the accept value
        accept: bool = True

        # Display a confirmation dialog box
        reply = self.dlg_confirm('Quit',
                                 'Are you should you would like to quit? All your progress will be saved.')

        # Execute all functions that have been assigned to the on_close list
        if reply == QMessageBox.Yes:
            for func in self.on_close:
                result = func()
                if result is not None:
                    accept = accept and result
                if not accept:
                    break
        else:
            accept = False

        # Accept the close if all conditions are met
        if accept:
            out = [self.mode]
            out.extend(self.recent_files[:3])
            out = [line + '\n' for line in out]
            with open('colds.cfg', 'w') as f:
                f.writelines(out)
            a0.accept()
        else:
            a0.ignore()

    """
    ACTIONS
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Project Menu
    def new(self):
        """
        This method provides all the dialog boxes required when creating a new project

        :return: No return, stores values as a dictionary in class property user_input.
        """
        # Get a name for the project
        proj_name, ok1 = QInputDialog.getText(self,
                                              'Project Name',
                                              'What is the project name?')

        # Open the folder selection dialog box
        if ok1:
            folder_path = QFileDialog.getExistingDirectory(self,
                                                           "Select a directory")
            if folder_path:
                self.create_project(proj_name,
                                    folder_path)

    def create_project(
            self,
            name: str,
            proj_path: str
    ):
        """
        Method to be superseded in application for handling project creation.

        :param name:      Name of new project
        :type name:       str

        :param proj_path: Path to new project
        :type proj_path:  str

        :return:          None
        """
        pass

    def open(self):
        if self.sender().text() == 'Open Project':
            file_path, _ = QFileDialog.getOpenFileName(self,
                                                       "Open Project File",
                                                       "",
                                                       "QGIS Project Files (*.qgz)")
        else:
            file_path = self.sender().text()
        if file_path:
            self.open_project(file_path)

    def open_project(
            self,
            filename: str
    ):
        """
        Method to be superseded in application for handling project creation.

        :param filename: Path to project
        :type filename:  str

        :return:         None
        """
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # Add Menu
    def open_aoi(self):
        """
        This method provides a file dialog to open a vector file to be added as a project Area Of Interest.

        :return: No return, but stores a result in the class property dictionary user_input.
        """
        file_path, _ = QFileDialog.getOpenFileNames(self,
                                                    "Add Area of Interest",
                                                    "",
                                                    "Vector Files (*.gpkg;*.shp)")

        if len(file_path) > 0:
            self.add_vector('AOI', file_path)

    def open_cloud(self):
        """
        This method provides a file dialog to open a las file to be added to the project.

        :return: No return, but stores a result in the class property dictionary user_input.
        """
        file_path, _ = QFileDialog.getOpenFileNames(self,
                                                    "Add Point Cloud",
                                                    "",
                                                    "LAS/LAZ Point Cloud (*.las;*.laz)")
        if len(file_path) > 0:
            self.add_cloud(file_path)

    def open_water(self):
        """
        This method provides a file dialog to open a vector file to be added to the project as Water Features.

        :return: No return, but stores a result in the
        """
        file_path, _ = QFileDialog.getOpenFileNames(self,
                                                    "Add Water Features",
                                                    "",
                                                    "Vector Files (*.gpkg;*.shp)")

        if len(file_path) > 0:
            self.add_vector('Water Feature', file_path)

    def add_cloud(
            self,
            filename: list[str]
    ):
        """
        Method to be superseded in main application to handle adding point clouds to the project

        :param filename: Path(s) the point cloud(s) to be loaded.
        :type filename:  list[str]

        :return:         None
        """
        pass

    def add_vector(
            self,
            lyr_type: str,
            filename: list[str]
    ):
        """
        Method to be superseded in the main application to handle adding vector files to the project.

        :param lyr_type: Layer Type to be added.
        :type lyr_type:  str

        :param filename: Path(s) to the vector file(s) to be added.
        :type filename:  list[str]

        :return:         None
        """
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # Table View
    def change_type(self):
        """
        Method to be superseded in the Main application to update the display table after toggling type combo selector.

        :return: None
        """
        pass

    def remove_file(
            self,
            index: int
    ):
        """
        Method to be superseded in the Main application to remove a specif file from the table view.

        :param index: Index of the item to be removed.
        :type index:  int

        :return:      None
        """
        pass

    def lock(self):
        """
        Method to be superseded in the main application to prevent more additions to the input files.

        :return: None
        """
        print('Locked')

    def click_tile(self):
        self.wgt_message.msg_box.append('Tile Clouds')

    def click_classify(self):
        self.wgt_message.msg_box.append('Classify')

    def click_dem(self):
        self.wgt_message.msg_box.append('DEM')

    def click_sinkholes(self):
        self.wgt_message.msg_box.append('Sinkholes')

    def click_view(self):
        self.wgt_message.msg_box.append('View')

    """ ----------------------------------------------------------------------------------------------------------------
    DIALOG BOXES
    """
    def dlg_confirm(
            self,
            title: str,
            message: str
    ) -> int:
        """
        This method generates a dialog box to confirm a particular action, and returns the user selection.

        :param title:   Title for the dialog box
        :type title:    str

        :param message: Text to be displayed in the dialog box.
        :type message:  str

        :return:        int representing user selection.
        """
        # Create the message box, and configure its settings.
        msg_box: QMessageBox = QMessageBox(self)
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)

        # Apply the custom text to the window.
        msg_box.setWindowTitle(title)
        msg_box.setText(message)

        msg_box.exec_()

        return msg_box.result()

    def dlg_error(
            self,
            title: str,
            message: str
    ):
        result = QMessageBox.critical(self,
                                      f'{title} Error',
                                      message)

    def dlg_list(
            self,
            title: str,
            message: str,
            vals: list[str]
    ) -> list[str] | None:
        """
        This method generates a dialog box with a list of values, and returns the selected values.

        :param title:   Title for the dialog box
        :type title:    str

        :param message: Text to be displayed in the dialog box.
        :type message:  str

        :param vals:    List of values to be displayed in the dialog box.
        :type vals:     list[str]

        :return:        list[str] of the selected values or None
        """
        dlg: ListSelectorDialog = ListSelectorDialog(title,
                                                     message,
                                                     vals,
                                                     self)
        if dlg.exec():
            result = [val.text() for val in dlg.list_box.selectedItems()]
        else:
            result = None

        return result

    def dlg_warning(
            self,
            title: str,
            message: str
    ):
        result = QMessageBox.warning(self,
                                     f'{title} Warning',
                                     message)

    # TODO: REMOVE ALL AFTER THIS POINT
    def test1(self):
        pass

    def test2(self):
        pass
