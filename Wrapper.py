import sys
import os
import shutil

from src.GUI.ui_py.MainWindowUI import Ui_simulation_dp
from src.SimulationThread import SimulationThread
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib as mpl
from PIL import Image
from scipy.optimize import curve_fit
from scipy import constants
import pandas as pd
import numpy as np


def map_data_to_extension(data_value, relationship_df):
    if data_value < relationship_df['data'].min() or data_value > relationship_df['data'].max():
        return 0
    idx = (np.abs(relationship_df['data'] - data_value)).argmin()
    return relationship_df.iloc[idx]['extension']


# Define a function to fit, e.g., a polynomial
def fitting_function(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


# Define the fitted function using the parameters
def fitted_extension(data_value, params):
    return fitting_function(data_value, *params)


# Adjust the function to map data to extension and apply the constraints
def map_and_adjust_extension(data_value, relationship_df):
    if data_value < relationship_df['data'].min() or data_value > relationship_df['data'].max():
        return 0
    idx = (np.abs(relationship_df['data'] - data_value)).argmin()
    adjusted_extension = min(relationship_df.iloc[idx]['extension'] / 1000, 50)  # Divide by 1000 and cap at 45

    if adjusted_extension < -5:
        adjusted_extension = -5  # Replace values less than 0 with -2
    return adjusted_extension


class mainGUI(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        # setup GUI
        self.ui = Ui_simulation_dp()
        self.ui.setupUi(self)
        # set canvas
        # canvas for calibration result
        fig_cal = Figure()
        self.ui.figure_cal = FigureCanvas(fig_cal)
        self.ui.figure_cal.setParent(self.ui.w_figure_cali)
        self.ui.figure_cal.setGeometry(QtCore.QRect(QtCore.QPoint(0, 0), self.ui.w_figure_cali.size()))
        self.ui.figure_cal.axes = fig_cal.add_subplot(111)
        # canvas for T1 Map
        fig_T1 = Figure()
        self.ui.figure_T1 = FigureCanvas(fig_T1)
        self.ui.figure_T1.setParent(self.ui.w_figure_T1)
        self.ui.figure_T1.setGeometry(QtCore.QRect(QtCore.QPoint(0, 0), self.ui.w_figure_T1.size()))
        self.ui.figure_T1.axes = fig_T1.add_subplot(111)
        # canvas for Extension Map
        fig_extension = Figure()
        self.ui.figure_extension = FigureCanvas(fig_extension)
        self.ui.figure_extension.setParent(self.ui.w_figure_extension)
        self.ui.figure_extension.setGeometry(QtCore.QRect(QtCore.QPoint(0, 0), self.ui.w_figure_extension.size()))
        self.ui.figure_extension.axes = fig_extension.add_subplot(111)
        # canvas for Tension Map
        fig_tension = Figure()
        self.ui.figure_tension = FigureCanvas(fig_tension)
        self.ui.figure_tension.setParent(self.ui.w_figure_tension)
        self.ui.figure_tension.setGeometry(QtCore.QRect(QtCore.QPoint(0, 0), self.ui.w_figure_tension.size()))
        self.ui.figure_tension.axes = fig_tension.add_subplot(111)

        # init toolbox
        # for calibration result
        self.ui.figure_cal_toolbar = NavigationToolbar(self.ui.figure_cal, self.ui.w_toolbar_cali)
        self.ui.figure_cal_toolbar.setGeometry(QtCore.QRect(0, 0, self.ui.w_toolbar_cali.size().width(), 35))
        self.ui.figure_cal_toolbar.setParent(self.ui.w_toolbar_cali)
        # for T1 Map
        self.ui.figure_T1_toolbar = NavigationToolbar(self.ui.figure_T1, self.ui.w_toolbar_T1)
        self.ui.figure_T1_toolbar.setGeometry(QtCore.QRect(0, 0, self.ui.w_toolbar_T1.size().width(), 35))
        self.ui.figure_T1_toolbar.setParent(self.ui.w_toolbar_T1)
        # for Extension Map
        self.ui.figure_extension_toolbar = NavigationToolbar(self.ui.figure_extension, self.ui.w_toolbar_extension)
        self.ui.figure_extension_toolbar.setGeometry(QtCore.QRect(0, 0, self.ui.w_toolbar_extension.size().width(), 35))
        self.ui.figure_extension_toolbar.setParent(self.ui.w_toolbar_extension)
        # for Tension Map
        self.ui.figure_tension_toolbar = NavigationToolbar(self.ui.figure_tension, self.ui.w_toolbar_tension)
        self.ui.figure_tension_toolbar.setGeometry(QtCore.QRect(0, 0, self.ui.w_toolbar_tension.size().width(), 35))
        self.ui.figure_tension_toolbar.setParent(self.ui.w_toolbar_tension)
        # load settings
        self.load_settings()
        # connect push button
        self.connect_pb_methods()
        # connect thread
        self.sThread = SimulationThread()
        self.sThread.status_update.connect(self.update_progress_bar)
        self.sThread.time_update.connect(self.update_status_bar_time)
        self.sThread.simulation_result.connect(self.processing_simulation_result)

        # show previous result
        self.update_calib_result()

    @QtCore.pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def processing_simulation_result(self, x, average, std):
        self.ui.statusbar.showMessage('Simulation has finished!')
        self.ui.progressBar.setValue(100)
        self.save_settings()
        current_file_path = os.path.abspath("__file__")
        parent_directory = os.path.dirname(current_file_path)
        # read the parameters
        temp_dir = os.path.join(parent_directory, 'cache/temp_result/Parameters.txt')
        with open(temp_dir, 'r') as f_save_data:
            dic_parameters = {}
            for line in f_save_data.readlines():
                if line[-1] == '\n':
                    line = line[:-1]
                [key, value] = line.split('=')
                dic_parameters[key] = float(value)

        # get T1 average data
        x = x
        T1_average = average

        height_sim_start = dic_parameters.get('height_sim_start') * 10 ** 9  # nm
        height_minimum = dic_parameters.get('height_minimum') * 10 ** 9  # nm

        extension = np.array(x) - height_minimum

        extension_new = []
        T1_new = []
        for i in np.where(np.array(extension) > 0)[0]:
            extension_new.append(extension[i])
            T1_new.append(T1_average[i])
        extension_new = np.array(extension_new)

        extension_new1 = []
        T1_new1 = []
        for i in np.where(np.array(extension_new) < 6.0)[0]:
            extension_new1.append(extension_new[i])
            T1_new1.append(T1_new[i])
        extension_new1 = np.array(extension_new1)
        distance = extension_new1 + height_minimum

        p = 3.7 * 10 ** (-10)
        T = 300
        length_of_PEG = 0.318 * ((float(self.ui.leMolWei.text()) - 18.02)/44.05)

        gd_density_catch = dic_parameters.get('gd_density_catch')  # density of gd (1/nm^2)
        height_free = dic_parameters.get('height_free') * 10 ** 9
        balance_length = height_free - height_minimum

        ratio_x_Lc = extension_new1 / length_of_PEG

        F = np.array(
            (T * constants.k) * (0.25 / (1 - ratio_x_Lc) ** 2 - 0.25 + ratio_x_Lc) / p * 10 ** 12)  # Force in pN

        F_0 = (T * constants.k) * (
                0.25 / (1 - balance_length / length_of_PEG) ** 2 - 0.25 + balance_length / length_of_PEG) / p * 10 ** 12
        F_relative = np.array(F - F_0)

        F_norm = (F_relative * gd_density_catch) * (1 / 1000) * 1000000 * 1000

        # Save T1-Force relationship
        temp_dir = os.path.join(parent_directory, 'cache/temp_processed_result/T1_Force_relationship.txt')
        with open(temp_dir, 'w') as f_save_data:
            for each_line_num in range(np.size(T1_new1)):
                f_save_data.write(str(T1_new1[each_line_num]) + '\t' + str(F_norm[each_line_num]) + '\n')

        # Save T1-extend relationship
        temp_dir = os.path.join(parent_directory, 'cache/temp_processed_result/T1_Extension_relationship.txt')
        with open(temp_dir, 'w') as f_save_data:
            for each_line_num in range(np.size(T1_new1)):
                f_save_data.write(str(T1_new1[each_line_num]) + '\t' + str(extension_new1[each_line_num]) + '\n')

        # update figure
        self.update_calib_result()



    @QtCore.pyqtSlot()
    def connect_pb_methods(self):
        self.ui.pbSimParamLock.clicked.connect(self.sim_param_lock)
        self.ui.pbSensorInfoLock.clicked.connect(self.sensor_info_lock)
        self.ui.pbCali.clicked.connect(self.calibration)
        self.ui.pbInputData.clicked.connect(self.input_data)
        self.ui.pbSaveData.clicked.connect(self.save_data)

    @QtCore.pyqtSlot()
    def sim_param_lock(self):
        self.ui.sbSimParamStep.setEnabled(False)
        self.ui.sbSimParamRepeat.setEnabled(False)
        self.ui.pbSimParamLock.setText('Unlock')
        self.ui.pbSimParamLock.clicked.connect(self.sim_param_unlock)

    @QtCore.pyqtSlot()
    def sim_param_unlock(self):
        self.ui.sbSimParamStep.setEnabled(True)
        self.ui.sbSimParamRepeat.setEnabled(True)
        self.ui.pbSimParamLock.setText('Lock')
        self.ui.pbSimParamLock.clicked.connect(self.sim_param_lock)

    @QtCore.pyqtSlot()
    def sensor_info_lock(self):
        self.ui.leNVDepth.setEnabled(False)
        self.ui.leNVDensity.setEnabled(False)
        self.ui.leMolDensity.setEnabled(False)
        self.ui.leMolRad.setEnabled(False)
        self.ui.leMolWei.setEnabled(False)
        self.ui.pbSensorInfoLock.setText('Unlock')
        self.ui.pbSensorInfoLock.clicked.connect(self.sensor_info_unlock)

    def sensor_info_unlock(self):
        self.ui.leNVDepth.setEnabled(True)
        self.ui.leNVDensity.setEnabled(True)
        self.ui.leMolDensity.setEnabled(True)
        self.ui.leMolRad.setEnabled(True)
        self.ui.leMolWei.setEnabled(True)
        self.ui.pbSensorInfoLock.setText('Lock')
        self.ui.pbSensorInfoLock.clicked.connect(self.sensor_info_lock)

    @QtCore.pyqtSlot()
    def calibration(self):
        self.update_progress_bar(0)
        self.sThread.load_number_of_steps = int(self.ui.sbSimParamStep.value())
        self.sThread.load_number_of_repeats = int(self.ui.sbSimParamRepeat.value())
        self.sThread.load_NV_depth = float(self.ui.leNVDepth.text())
        self.sThread.load_NV_density = float(self.ui.leNVDensity.text())
        self.sThread.load_molecule_density = float(self.ui.leMolDensity.text())
        self.sThread.load_molecule_radius = float(self.ui.leMolRad.text())
        self.sThread.load_molecule_weight = float(self.ui.leMolWei.text())
        self.sThread.load_T1_bulk = float(self.ui.leT1Bulk.text())
        self.sThread.init_parameters()
        self.sThread.start()
        self.ui.statusbar.showMessage('Simulation started!')

    def load_settings(self):
        # Load the settings from the settings file
        sensor_settings_path = 'config/settings/sensor_calibration/default_sensor.txt'
        sensor_settings = []
        with open(sensor_settings_path, 'r') as f:
            for line in f.readlines():
                if line[-1] == '\n':
                    line = line[:-1]
                sensor_settings.append(line.split('=')[1])


        # Set the settings in the GUI
        self.ui.leNVDepth.setText(sensor_settings[0])
        self.ui.leNVDensity.setText(sensor_settings[1])
        self.ui.leMolDensity.setText(sensor_settings[2])
        self.ui.leMolRad.setText(sensor_settings[3])
        self.ui.leMolWei.setText(sensor_settings[4])

        simulation_settings_path = 'config/settings/simulation_parameter.txt'
        simulation_settings = []
        with open(simulation_settings_path, 'r') as f:
            for line in f.readlines():
                if line[-1] == '\n':
                    line = line[:-1]
                simulation_settings.append(line.split('=')[1])

        # Set the settings in the GUI
        self.ui.sbSimParamStep.setValue(int(simulation_settings[0]))
        self.ui.sbSimParamRepeat.setValue(int(simulation_settings[1]))

        baseline_settings_path = 'config/settings/baseline.txt'
        baseline_settings = []
        with open(baseline_settings_path, 'r') as f:
            for line in f.readlines():
                if line[-1] == '\n':
                    line = line[:-1]
                baseline_settings.append(line.split('=')[1])
        # Set the settings in the GUI
        self.ui.leT1Bulk.setText(baseline_settings[0])
        self.ui.leBaselineAir.setText(baseline_settings[1])
        self.ui.leBaselineLiq.setText(baseline_settings[2])

    def save_settings(self):
        # Save the settings to the settings file
        sensor_settings_path = 'config/settings/sensor_calibration/default_sensor.txt'
        sensor_settings = [self.ui.leNVDepth.text(), self.ui.leNVDensity.text(), self.ui.leMolDensity.text(),
                           self.ui.leMolRad.text(), self.ui.leMolWei.text()]
        with open(sensor_settings_path, 'w') as f:
            f.write('NV_DEPTH' + '=' + sensor_settings[0] + '\n')
            f.write('NV_DENSITY' + '=' + sensor_settings[1] + '\n')
            f.write('MOLECULE_DENSITY' + '=' + sensor_settings[2] + '\n')
            f.write('MOLECULE_RADIUS' + '=' + sensor_settings[3] + '\n')
            f.write('MOLECULE_WEIGHT' + '=' + sensor_settings[4] + '\n')

        simulation_settings_path = 'config/settings/simulation_parameter.txt'
        simulation_settings = [str(self.ui.sbSimParamStep.value()), str(self.ui.sbSimParamRepeat.value())]
        with open(simulation_settings_path, 'w') as f:
            f.write('NUMBER_OF_STEP' + '=' + simulation_settings[0] + '\n')
            f.write('NUMBER_OF_REPEAT' + '=' + simulation_settings[1] + '\n')

        baseline_settings_path = 'config/settings/baseline.txt'
        baseline_settings = [self.ui.leT1Bulk.text(), self.ui.leBaselineAir.text(), self.ui.leBaselineLiq.text()]
        with open(baseline_settings_path, 'w') as f:
            f.write('T1_BULK' + '=' + baseline_settings[0] + '\n')
            f.write('BASELINE_IN_AIR' + '=' + baseline_settings[1] + '\n')
            f.write('BASELINE_IN_LIQUID' + '=' + baseline_settings[2] + '\n')


    @QtCore.pyqtSlot()
    def input_data(self):
        self.ui.figure_T1.figure.clear()
        self.ui.figure_T1.axes = self.ui.figure_T1.figure.add_subplot(111)
        self.ui.figure_extension.figure.clear()
        self.ui.figure_extension.axes = self.ui.figure_extension.figure.add_subplot(111)
        self.ui.figure_tension.figure.clear()
        self.ui.figure_tension.axes = self.ui.figure_tension.figure.add_subplot(111)
        _directory = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Data', '', 'Excel (*.xlsx)')
        _directory = str(_directory[0])
        if _directory != '':
            try:
                self.data_t1 = pd.read_excel(_directory)

            except BaseException as e:
                self.ui.statusbar.showMessage(str(e) + '\t May select wrong file. ')
                return

            self.t1_data_processing(self.data_t1)
        else:
            self.ui.statusbar.showMessage('No file is selected!')
        # move back to the dir where this file at
        os.chdir(os.path.dirname(__file__))

    @QtCore.pyqtSlot(int)
    def update_progress_bar(self, progress):
        self.ui.progressBar.setValue(progress)

    @QtCore.pyqtSlot(int)
    def update_status_bar_time(self, time):
        self.ui.statusbar.showMessage('Simulation has run for ' + str(time) + 's')



    def t1_data_processing(self, data):
        # Convert the dataframe to a numpy array (excluding the first column)
        data_array = data.iloc[:, 1:].to_numpy()

        # Load the mask image
        mask_path = 'config/ROI/default_ROI.tif'
        mask_image = Image.open(mask_path).convert('L')

        # Resize the mask image to match the dimensions of the data
        mask_image_resized = mask_image.resize((data_array.shape[1], data_array.shape[0]))

        # Convert the resized mask image to a numpy array
        mask_array = np.array(mask_image_resized)

        # Normalize the mask to binary values (0 and 1)
        mask_array = (mask_array > 0).astype(int)

        # Apply the mask to the data
        masked_data = data_array * mask_array

        # Scale the data
        self.scaled_data_array = masked_data / 1000000

        self.mapColor = 'viridis'
        # See https://matplotlib.org/tutorials/colors/colormaps.html for colormap
        self.image_t1 = self.ui.figure_T1.axes.imshow(self.scaled_data_array, cmap=mpl.colormaps.get_cmap(self.mapColor), aspect='auto')
        # See https://matplotlib.org/gallery/images_contours_and_fields/interpolation_methods.html for interpolation
        self.cbar = self.ui.figure_T1.figure.colorbar(self.image_t1)
        self.ui.figure_T1.figure.tight_layout()
        self.ui.figure_T1.draw()

        # Load the T1-extension relationship data
        extension_relationship_path = 'cache/temp_processed_result/T1_Extension_relationship.txt'
        extension_relationship_data = pd.read_csv(extension_relationship_path, sep='\t', header=None, names=['data', 'extension'])

        # Apply the mapping to the scaled data array
        self.mapped_data_array_extension = np.array(
            [[map_data_to_extension(value, extension_relationship_data) for value in row] for row in (data_array / 1000000)])
        # Apply the mask to the mapped data
        self.masked_mapped_data_array_extension = self.mapped_data_array_extension * mask_array
        # plot extension map
        self.image_extension = self.ui.figure_extension.axes.imshow(self.masked_mapped_data_array_extension,
                                                      cmap=mpl.colormaps.get_cmap(self.mapColor), aspect='auto')
        # See https://matplotlib.org/gallery/images_contours_and_fields/interpolation_methods.html for interpolation
        self.cbar = self.ui.figure_extension.figure.colorbar(self.image_extension)
        self.ui.figure_extension.figure.tight_layout()
        self.ui.figure_extension.draw()

        # Load the T1-force relationship data
        force_relationship_path = 'cache/temp_processed_result/T1_Force_relationship.txt'
        force_relationship_data = pd.read_csv(force_relationship_path, sep='\t', header=None, names=['data', 'extension'])

        self.adjusted_mapped_data_array_tension = np.array([[map_and_adjust_extension(value, force_relationship_data) for value in row] for row in(data_array / 1000000)])

        # Apply the mask to the adjusted mapped data
        self.masked_adjusted_mapped_data_array_tension = self.adjusted_mapped_data_array_tension * mask_array
        # plot tension map
        self.image_tension = self.ui.figure_tension.axes.imshow(self.masked_adjusted_mapped_data_array_tension,
                                                                    cmap=mpl.colormaps.get_cmap(self.mapColor),
                                                                    aspect='auto')
        # See https://matplotlib.org/gallery/images_contours_and_fields/interpolation_methods.html for interpolation
        self.cbar = self.ui.figure_tension.figure.colorbar(self.image_tension)
        self.ui.figure_tension.figure.tight_layout()
        self.ui.figure_tension.draw()
    @QtCore.pyqtSlot()
    def save_data(self):
        folder_path_save = QtWidgets.QFileDialog.getExistingDirectory(self, 'Chose folder')
        shutil.copy('cache/temp_result/T1_average.txt', os.path.join(folder_path_save, 'T1_average.txt'))
        shutil.copy('cache/temp_result/T1_std.txt', os.path.join(folder_path_save, 'T1_std.txt'))
        shutil.copy('cache/temp_result/Parameters.txt', os.path.join(folder_path_save, 'Parameters.txt'))
        shutil.copy('cache/temp_processed_result/T1_Extension_relationship.txt',
                    os.path.join(folder_path_save, 'T1_Extension_relationship.txt'))
        shutil.copy('cache/temp_processed_result/T1_Force_relationship.txt',
                    os.path.join(folder_path_save, 'T1_Force_relationship.txt'))
        np.savetxt(os.path.join(folder_path_save,'T1_map.csv'), self.scaled_data_array, delimiter=',', fmt='%f')
        np.savetxt(os.path.join(folder_path_save,'Extension_map.csv'), self.masked_mapped_data_array_extension, delimiter=',', fmt='%f')
        np.savetxt(os.path.join(folder_path_save,'Tension_map.csv'), self.masked_adjusted_mapped_data_array_tension, delimiter=',', fmt='%f')


    def update_calib_result(self):
        # init raw_data plot
        self.ui.figure_cal.figure.clear()
        self.ui.figure_cal.axes = self.ui.figure_cal.figure.add_subplot(111)
        self.ui.figure_cal.axes.set_ylabel('T1 (ms)')
        self.ui.figure_cal.axes.set_title('Extension (nm)')

        # Load the data from the provided file
        file_path = 'cache/temp_processed_result/T1_Extension_relationship.txt'
        data = pd.read_csv(file_path, delimiter='\t', header=None, names=['T1', 'Extension'])
        self.ui.figure_cal.axes.plot(data['Extension'], data['T1'])
        # Plot the data
        self.ui.figure_cal.draw()

    @QtCore.pyqtSlot(QtCore.QEvent)
    def closeEvent(self, event):
        """
        When close the event, ask if to save the Defaults
        :param event:
        :return: None
        """
        quit_message = 'Quit Program?'
        reply = QtWidgets.QMessageBox.question(self, 'Message', quit_message, QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        if reply == QtWidgets.QMessageBox.Ok:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myWindow = mainGUI()
    myWindow.show()
    sys.exit(app.exec_())
