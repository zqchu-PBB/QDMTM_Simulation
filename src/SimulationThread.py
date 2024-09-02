from PyQt5 import QtCore
import numpy as np
import scipy.constants as const
import random
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm


class SimulationThread(QtCore.QThread):
    SIGNAL_status_update = QtCore.pyqtSignal(int, name='status_update')
    SIGNAL_time_update = QtCore.pyqtSignal(int, name='time_update')
    SIGNAL_simulation_result = QtCore.pyqtSignal(np.ndarray, np.ndarray, np.ndarray, name='simulation_result')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.status_percentage = 0
        self.load_number_of_steps = 50
        self.load_number_of_repeats = 1
        self.load_NV_depth = 5
        self.load_NV_density = 0.001
        self.load_molecule_density = 0.018
        self.load_molecule_radius = 0.51
        self.load_molecule_weight = 1000
        self.load_T1_bulk = 900
        # Basic constant
        self.mu_b = const.physical_constants['Bohr magneton'][0]
        self.gama_gd = 10 ** 10
        self.D = 2.87 * (10 ** 9)
        # Defined constant
        self.S = 7 / 2
        self.a = self.load_molecule_radius * (10 ** (-9))  # molecule radius
        self.a_s = 1.4 * (10 ** (-10))  # molecule radius of the solution
        self.eta = 8.9 * (10 ** (-4))  # solution viscosity
        self.T = 300  # Temperature of the solution
        self.S_i = 7 / 2  # the S of the particle
        self.g_i = 2.017  # the g factor of the particle
        self.omega_0 = 2 * const.pi * self.D
        # Calculated constant result
        self.gama_i = self.g_i * self.mu_b / const.hbar  # the spin gyromagnetic ration of the particle
        self.Cs = self.S * (self.S + 1) / 3
         # defined  variables
        self.area_distribution = 1 * (10 ** 3)  # length of the distribution area(nm)
        self.area_calculation = 100  # length of the calculation area around the NV(nm)

    def init_parameters(self):
        # Defined constant
        self.a = self.load_molecule_radius * (10 ** (-9))  # molecule radius
        self.f_r = 1 / ((6 * self.a_s / self.a) + (1 + 3 * self.a_s / (self.a + 2 * self.a_s)) / (1 + 2 * self.a_s / self.a) ** 3)
        # defined  variables
        self.gd_density_all = self.load_molecule_density  # density of gd (1/nm^2)
        self.gd_density_catch = self.gd_density_all

        self.height_free = (((self.load_molecule_weight - 18.02)/44.05)**0.6) * 0.318 * (10 ** (-9)) + self.a  # default height of Gd molecule


        self.height_sim_start = 0.3 * (10 ** (-9))
        self.height_minimum = self.a  # minimum height of Gd (because of the molecule size)
        self.h = self.load_NV_depth * (10 ** (-9))  # The depth of the NV Center
        self.NV_density = self.load_NV_density  # The density of NV (1/nm^2)
        self.T1_bulk = self.load_T1_bulk * (10 ** (-6))  # T1 of bulk diamond in s



    def run(self):
        t0 = time.perf_counter()
        self.T1_final_result_all = []
        for repeat_index in range(self.load_number_of_repeats):
            # initialize
            self.free_Gd_location = []
            self.moving_Gd_location = []
            self.All_NV_location = []
            # Create Gd molecule
            self.make_Gd(density_gd=(self.gd_density_all - self.gd_density_catch), area=self.area_distribution, height_gd=self.height_free,
                    name=self.free_Gd_location)
            self.make_Gd(density_gd=self.gd_density_catch, area=self.area_distribution, height_gd=self.height_free, name=self.moving_Gd_location)
            self.free_Gd_location = np.array(self.free_Gd_location)
            self.moving_Gd_location = np.array(self.moving_Gd_location)
            # Create NV center
            self.make_NV(density_nv=self.NV_density, area=self.area_distribution, depth=-self.h, name=self.All_NV_location)
            self.All_NV_location = np.array(self.All_NV_location)

            # Select the NV in the light spot.
            self.valid_NV_location = []
            for i in range(np.size(self.All_NV_location, 0)):
                if abs(self.All_NV_location[i][0]) <= (300 * (10 ** (-9))) and abs(self.All_NV_location[i][1]) <= (
                        300 * (10 ** (-9))):
                    self.valid_NV_location.append(self.All_NV_location[i])
            self.valid_NV_location = np.array(self.valid_NV_location)

            # Select the Gd near NV centers
            self.valid_free_Gd_location = []
            self.valid_moving_Gd_location = []
            for i in range(np.size(self.valid_NV_location, 0)):  # for each NV, find Gd nearby
                self.valid_free_Gd_location.append([])
                self.valid_moving_Gd_location.append([])
                for j in range(np.size(self.free_Gd_location, 0)):  # Select Free Gd
                    if abs(self.free_Gd_location[j][0] - self.valid_NV_location[i][0]) <= (self.area_calculation / 2) * (10 ** (-9)) and \
                            abs(self.free_Gd_location[j][1] - self.valid_NV_location[i][1]) <= (self.area_calculation / 2) * (10 ** (-9)):
                        self.valid_free_Gd_location[i].append(self.free_Gd_location[j])
                for j in range(np.size(self.moving_Gd_location, 0)):  # Select Moving Gd
                    if abs(self.moving_Gd_location[j][0] - self.valid_NV_location[i][0]) <= (self.area_calculation / 2) * (10 ** (-9)) and \
                            abs(self.moving_Gd_location[j][1] - self.valid_NV_location[i][1]) <= (self.area_calculation / 2) * (
                            10 ** (-9)):
                        self.valid_moving_Gd_location[i].append(self.moving_Gd_location[j])
            # self.valid_free_Gd_location = np.array(self.valid_free_Gd_location)
            # self.valid_moving_Gd_location = np.array(self.valid_moving_Gd_location)

            # change the height for each Gd
            self.step = self.load_number_of_steps  # How many steps we need to calculate
            self.height_change = 12 * (10 ** (-9))  # How much height does the Gd need to change

            self.T1 = []
            self.xi = []

            for Valid_NV_index in range(np.size(self.valid_NV_location, 0)):

                self.T1.append([])
                self.xi.append([])
                height_temp = self.height_sim_start  # Start from the height_sim_start

                for moving_Gd_index in range(np.size(self.valid_moving_Gd_location[Valid_NV_index], 0)):
                    self.valid_moving_Gd_location[Valid_NV_index][moving_Gd_index][2] = height_temp

                for free_Gd_index in range(np.size(self.valid_free_Gd_location[Valid_NV_index], 0)):
                    self.valid_free_Gd_location[Valid_NV_index][free_Gd_index][2] = height_temp

                # for j in tqdm(range(step), desc='Change height'):
                for j in range(self.step):
                    if not self.valid_free_Gd_location[Valid_NV_index]:
                        merge_Gd_temp = self.valid_moving_Gd_location[Valid_NV_index]
                    else:
                        merge_Gd_temp = np.vstack((self.valid_free_Gd_location[Valid_NV_index],
                                                   self.valid_moving_Gd_location[Valid_NV_index]))
                    b_transverse = self.b_transverse_calculation(location_of_gd=merge_Gd_temp,
                                                                 location_of_nv=self.valid_NV_location[Valid_NV_index])
                    t_c = self.t_c_calculation(merge_Gd_temp)
                    GAMMA_1 = 1 / self.T1_bulk + (3 * (self.gama_i ** 2) * b_transverse * t_c) / (1 + (self.omega_0 ** 2) * (t_c ** 2))
                    T1_temp = (1 / GAMMA_1) * (10 ** 3)

                    self.T1[Valid_NV_index].append(T1_temp)
                    self.xi[Valid_NV_index].append(height_temp)

                    height_temp = height_temp + self.height_change / self.step
                    for moving_Gd_index in range(np.size(self.valid_moving_Gd_location[Valid_NV_index], 0)):
                        self.valid_moving_Gd_location[Valid_NV_index][moving_Gd_index][2] = height_temp
                    if height_temp >= self.height_free:
                        for free_Gd_index in range(np.size(self.valid_free_Gd_location[Valid_NV_index], 0)):
                            self.valid_free_Gd_location[Valid_NV_index][free_Gd_index][2] = self.height_free
                    else:
                        for free_Gd_index in range(np.size(self.valid_free_Gd_location[Valid_NV_index], 0)):
                            self.valid_free_Gd_location[Valid_NV_index][free_Gd_index][2] = height_temp

                self.status_update.emit(int(100 * (np.size(self.valid_NV_location, 0)*repeat_index+Valid_NV_index)
                                            /(np.size(self.valid_NV_location, 0)*self.load_number_of_repeats)))
                self.time_update.emit(int(time.perf_counter() - t0))

            # Calculate average T1
            self.T1_average = []
            for i in range(np.size(self.T1[0])):
                temp = 0
                for j in range(np.size(self.valid_NV_location, 0)):
                    temp = temp + self.T1[j][i]
                temp = temp / np.size(self.valid_NV_location, 0)
                self.T1_average.append(temp)

            self.T1_final_result_all.append(self.T1_average)
            self.x = np.array(self.xi[0]) * (10 ** 9)

        self.T1_final_result_average = np.average(self.T1_final_result_all, axis=0)
        self.T1_final_result_std = np.std(self.T1_final_result_all, axis=0)

        # store T1 Average
        current_file_path = os.path.abspath("__file__")
        parent_directory = os.path.dirname(current_file_path)


        temp_dir = os.path.join(parent_directory, 'cache/temp_result/T1_average.txt')
        with open(temp_dir, 'w') as f_save_data:
            for each_line_num in range(np.size(self.x)):
                f_save_data.write(str(self.x[each_line_num]) + '\t' + str(self.T1_final_result_average[each_line_num]) + '\n')
        temp_dir = os.path.join(parent_directory, 'cache/temp_result/T1_std.txt')
        with open(temp_dir, 'w') as f_save_data:
            for each_line_num in range(np.size(self.x)):
                f_save_data.write(
                    str(self.x[each_line_num]) + '\t' + str(self.T1_final_result_std[each_line_num]) + '\n')

        temp_dir = os.path.join(parent_directory, 'cache/temp_result/Parameters.txt')
        with open(temp_dir, 'w') as f_save_data:
            f_save_data.write(str('gd_density_all') + '=' + str(self.gd_density_all) + '\n')  # density of gd (1/nm^2)
            f_save_data.write(
                str('gd_density_catch') + '=' + str(self.gd_density_catch) + '\n')  # density of moving gd (1/nm^2)
            f_save_data.write(
                str('height_sim_start') + '=' + str(self.height_sim_start) + '\n')  # height of simulation start
            f_save_data.write(str('height_free') + '=' + str(self.height_free) + '\n')  # default height of Gd molecule
            f_save_data.write(str('height_minimum') + '=' + str(self.height_minimum) + '\n')  # minimum height of Gd (because of the molecule size)
            f_save_data.write(str('h') + '=' + str(self.h) + '\n')  # The depth of the NV Center
            f_save_data.write(str('NV_density') + '=' + str(self.NV_density) + '\n')  # The density of NV (1/nm^2)

        self.simulation_result.emit(np.array(self.x),
                                    np.array(self.T1_final_result_average),
                                    np.array(self.T1_final_result_std))


    # calculation of alpha_i
    def angle_calculate(self, vector, direction):
        """
        Calculate the angle of the NV center and Gd
        :param vector: list of the vectors
        :param direction: 0,1,2,3 represent for 4 different direction of the NV center
        :return:
        """
        angle = []
        for vec_index in range(np.size(vector, 0)):
            if direction == 0:
                # [-1, 1, 1]
                cos_temp = (-vector[vec_index][0] + vector[vec_index][1] + vector[vec_index][2]) / (
                            np.sqrt(3) * np.sqrt(
                        vector[vec_index][0] ** 2 + vector[vec_index][1] ** 2 + vector[vec_index][2] ** 2))
            elif direction == 1:
                # [1, -1, 1]
                cos_temp = (vector[vec_index][0] - vector[vec_index][1] + vector[vec_index][2]) / (np.sqrt(3) * np.sqrt(
                    vector[vec_index][0] ** 2 + vector[vec_index][1] ** 2 + vector[vec_index][2] ** 2))
            elif direction == 2:
                # [1, 1, -1]
                cos_temp = (vector[vec_index][0] + vector[vec_index][1] - vector[vec_index][2]) / (np.sqrt(3) * np.sqrt(
                    vector[vec_index][0] ** 2 + vector[vec_index][1] ** 2 + vector[vec_index][2] ** 2))
            elif direction == 3:
                # [-1, -1, -1]
                cos_temp = (-vector[vec_index][0] - vector[vec_index][1] - vector[vec_index][2]) / (
                            np.sqrt(3) * np.sqrt(
                        vector[vec_index][0] ** 2 + vector[vec_index][1] ** 2 + vector[vec_index][2] ** 2))
            else:
                cos_temp = 1
                print('check direction!')
            if cos_temp >= 1:
                cos_temp = 1
            if cos_temp <= -1:
                cos_temp = -1
            angle_i = np.arccos(cos_temp)
            angle.append(angle_i)
        angle = np.array(angle)
        return angle

    # calculation of B-transverse
    def b_transverse_calculation(self, location_of_gd, location_of_nv):  # location is an array/list store the location of Gd
        """
        Calculate the b_transverse based on the location of Gd and NV
        :param location_of_gd: list of Gd location
        :param location_of_nv: location of NV center(including direction)
        :return:
        """
        # Calculate the vector
        vector = []
        for gd_index in range(np.size(location_of_gd, 0)):
            vector.append(location_of_gd[gd_index] - location_of_nv[:-1])
        if not vector:
            return 0
        else:
            vector = np.array(vector)
            direction = location_of_nv[-1]
            alpha_i = self.angle_calculate(vector, direction)
            length = np.linalg.norm(vector, axis=1)

            c1 = (const.mu_0 * self.gama_i * const.hbar) / (4 * const.pi)
            last_term = (2 + 3 * (np.power(np.sin(alpha_i), 2))) / np.power(length, 6)
            b_transverse_temp = np.sum((c1 ** 2) * self.Cs * last_term)
            return b_transverse_temp

    # calculation R
    def t_c_calculation(self, location):
        """
        Give an array of the location of magnetic molecule, return the t_c
        :param location: list of the location of Gd molecules
        :return:
        """
        sum_temp = []
        for gd_location_index_1 in range(np.size(location, 0)):
            vector = []
            for gd_location_index_2 in range(np.size(location, 0)):
                if gd_location_index_2 != gd_location_index_1:
                    vector_temp = location[gd_location_index_1] - location[gd_location_index_2]
                    vector.append(vector_temp)
            if not vector:
                pass
            else:
                length = np.linalg.norm(vector, axis=1)
                sum_temp.append(np.sum(np.power(1 / length, 6)))
        sum_temp = np.array(sum_temp)

        r_dip = np.average(
            ((const.mu_0 * self.Cs * const.hbar * np.sqrt(6) * (self.gama_i ** 2)) / (4 * const.pi)) * np.sqrt(sum_temp))
        r_vib = 2.1 * (10 ** 9)
        r_rot = 0
        # print('\nr_dip:', r_dip, '\nr_vib:', r_vib, '\nr_rot:', r_rot)
        t_c_inside = 1 / (r_dip + r_rot + r_vib)
        return t_c_inside

    # Function to generate Gd location(input should be nm, out put is m)
    def make_Gd(self, density_gd, area, height_gd, name):
        """
        Create Gd molecules based on the setting parameters
        :param density_gd: density of gd (1/nm^2)
        :param area: length of the distribution area(nm)
        :param height_gd: default height of Gd molecule
        :param name: The depth of the NV Center
        :return:
        """
        if density_gd <= 0:
            name = []
        else:
            n = int((area ** 2) * density_gd + 1)
            for gd_index in range(n):
                x_temp = random.uniform(-area / 2, area / 2) * (10 ** (-9))
                y_temp = random.uniform(-area / 2, area / 2) * (10 ** (-9))
                name.append([x_temp, y_temp, height_gd])

    def make_NV(self, density_nv, area, depth, name):
        """
        Create NV based on the setting parameters
        :param density_nv: The density of NV (1/nm^2)
        :param area: length of the distribution area(nm)
        :param depth: The depth of the NV Center
        :param name:
        :return:
        """
        n = int((area ** 2) * density_nv + 1)
        for NV_index in range(n):
            x_temp = random.uniform(-area / 2, area / 2) * (10 ** (-9))
            y_temp = random.uniform(-area / 2, area / 2) * (10 ** (-9))
            direction = random.randint(0, 3)
            name.append([x_temp, y_temp, depth, direction])
