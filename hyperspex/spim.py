import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import despike
from scipy import constants

class Spim:

    def __init__(self, spim_values=None, x=None, y=None, wavelength=None):

        self._spim_values = spim_values
        self._x = x
        self._y = y
        self._wavelength = wavelength
        self._energy = constants.c*constants.h/constants.e/wavelength

    def show_slice(self, energy_min, energy_max):

        plt.show()

    def load_from_path(self, filepath, remove_spike=False):
        with np.load(filepath) as data:
            x = np.unique(data['x'])
            y = np.unique(data['y'])
            spim = np.zeros((np.array(data.files[3:]).size, y.size, x.size))
            i = 0
            for key in data.files[3:]:
                if remove_spike:
                    img = despike.clean(np.array(data[key]).reshape(y.size, x.size))
                else:
                    img = np.array(data[key]).reshape(y.size, x.size)
                spim[i] = img
                i += 1
            data.files = np.array([d.replace('nm', 'e-9') for d in data.files])
            wavelength = np.array(data.files[3:], dtype=float)
        self._spim_values = spim.T
        self._x = x
        self._y = y
        self._wavelength = wavelength

    def slice_spim(self, energy_min, energy_max):

        i_min = (np.abs(self._energy - energy_min)).argmin()
        i_max = (np.abs(self._energy - energy_max)).argmin()
        return SliceObject(self._spim_values[:,:,i_min:i_max], self._x, self._y)

    def integrate_area(self, energy_min, energy_max):

        i_min = (np.abs(self._energy - energy_min)).argmin()
        i_max = (np.abs(self._energy - energy_max)).argmin()
        return SliceObject(self._spim_values[:,:,i_min:i_max].mean(2), self._x, self._y)

    def point_spectrum(self, x_pos, y_pos):

        i_x = (np.abs(self._x - x_pos)).argmin()
        i_y = (np.abs(self._y - y_pos)).argmin()
        return self._spim_values[i_x,i_y]

    def area_spectrum(self, x_min, x_max, y_min, y_max):

        i_min = (np.abs(self._x - x_min)).argmin()
        i_max = (np.abs(self._x - x_max)).argmin()
        j_min = (np.abs(self._y - y_min)).argmin()
        j_max = (np.abs(self._y - y_max)).argmin()
        return self._spim_values[i_min:i_max,j_min:j_max].mean((0,1))

class SliceObject:

    def __init__(self, slice_values, x, y):

        self._slice_values = slice_values
        self._x = x
        self._y = y

    def show(self):

        plt.imshow(self._slice_values, extent=[self._x.min(), self._x.max(),self._y.min(), self._y.max()])

class SpectrumObject:

    def __init__(self, spectrum, energy):

        self._spectrum = spectrum
        self._energy = energy

    def plot(self):

        plt.plot(self._energy, self._spectrum)