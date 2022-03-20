import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import despike
import hyperspex.hyperspex as hx
from scipy import constants

class Spim:

    def __init__(self, data=np.zeros((10, 10, 10)), x=np.zeros(10), y=np.zeros(10), wavelength=np.ones(10)):

        self._data = data
        self._x = x
        self._y = y
        self._wavelength = wavelength
        self._energy = constants.c*constants.h/constants.e/wavelength

    def explore(self):
        
        self._hx = hx.Hyperspex(self._data, self._x, self._y, self._wavelength)
        hx.Hyperspex.start_app()

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
        self._data = spim.T
        self._x = x
        self._y = y
        self._wavelength = wavelength
        self._energy = constants.c * constants.h / constants.e / wavelength

    def averaged_image(self, val_min, val_max, energy=True):

        if val_max<val_min:
            val_min, val_max = val_max, val_min

        if energy:
            i_max = (np.abs(self._energy - val_min)).argmin()
            i_min = (np.abs(self._energy - val_max)).argmin()
        else:
            i_min = (np.abs(self._wavelength - val_min)).argmin()
            i_max = (np.abs(self._wavelength - val_max)).argmin()
        return self._data[:, :, i_min:i_max].mean(2)

    def local_spectrum(self, x_pos, y_pos):

        i_x = (np.abs(self._x - x_pos)).argmin()
        i_y = (np.abs(self._y - y_pos)).argmin()
        return self._data[i_x,i_y]

    def averaged_spectrum(self, x_min, x_max, y_min, y_max):

        i_min = (np.abs(self._x - x_min)).argmin()
        i_max = (np.abs(self._x - x_max)).argmin()
        j_min = (np.abs(self._y - y_min)).argmin()
        j_max = (np.abs(self._y - y_max)).argmin()
        return self._data[i_min:i_max,j_min:j_max].mean((0,1))
