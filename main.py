# This Python file uses the following encoding: utf-8
import sys
import os

from functools import partial
from qtpy.QtWidgets import QApplication
from qtpy.QtWidgets import QApplication, QDockWidget, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5.Qt import QRectF, QPoint
import pyqtgraph as pg
from qtpy import uic
from style.colordefs import ColorScaleInferno, ColorScaleMagma
from gui.colorbar.colorbar import ColorbarWidget
from gui.scientific_spinbox.scientific_spinbox import ScienDSpinBox

import numpy as np

class SpectrumDock(QDockWidget):

    def __init__(self):
        """ Create the laser scanner window.
        """
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'ui_spectrum_dock.ui')

        # Load it
        super().__init__()
        uic.loadUi(ui_file, self)

class ImageDock(QDockWidget):

    def __init__(self):
        """ Create the laser scanner window.
        """
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'ui_image_dock.ui')

        # Load it
        super().__init__()
        uic.loadUi(ui_file, self)

class Hyperspex:

    def __init__(self, data, x_range=None, y_range=None, wavelength_range=None):

        self._spectrum_dock = SpectrumDock()
        self._image_dock = ImageDock()

        self._data = data
        self._x_range = x_range
        self._y_range = y_range
        self._wavelength_range = wavelength_range

        self.init_spectrum()
        self.init_image()

        self.show()

    def show(self):
        """Make window visible and put it above all other windows.
        """
        QDockWidget.show(self._spectrum_dock)
        QDockWidget.show(self._image_dock)

    def init_spectrum(self):

        self.wl_min = self._wavelength_range.min()
        self.wl_max = self._wavelength_range.max()

        self._wl_min_widget = ScienDSpinBox()
        self._wl_min_widget.setSuffix("m")
        self._wl_min_widget.setMinimumWidth(100)
        self._wl_min_widget.setMinimum(self._wavelength_range.min())
        self._wl_min_widget.setMaximum(self._wavelength_range.max())
        self._wl_min_widget.setValue(self.wl_min)
        self._spectrum_dock.wavelength_min.addWidget(self._wl_min_widget)

        self._wl_max_widget = ScienDSpinBox()
        self._wl_max_widget.setSuffix("m")
        self._wl_max_widget.setMinimumWidth(100)
        self._wl_max_widget.setMinimum(self._wavelength_range.min())
        self._wl_max_widget.setMaximum(self._wavelength_range.max())
        self._wl_max_widget.setValue(self.wl_max)
        self._spectrum_dock.wavelength_max.addWidget(self._wl_max_widget)

        self._spectral_roi = pg.LinearRegionItem(values=[0, 100], orientation=pg.LinearRegionItem.Vertical,
                    brush="#fc03032f", bounds=[self._wavelength_range.min(), self._wavelength_range.max()])
        self._spectrum_dock.spectrum.addItem(self._spectral_roi)

        self._spectral_roi.sigRegionChanged.connect(partial(self._update_spectral_roi, 0))
        self._wl_min_widget.valueChanged.connect(partial(self._update_spectral_roi, 1))
        self._wl_max_widget.valueChanged.connect(partial(self._update_spectral_roi, 1))

    def init_image(self):

        self.pos1 = [0, self._x_range.size]
        self.pos2 = [0, self._y_range.size]

        self.my_colors = ColorScaleInferno()
        self._image = pg.ImageItem(image=self.get_image(), axisOrder='row-major')
        self._image.setLookupTable(self.my_colors.lut)
        self._image_dock.image.addItem(self._image)

        self._colorbar = ColorbarWidget(self._image)
        self._image_dock.colorbar.addWidget(self._colorbar)

        self._image_roi = pg.ROI([self.pos1[0], self.pos2[0]], [self.pos1[1], self.pos2[1]],
                                 maxBounds=QRectF(QPoint(0,0), QPoint(self._x_range.size, self._y_range.size)))
        self._image_roi.addScaleHandle((1,0), (0,1))
        self._image_roi.addScaleHandle((0,1), (1,0))
        self._image_dock.image.addItem(self._image_roi)

        self._image_roi.sigRegionChanged.connect(self._update_image_roi)

        self._spectrum_plot = self._spectrum_dock.spectrum.plot()
        self._spectrum_plot.setData(self._wavelength_range, self.get_spectrum())

    def _update_spectral_roi(self, index):

        if index == 0:
            self.wl_min, self.wl_max = self._spectral_roi.getRegion()
            self._wl_min_widget.setValue(self.wl_min)
            self._wl_max_widget.setValue(self.wl_max)

        elif index == 1:
            self.wl_min, self.wl_max = self._wl_min_widget.value(), self._wl_max_widget.value()
            self._spectral_roi.setRegion([self.wl_min, self.wl_max])

        self._image.setImage(self.get_image())

    def _update_image_roi(self):

        self.pos1, self.pos2 = self._image_roi.getArraySlice(self.get_image(), self._image, returnSlice=False)[0]

        self.pos1, self.pos2 = list(self.pos1), list(self.pos2)
        if self.pos1[0] > self.pos2[0]:
            self.pos1[0], self.pos2[0] = self.pos2[0], self.pos1[0]
        if self.pos1[1] > self.pos2[1]:
            self.pos1[1], self.pos2[1] = self.pos2[1], self.pos1[1]

        if self.pos1[0] == self.pos2[0]:
            if self.pos2[0]<self._data.shape[0]:
                self.pos2[0] += 1
            else:
                self.pos1[0] -= 1
        if self.pos1[1] == self.pos2[1]:
            if self.pos2[1] < self._data.shape[1]:
                self.pos2[1] += 1
            else:
                self.pos1[1] -= 1

        self._spectrum_plot.setData(self._wavelength_range, self.get_spectrum())

    def get_spectrum(self):

        spectrum = self._data[self.pos1[0]:self.pos2[0]+1,self.pos1[1]:self.pos2[1]+1].mean(axis=(0,1))
        return spectrum

    def get_image(self):

        i_min = self.wavelength_index(self.wl_min)
        i_max = self.wavelength_index(self.wl_max)
        image = self._data[:, :, i_min:i_max+1].mean(axis=2)
        return image

    def wavelength_index(self, wavelength):

        index = np.abs(self._wavelength_range - wavelength).argmin()
        return index

    def set_data(self, data):

        self._data = np.array(data)

    def set_x_range(self, x_range):

        self._x_range = np.array(x_range)

    def set_y_range(self, y_range):

        self._y_range = np.array(y_range)

    def set_wavelength_range(self, wavelength_range):

        self._wavelength_range = np.array(wavelength_range)

        self._wl_max_widget.setMinimum(self._wavelength_range.min())
        self._wl_max_widget.setMaximum(self._wavelength_range.max())
        self._wl_max_widget.setValue(self._wavelength_range.max())

        self._wl_min_widget.setMinimum(self._wavelength_range.min())
        self._wl_min_widget.setMaximum(self._wavelength_range.max())
        self._wl_min_widget.setValue(self._wavelength_range.min())

        self._spectral_roi.setBounds([self._wavelength_range.min(), self._wavelength_range.max()])

def load_spim(filepath):
    with np.load(filepath) as data:
        spim = []
        x = np.unique(data['x'])
        y = np.unique(data['y'])
        wavelength = np.array(data.files[3:], dtype=float)
        spim = []
        for key in data.files[3:]:
            spim.append(data[key])
    spim = np.array(spim).reshape(2048, y.size, x.size).T
    return {
        "x":x,
        "y":y,
        "wavelength":wavelength,
        "spim":spim,
    }

if __name__ == "__main__":
    app = QApplication([])
    dirpath = r"C:\Users\L2C_USER\Desktop\Samples\LPCNO\Experimental data\S2\20210113\spim_notebook\\"
    filepath = np.array([dirpath + filename for t in os.walk(dirpath) for filename in t[2] if filename[-3:] == 'npz'])
    spim_dict = load_spim(filepath[1])
    hyperspex = Hyperspex(spim_dict["spim"]-300, spim_dict["x"], spim_dict["y"], spim_dict["wavelength"])
    sys.exit(app.exec_())
