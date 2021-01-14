# This Python file uses the following encoding: utf-8
import sys
import os

from functools import partial
from qtpy.QtWidgets import QApplication
from qtpy.QtWidgets import QApplication, QDockWidget, QMainWindow
from PyQt5.QtCore import Qt
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

class MainWindow(QMainWindow):

    def __init__(self):
        """ Create the laser scanner window.
        """
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'ui_mainwindow.ui')

        # Load it
        super().__init__()
        uic.loadUi(ui_file, self)
        self.setStyleSheet(open('style/qdark.qss').read())
        self.show()

class Hyperspex:

    def __init__(self, data, x_range=None, y_range=None, wavelength_range=None):

        self._mw = MainWindow()
        self._spectrum_dock = SpectrumDock()
        self._image_dock = ImageDock()
        self._mw.addDockWidget(Qt.LeftDockWidgetArea, self._spectrum_dock)
        self._mw.addDockWidget(Qt.RightDockWidgetArea, self._image_dock)

        self.set_data(data)
        self.set_x_range(x_range)
        self.set_y_range(y_range)
        self.set_wavelength_range(wavelength_range)

        self.init_spectrum()
        #self.init_image()

        self.show()

    def show(self):
        """Make window visible and put it above all other windows.
        """
        QMainWindow.show(self._mw)
        self._mw.activateWindow()
        self._mw.raise_()

    def init_spectrum(self):

        self._wl_min_widget = ScienDSpinBox()
        self._wl_min_widget.setSuffix("m")
        self._wl_min_widget.setMinimumWidth(100)
        self._wl_min_widget.setMinimum(self._wavelength_range.min())
        self._wl_min_widget.setMaximum(self._wavelength_range.max())
        self._wl_min_widget.setValue(self._wavelength_range.min())
        self._spectrum_dock.wavelength_min.addWidget(self._wl_min_widget)

        self._wl_max_widget = ScienDSpinBox()
        self._wl_max_widget.setSuffix("m")
        self._wl_max_widget.setMinimumWidth(100)
        self._wl_max_widget.setMinimum(self._wavelength_range.min())
        self._wl_max_widget.setMaximum(self._wavelength_range.max())
        self._wl_max_widget.setValue(self._wavelength_range.max())
        self._spectrum_dock.wavelength_max.addWidget(self._wl_max_widget)

        self._spectral_roi = pg.LinearRegionItem(values=[0, 100], orientation=pg.LinearRegionItem.Vertical,
                                    brush="#fc03032f")
        self._spectrum_dock.spectrum.addItem(self._spectral_roi)

        self._spectrum_dock.spectrum.plot(self._wavelength_range, self.spectrum_data())

        self._spectral_roi.sigRegionChanged.connect(partial(self._update_spectral_roi, 0))
        self._wl_min_widget.valueChanged.connect(partial(self._update_spectral_roi, 1))
        self._wl_max_widget.valueChanged.connect(partial(self._update_spectral_roi, 1))

    def _update_spectral_roi(self, index):

        if index == 0:
            self.wl_min, self.wl_max = self._spectral_roi.getRegion()
            self._wl_min_widget.setValue(self.wl_min)
            self._wl_max_widget.setValue(self.wl_max)

        elif index == 1:
            self.wl_min, self.wl_max = self._wl_min_widget.value(), self._wl_max_widget.value()
            self._spectral_roi.setRegion([self.wl_min, self.wl_max])

    def get_spectrum_data(self):

        self._data[self.pos1[0]:]

    def init_image(self):

        self.my_colors = ColorScaleInferno()
        self._image = pg.ImageItem(image=self._data[:,:,0], axisOrder='row-major')
        self._image.setLookupTable(self.my_colors.lut)
        self._mw.image.addItem(self._image)
        self._colorbar = ColorbarWidget(self._image)
        self._mw.colorbar.addWidget(self._colorbar)

        self._image_roi = pg.ROI([0,0], [100,100])
        self._image_roi.addScaleHandle((1,0), (0,1))
        self._image_roi.addScaleHandle((0,1), (1,0))
        self._mw.image.addItem(self._image_roi)

    def set_data(self, data):

        self._data = np.array(data)

    def set_x_range(self, x_range):

        self._x_range = np.array(x_range)

    def set_y_range(self, y_range):

        self._y_range = np.array(y_range)

    def set_wavelength_range(self, wavelength_range):

        self._wavelength_range = np.array(wavelength_range)

    def manage_roi_center(self, index):

        if index == 0:
            self._center_wavelength = self._mw.spectrum_roi_center.value()
            self._spectrum_roi_center_spinbox.setValue(self._center_wavelength)

        if index == 1:
            self._center_wavelength = self._spectrum_roi_center_spinbox.value()
            self._mw.spectrum_roi_center.setSliderPosition(self._center_wavelength)

        vmin = np.where(np.isclose(self._wavelength_range, self._roi_center_wavelength - self._roi_wavelength_width/2))[0]
        vmax = np.where(np.isclose(self._wavelength_range, self._roi_center_wavelength + self._roi_wavelength_width/2))[0]
        self._spectrum_roi.setRegion([vmin, vmax])

    def manage_roi_width(self, index):

        if index == 0:
            value = self._mw.spectrum_roi_width.value()
            self._spectrum_roi_width_spinbox.setValue(value)
        elif index == 1:
            value = self._spectrum_roi_center_spinbox.value()
            self._mw.spectrum_roi_width.setSliderPosition(value)


if __name__ == "__main__":
    app = QApplication([])
    hyperspex = Hyperspex(np.random.random((200,200,200)), np.arange(0,100), np.arange(0,100), np.arange(0,100))
    sys.exit(app.exec_())
