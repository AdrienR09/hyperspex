# This Python file uses the following encoding: utf-8
import sys
import os

from functools import partial
from lmfit import Model, Parameters
from scipy.constants import c, h, e
import despike
import numpy as np

from qtpy.QtWidgets import QApplication, QMainWindow
import pyqtgraph as pg
from qtpy import uic
from qtpy import QtCore
from qtpy.QtCore import Qt, QRectF, QPoint

from .gui.style.colordefs import ColorScaleInferno, QudiPalette
from .gui.colorbar.colorbar import ColorbarWidget
from .gui.scientific_spinbox.scientific_spinbox import ScienDSpinBox

class ImageWindow(QMainWindow, QtCore.QObject):

    _sigUpdateSpectrum = QtCore.Signal()

    def __init__(self, data, x_range, y_range):
        """ Create the laser scanner window.
        """
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'gui/ui_image_window.ui')

        # Load it
        super(ImageWindow, self).__init__()
        uic.loadUi(ui_file, self)

        with open(os.path.join(this_dir, "gui/style/qdark.qss"), 'r') as stylesheetfile:
            stylesheet = stylesheetfile.read()
        self.setStyleSheet(stylesheet)

        self._data = data
        self._x_range = x_range
        self._y_range = y_range
        self._pos1 = (0, 0)
        self._pos2 = (self._data.shape[0], self._data.shape[1])
        self._i_min = 0
        self._i_max = self._data.shape[2]

        self.init_image()

    def init_image(self):

        self.my_colors = ColorScaleInferno()
        self._image = pg.ImageItem(image=self.image, axisOrder='row-major')
        self._image.setLookupTable(self.my_colors.lut)
        self.image_view.addItem(self._image)

        self._colorbar = ColorbarWidget(self._image)
        self.colorbar.addWidget(self._colorbar)

        self._image_roi = pg.ROI(self._pos1, self._pos2)
                                 #maxBounds=QRectF(QPoint(0, 0), QPoint(self._x_range.size, self._y_range.size)))
        self._image_roi.addScaleHandle((1, 0), (0, 1))
        self._image_roi.addScaleHandle((0, 1), (1, 0))
        self.image_view.addItem(self._image_roi)

        self._image_roi.sigRegionChanged.connect(self._update_roi)
        self.image_type.currentTextChanged.connect(self.plot)

    def _update_roi(self):

        (x1, x2), (y1, y2) = self._image_roi.getArraySlice(self.image, self._image, returnSlice=False)[0]

        if x1 == x2:
            if x2 < self._data.shape[0]:
                x2 += 1
            else:
                x1 -= 1
        if y1 == y2:
            if y2 < self._data.shape[1]:
                y2 += 1
            else:
                y1 -= 1

        self._pos1, self._pos2 = (x1, y1), (x2, y2)

        self._sigUpdateSpectrum.emit()

    def _update_image(self):
        self._i_min, self._i_max = self.sender().roi_index()
        self.plot()

    @property
    def image(self):
        image = self._data
        return image

    @property
    def roi_pos(self):
        return self._pos1, self._pos2

    def plot(self):
        self._image.setImage(self.image)

class SpectrumWindow(QMainWindow, QtCore.QObject):

    _sigUpdateAxisImage = QtCore.Signal()
    _sigPlot = QtCore.Signal()

    def __init__(self, data, x_range):
        """ Create the laser scanner window.
        """
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'gui/ui_spectrum_window.ui')

        # Load it
        super(SpectrumWindow, self).__init__()
        uic.loadUi(ui_file, self)

        with open(os.path.join(this_dir, "gui/style/qdark.qss"), 'r') as stylesheetfile:
            stylesheet = stylesheetfile.read()
        self.setStyleSheet(stylesheet)

        self._data = data
        self._x_range = x_range
        self._range_type = "energy"

        self._pos1 = (0, 0)
        self._pos2 = (self._data.shape[0], self._data.shape[1])

        self.init_spectrum()

    def init_spectrum(self):

        self._spectral_roi = np.array([self.range.min(), self.range.max()])

        self._roi_min = ScienDSpinBox()
        self._roi_min.setMinimumWidth(100)
        self.roi_layout.addWidget(self._roi_min)

        self._roi_max = ScienDSpinBox()
        self._roi_max.setMinimumWidth(100)
        self.roi_layout.addWidget(self._roi_max)

        self._roi_wg = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Vertical, brush="#fc03032f")
        self.spectrum_view.addItem(self._roi_wg)

        self.range_btn.setText("wavelength")

        self._update_roi_range()

        self._roi_wg.sigRegionChanged.connect(partial(self._update_roi, False))
        self._roi_min.valueChanged.connect(partial(self._update_roi, True))
        self._roi_max.valueChanged.connect(partial(self._update_roi, True))
        self.range_btn.clicked.connect(self._change_range_type)
        self.plot_btn.clicked.connect(self._sigPlot.emit)
        self.fit_btn.clicked.connect(self._sigUpdateImageFit.emit)

        self._plot = self.spectrum_view.plot()
        self.plot()

    def _update_roi_range(self):

        _min = self.range.min()
        _max = self.range.max()

        if _min > _max:
            _min, _max = _max, _min

        if self._range_type == "energy":
            self._roi_min.setSuffix("eV")
        else:
            self._roi_min.setSuffix("m")
        self._roi_min.setMinimum(_min)
        self._roi_min.setMaximum(_max)
        self._roi_min.setValue(self._spectral_roi.min())

        if self._range_type == "energy":
            self._roi_max.setSuffix("eV")
        else:
            self._roi_max.setSuffix("m")
        self._roi_max.setMinimum(_min)
        self._roi_max.setMaximum(_max)
        self._roi_max.setValue(self._spectral_roi.max())

        self._roi_wg.setBounds([_min, _max])
        self._roi_wg.setRegion([self._spectral_roi.min(), self._spectral_roi.max()])

    def _update_roi(self, spin_widget):

        if spin_widget:
            _min, _max = self._roi_min.value(), self._roi_max.value()
            self._spectral_roi[0] = _min
            self._spectral_roi[1] = _max
            self._roi_wg.setRegion([_min, _max])

        else:
            _min, _max = self._roi_wg.getRegion()
            self._spectral_roi[0] = _min
            self._spectral_roi[1] = _max
            self._roi_min.setValue(_min)
            self._roi_max.setValue(_max)

        self._sigUpdateImage.emit()

    @property
    def range(self):
        if self._range_type == "energy":
            return self._hyperspex._energy
        else:
            return self._hyperspex._wavelength

    @property
    def spectrum(self):
        spectrum = self._hyperspex._data[self._pos1[0]:self._pos2[0],self._pos1[1]:self._pos2[1]].mean(axis=(0, 1))
        return spectrum

    def _change_range_type(self):
        self._spectral_roi = c * h / e / self._spectral_roi[::-1]
        if self._range_type == "energy":
            self._range_type = "wavelength"
            self.range_btn.setText("energy")
        else:
            self._range_type = "energy"
            self.range_btn.setText("wavelength")
        self.plot()
        self._update_roi_range()

    def _update_spectrum(self):
        self._pos1, self._pos2 = self.sender().roi_pos()
        self.plot()

    def range_index(self, value):
        index = np.abs(self.range - value).argmin()
        return index

    def roi_index(self):
        i_min = self.range_index(self._spectral_roi[0])
        i_max = self.range_index(self._spectral_roi[1])
        if i_min > i_max:
            i_min, i_max = i_max, i_min
        return i_min, i_max

    def plot(self):
        self._plot.setData(self.range, self.spectrum)

class Hyperspex(QtCore.QObject):

    app = QApplication([])

    @classmethod
    def start_app(cls):
        sys.exit(cls.app.exec_())

    def __init__(self, data=np.zeros((100, 100, 100)), projection=None, **kwargs):
        
        if list(shape(data)) != [len(v) for k, v in kwargs.items()]:
            raise "Explorator object parameter data has a shape different than the axis parameters."

        self._data = data
        self._axis = {}
        for ax, ax_range in kwargs.items():
            self._axis[ax] = ax_range
            if ax == "wavelength":
                self._axis["energy"] = c * h / e / ax_range
            if ax == "energy":
                self._axis["wavelength"] = c * h / e / ax_range

        self._spectrum = SpectrumWindow(self)
        self._image = ImageWindow(self)

        self._image._sigUpdateSpectrum.connect(self._spectrum._update_spectrum)
        self._spectrum._sigUpdateImage.connect(self._image._update_image)
        self._spectrum._sigUpdateImageFit.connect(self._image._update_image_fit)

        QMainWindow.show(self._spectrum)
        QMainWindow.show(self._image)

    def load_from_path(self, filepath, remove_spike=False, remove_background=True):
        with np.load(filepath) as raw:
            x = np.unique(raw['x'])
            y = np.unique(raw['y'])
            data = np.zeros((np.array(raw.files[3:]).size, y.size, x.size))
            if remove_background:
                data -= data.min()
            i = 0
            for key in raw.files[3:]:
                if remove_spike:
                    img = despike.clean(np.array(raw[key]).reshape(y.size, x.size))
                else:
                    img = np.array(raw[key]).reshape(y.size, x.size)
                data[i] = img
                i += 1
            raw.files = np.array([d.replace('nm', 'e-9') for d in raw.files])
            wavelength = np.array(raw.files[3:], dtype=float)
        self._data = data.T
        self._x = x
        self._y = y
        self._wavelength = wavelength
        self._energy = c * h / e / wavelength

    def averaged_image(self, val_min, val_max, energy=True):

        if val_max < val_min:
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
        return self._data[i_x, i_y]

    def averaged_spectrum(self, x_min, x_max, y_min, y_max):

        i_min = (np.abs(self._x - x_min)).argmin()
        i_max = (np.abs(self._x - x_max)).argmin()
        j_min = (np.abs(self._y - y_min)).argmin()
        j_max = (np.abs(self._y - y_max)).argmin()
        return self._data[i_min:i_max, j_min:j_max].mean((0, 1))