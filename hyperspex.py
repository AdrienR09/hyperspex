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

from hyperspex.style.colordefs import ColorScaleInferno, QudiPalette
from hyperspex.gui.colorbar.colorbar import ColorbarWidget
from hyperspex.gui.scientific_spinbox.scientific_spinbox import ScienDSpinBox

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
        self._pos2 = (data.shape[0], data.shape[1])
        self._i_min = 0
        self._i_max = data.shape[2]
        self._image_fit = np.zeros((self._data.shape[0], self._data.shape[1], 4))

        self.init_image()

    def init_image(self):

        self.my_colors = ColorScaleInferno()
        self._image = pg.ImageItem(image=self.image, axisOrder='row-major')
        self._image.setLookupTable(self.my_colors.lut)
        self.image_view.addItem(self._image)

        self._colorbar = ColorbarWidget(self._image)
        self.colorbar.addWidget(self._colorbar)

        image_type = {'Image' : 'IMAGE',
                      'Fit - x0' : 'x0',
                      'Fit - w' : 'w',
                      'Fit - A' : 'A',
                      'Fit - B' : 'B'}
        for k, v in image_type.items():
            self.image_type.addItem(k, v)
            if v == 'IMAGE':
                self.image_type.setCurrentText(k)

        self._image_roi = pg.ROI(self._pos1, self._pos2)
                                 #maxBounds=QRectF(QPoint(0, 0), QPoint(self._x_range.size, self._y_range.size)))
        self._image_roi.addScaleHandle((1, 0), (0, 1))
        self._image_roi.addScaleHandle((0, 1), (1, 0))
        self.image_view.addItem(self._image_roi)

        self._image_roi.sigRegionChanged.connect(self._update_roi)
        self.image_type.currentTextChanged.connect(self.plot)

    def set_data(self, data):

        self._data = data
        self.plot()

    def _update_roi(self):

        (x1, x2), (y1, y2) = self._image_roi.getArraySlice(self.image, self._image, returnSlice=False)[0]

        if x1 == x2:
            if x2<self._data.shape[0]:
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

    def _update_image_fit(self):
        self._image_fit = self.sender().fit_image()
        self.plot()

    @property
    def image(self):
        image_type = self.image_type.currentData()
        if image_type == 'x0':
            image = self._image_fit[:, :, 0]
        elif image_type == 'w':
            image = self._image_fit[:, :, 1]
        elif image_type == 'A':
            image = self._image_fit[:, :, 2]
        elif image_type == 'B':
            image = self._image_fit[:, :, 3]
        else:
            image = self._data[:, :, self._i_min:self._i_max+1].mean(axis=2)
        return image

    def roi_pos(self):
        return self._pos1, self._pos2

    def plot(self):
        self._image.setImage(self.image)

class SuperImageWindow(ImageWindow):

    def __init__(self, data, x_range, y_range):

        self._pos1 = (0, 0)
        self._pos2 = (data.shape[0], data.shape[1])

        ImageWindow.__init__(self, data, x_range, y_range)

    @property
    def image(self):
        image = self._data[self._pos1[0]:self._pos2[0],self._pos1[1]:self._pos2[1]].mean(axis=(0, 1))
        return image

    def _update_image(self):
        self._pos1, self._pos2 = self.sender().roi_pos()
        self.plot()

class SpectrumWindow(QMainWindow, QtCore.QObject):

    _sigUpdateImage = QtCore.Signal()
    _sigUpdateImageFit = QtCore.Signal()

    def __init__(self, data, wavelength_range):
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
        self._data_fit = np.zeros(self._data.shape)
        self._wavelength_range = wavelength_range
        self._energy_range = c*h/e/self._wavelength_range
        self._range_type = "energy"
        self._fit_range = None

        self._pos1 = (0, 0)
        self._pos2 = (data.shape[0], data.shape[1])

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

        self._model_fit = Model(self.lorentzian)
        self._model_fit.nan_policy = 'omit'
        self.fit_Nmax.setMinimum(1)
        self.fit_Nmax.setMaximum(1e6)
        self.fit_Nmax.setValue(400)

        self.model_dict = {'Lorentzian': self.lorentzian,
                            'Gaussian': self.gaussian,
                            'Two Gaussian': self.two_gaussian}
        self.params_dict = {'Lorentzian': self.get_fit_params,
                            'Gaussian': self.get_fit_params,
                            'Two Gaussian': self.get_fit_params}

        for k, v in self.model_dict.items():
            self.fit_function.addItem(k, v)
            if v == 'LORENTZ':
                self.fit_function.setCurrentText(k)

        self._update_roi_range()

        self._roi_wg.sigRegionChanged.connect(partial(self._update_roi, False))
        self._roi_min.valueChanged.connect(partial(self._update_roi, True))
        self._roi_max.valueChanged.connect(partial(self._update_roi, True))
        self.range_btn.clicked.connect(self._change_range_type)
        self.fit_btn.clicked.connect(self._sigUpdateImageFit.emit)

        self._plot = self.spectrum_view.plot()
        self._plot_fit = self.spectrum_view.plot(pen=QudiPalette.c2)
        self.plot()

    def set_data(self, data):

        self._data = data
        self.plot()

    def _update_roi_range(self):

        _min = self.range.min()
        _max = self.range.max()

        if _min>_max:
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
            return self._energy_range
        else:
            return self._wavelength_range

    @property
    def range_fit(self):
        i_min, i_max = self._fit_range
        if self._range_type == "energy":
            return self._energy_range[i_min:i_max+1]
        else:
            return self._wavelength_range[i_min:i_max+1]

    @property
    def spectrum(self):
        spectrum = self._data[self._pos1[0]:self._pos2[0],self._pos1[1]:self._pos2[1]].mean(axis=(0, 1))
        return spectrum

    @property
    def spectrum_fit(self):
        spectrum_fit = self._data_fit[self._pos1[0]:self._pos2[0],self._pos1[1]:self._pos2[1]].mean(axis=(0, 1))
        return spectrum_fit

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
        if self._fit_range:
            self.plot_fit()
        self.plot()

    def range_index(self, value):
        index = np.abs(self.range - value).argmin()
        return index

    def roi_index(self):
        i_min = self.range_index(self._spectral_roi[0])
        i_max = self.range_index(self._spectral_roi[1])
        if i_min>i_max:
            i_min, i_max = i_max, i_min
        return i_min, i_max

    def plot(self):
        self._plot.setData(self.range, self.spectrum)

    def plot_fit(self):
        self._plot_fit.setData(self.range_fit, self.spectrum_fit)

    def add_model_fit(self, name, model_func, params_func):

        self.model_dict[name] = model_func
        self.params_dict[name] = params_func
        self.fit_function.addItem(name, model_func)

    def gaussian(self, x, peak_position, width, amplitude, background):
        return amplitude * np.exp(-((x - peak_position) / (2 * width) ) ** 2) + background

    def two_gaussian(self, x, peak_position_1, peak_position_2, width_1, width_2, amplitude_1, amplitude_2, background):
        gaussian_1 = amplitude_1 * np.exp(-((x - peak_position_1) / (2 * width_1) ) ** 2)
        gaussian_2 = amplitude_2 * np.exp(-((x - peak_position_2) / (2 * width_2) ) ** 2)
        return gaussian_1 + gaussian_2 + background

    def lorentzian(self, x, peak_position, width, amplitude, background):
        return amplitude * width / ( width**2 + (x - peak_position)**2 ) + background

    def _fit_spectrum(self, xdata, ydata, params):
        fit = self._model_fit.fit(ydata, params, x=xdata, max_nfev=self.fit_Nmax.value())
        return fit

    def fit_image(self):

        data_shape = self._data.shape
        i_min, i_max = self.roi_index()
        self._fit_range = (i_min, i_max)
        xdata = self.range[i_min:i_max+1]
        fit_image = np.zeros((data_shape[0], data_shape[1], 4))
        self._data_fit = np.zeros((data_shape[0], data_shape[1], i_max - i_min + 1))

        self._model_fit = Model(self.fit_function.currentData())
        params_func = self.params_dict[self.fit_function.currentText()]
        self._model_fit.nan_policy = 'omit'

        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                ydata = self._data[i, j, i_min:i_max+1]
                params = params_func(xdata, ydata)
                fit = self._fit_spectrum(xdata, ydata, params)
                self._data_fit[i, j, :] = fit.best_fit
                fit_image[i, j, :] = np.array(list(fit.best_values.values()))
        self.plot_fit()
        return fit_image

    def get_fit_params(self, xdata, ydata):

        params = Parameters()
        params.add('peak_position', min=xdata.min(), max=xdata.max(), value=xdata[(ydata - ydata.max()).argmin()])
        params.add('width', min = np.abs(xdata.max()-xdata.min())*1e-4,
                   value = np.abs(xdata.max()-xdata.min())/2)
        params.add('amplitude', min=0, value=ydata.max())
        params.add('background', min = 0, value=ydata.min())

        return params

class Hyperspex(QtCore.QObject):

    app = QApplication([])

    @classmethod
    def start_app(cls):
        sys.exit(cls.app.exec_())

    def __init__(self, data, x_range, y_range, wavelength_range, remove_background=True):

        if remove_background:
            self._data = data - data.min()
        else:
            self._data = data
        self._x_range = x_range
        self._y_range = y_range
        self._wavelength_range = wavelength_range

        self._spectrum = SpectrumWindow(self._data, wavelength_range)
        self._image = ImageWindow(self._data, x_range, y_range)

        self._image._sigUpdateSpectrum.connect(self._spectrum._update_spectrum)
        self._spectrum._sigUpdateImage.connect(self._image._update_image)
        self._spectrum._sigUpdateImageFit.connect(self._image._update_image_fit)

        self.show()

    def show(self):
        """Make window visible and put it above all other windows.
        """
        QMainWindow.show(self._spectrum)
        QMainWindow.show(self._image)

    def remove_background(self, background):

        self._data = self._data - background
        self._image.set_data(self._data)
        self._spectrum.set_data(self._data)

class Ultraspex(QtCore.QObject):

    app = QApplication([])

    @classmethod
    def start_app(cls):
        sys.exit(cls.app.exec_())

    def __init__(self, data, x_range, y_range, z_range, wavelength_range, remove_background=True):

        if remove_background:
            self._data = data - data.min()
        else:
            self._data = data
        self._x_range = x_range
        self._y_range = y_range
        self._z_range = z_range
        self._wavelength_range = wavelength_range

        self._image1 = SuperImageWindow(self._data, x_range, y_range)
        self._image2 = SuperImageWindow(np.transpose(self._data, (2, 3, 0, 1)), z_range, wavelength_range)

        self._image1._sigUpdateSpectrum.connect(self._image2._update_image)
        self._image2._sigUpdateSpectrum.connect(self._image1._update_image)

        self.show()

    def show(self):
        """Make window visible and put it above all other windows.
        """
        QMainWindow.show(self._image1)
        QMainWindow.show(self._image2)

    def remove_background(self, background):

        self._data = self._data - background
        self._image1.set_data(self._data)
        self._image2.set_data(self._data)

def load_spim(filepath, remove_spike=False):
    with np.load(filepath) as data:
        x = np.unique(data['x'])
        y = np.unique(data['y'])
        spim = np.zeros((np.array(data.files[3:]).size, y.size, x.size))
        i=0
        for key in data.files[3:]:
            if remove_spike:
                img = despike.clean(np.array(data[key]).reshape(y.size, x.size))
            else:
                img = np.array(data[key]).reshape(y.size, x.size)
            spim[i] = img
            i+=1
        data.files = np.array([d.replace('nm', 'e-9') for d in data.files])
        wavelength = np.array(data.files[3:], dtype=float)
    return {
        "x":x,
        "y":y,
        "wavelength":wavelength,
        "spim":spim.T,
    }

if __name__ == "__main__":
    import pandas as pd
    dirpath = r"/Users/adrien/Documents/Samples/KSU/C33/20210421-1037-58_xy.npz"
    spim_dict = load_spim(dirpath, remove_spike=False)
    hyperspex = Hyperspex(spim_dict["spim"], spim_dict["x"], spim_dict["y"], spim_dict["wavelength"])
    Hyperspex.start_app()