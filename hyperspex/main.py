# This Python file uses the following encoding: utf-8
import sys
import os

from functools import partial
from qtpy.QtWidgets import QApplication, QMainWindow
from PyQt5.Qt import QRectF, QPoint
import pyqtgraph as pg
from qtpy import uic
from qtpy import QtCore
from hyperspex.style.colordefs import ColorScaleInferno
from hyperspex.gui.colorbar.colorbar import ColorbarWidget
from hyperspex.gui.scientific_spinbox.scientific_spinbox import ScienDSpinBox
from lmfit import Model, Parameters

from scipy.constants import c, h, e
import numpy as np

class ImageFitWindow(QMainWindow, QtCore.QObject):

    _sigUpdateSpectrum = QtCore.Signal()

    def __init__(self, data, x_range, y_range):
        """ Create the laser scanner window.
        """
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'ui_image_window.ui')

        # Load it
        super(ImageFitWindow, self).__init__()
        uic.loadUi(ui_file, self)

        with open(os.path.join(this_dir, "style/qdark.qss"), 'r') as stylesheetfile:
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

        self._image_roi = pg.ROI(self._pos1, self._pos2,
                                 maxBounds=QRectF(QPoint(0, 0), QPoint(self._x_range.size, self._y_range.size)))
        self._image_roi.addScaleHandle((1, 0), (0, 1))
        self._image_roi.addScaleHandle((0, 1), (1, 0))
        self.image_view.addItem(self._image_roi)

        self._image_roi.sigRegionChanged.connect(self._update_roi)

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
        self._image_fit = self.sender().fit_image()
        self.plot()

    @property
    def image(self):
        print(self._image_fit[:,:,2])
        return self._image_fit[:,:,0]

    def roi_pos(self):
        return self._pos1, self._pos2

    def plot(self):
        self._image.setImage(self.image)

class ImageWindow(QMainWindow, QtCore.QObject):

    _sigUpdateSpectrum = QtCore.Signal()

    def __init__(self, data, x_range, y_range):
        """ Create the laser scanner window.
        """
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'ui_image_window.ui')

        # Load it
        super(ImageWindow, self).__init__()
        uic.loadUi(ui_file, self)

        with open(os.path.join(this_dir, "style/qdark.qss"), 'r') as stylesheetfile:
            stylesheet = stylesheetfile.read()
        self.setStyleSheet(stylesheet)

        self._data = data
        self._x_range = x_range
        self._y_range = y_range
        self._pos1 = (0, 0)
        self._pos2 = (data.shape[0], data.shape[1])
        self._i_min = 0
        self._i_max = data.shape[2]

        self.init_image()

    def init_image(self):

        self.my_colors = ColorScaleInferno()
        self._image = pg.ImageItem(image=self.image, axisOrder='row-major')
        self._image.setLookupTable(self.my_colors.lut)
        self.image_view.addItem(self._image)

        self._colorbar = ColorbarWidget(self._image)
        self.colorbar.addWidget(self._colorbar)

        self._image_roi = pg.ROI(self._pos1, self._pos2,
                                 maxBounds=QRectF(QPoint(0, 0), QPoint(self._x_range.size, self._y_range.size)))
        self._image_roi.addScaleHandle((1, 0), (0, 1))
        self._image_roi.addScaleHandle((0, 1), (1, 0))
        self.image_view.addItem(self._image_roi)

        self._image_roi.sigRegionChanged.connect(self._update_roi)

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

    @property
    def image(self):
        image = self._data[:, :, self._i_min:self._i_max+1].mean(axis=2)
        return image

    def roi_pos(self):
        return self._pos1, self._pos2

    def plot(self):
        self._image.setImage(self.image)

class SpectrumWindow(QMainWindow, QtCore.QObject):

    _sigUpdateImage = QtCore.Signal()
    _sigUpdateImageFit = QtCore.Signal()

    def __init__(self, data, wavelength_range):
        """ Create the laser scanner window.
        """
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'ui_spectrum_window.ui')

        # Load it
        super(SpectrumWindow, self).__init__()
        uic.loadUi(ui_file, self)

        with open(os.path.join(this_dir, "style/qdark.qss"), 'r') as stylesheetfile:
            stylesheet = stylesheetfile.read()
        self.setStyleSheet(stylesheet)

        self._data = data
        self._wavelength_range = wavelength_range
        self._energy_range = c*h/e/self._wavelength_range
        self._range_type = "wavelength"

        self._pos1 = (0, 0)
        self._pos2 = (data.shape[0], data.shape[1])

        self._gaussian_model = Model(self.gaussian)
        self._gaussian_model.nan_policy = 'omit'

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

        self.range_btn.setText("energy")

        self._update_roi_range()

        self._roi_wg.sigRegionChanged.connect(partial(self._update_roi, False))
        self._roi_min.valueChanged.connect(partial(self._update_roi, True))
        self._roi_max.valueChanged.connect(partial(self._update_roi, True))
        self.range_btn.clicked.connect(self._change_range_type)
        self.fit_btn.clicked.connect(self.start_fit)

        self._plot = self.spectrum_view.plot()
        self.plot()

    def start_fit(self):

        self._sigUpdateImageFit.emit()

    def _update_roi_range(self):

        _min = self.range.min()
        _max = self.range.max()

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
    def spectrum(self):
        spectrum = self._data[self._pos1[0]:self._pos2[0],self._pos1[1]:self._pos2[1]].mean(axis=(0, 1))
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
        if i_min>i_max:
            i_min, i_max = i_max, i_min
        return i_min, i_max

    def plot(self):
        self._plot.setData(self.range, self.spectrum)

    def gaussian(self, x, x0, w, A, B):
        return A * np.exp(-(x - x0) ** 2 / w) + B

    def gaussian_fit(self, xdata, ydata, params):
        fit = self._gaussian_model.fit(ydata, params, x=xdata, max_nfev=20)
        return fit

    def fit_image(self):

        data_shape = self._data.shape
        i_min, i_max = self.roi_index()
        xdata = self.range[i_min:i_max]
        fit_image = np.zeros((data_shape[0], data_shape[1], 4))

        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                ydata = self._data[i, j, i_min:i_max]
                params = self.get_fit_params(xdata, ydata)
                fit = self.gaussian_fit(xdata, ydata, params).best_values.values()
                fit_image[i, j, :] = np.array(list(fit))

        return fit_image

    def get_fit_params(self, xdata, ydata):

        params = Parameters()
        params.add('x0', value=xdata[(ydata - ydata.max()).argmin()])
        params.add('w', min = (xdata.max()-xdata.min())*1e-3,
                   value = (xdata.max()-xdata.min())/2)
        params.add('A', min = 0, value=xdata.max())
        params.add('B', min=0, value=xdata.min())

        return params

class Hyperspex(QtCore.QObject):

    app = QApplication([])

    def __init__(self, data, x_range=None, y_range=None, wavelength_range=None):

        self._spectrum = SpectrumWindow(data, wavelength_range)
        self._image = ImageFitWindow(data, x_range, y_range)

        self._image._sigUpdateSpectrum.connect(self._spectrum._update_spectrum)
        # self._spectrum._sigUpdateImage.connect(self._image._update_image)
        self._spectrum._sigUpdateImageFit.connect(self._image._update_image)

        self.show()

    def show(self):
        """Make window visible and put it above all other windows.
        """
        QMainWindow.show(self._spectrum)
        QMainWindow.show(self._image)

def load_spim(filepath):
    with np.load(filepath) as data:
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
    dirpath = r"/Users/adrien/Desktop/hBN/Samples/POSTECH/Experimental Data/sample_3_1/spim/15_11_2020/20201116-1756-11_sample_3_1_007K_1962ang_135uW_100um_1200rpm_220nm_40x80s_08um.npz"
    spim_dict = load_spim(dirpath)
    hyperspex1 = Hyperspex(spim_dict["spim"]-190, spim_dict["x"], spim_dict["y"], spim_dict["wavelength"])
    sys.exit(Hyperspex.app.exec_())
    # spectrum = SpectrumWindow(spim_dict["spim"], spim_dict["wavelength"])
    # image = ImageWindow(spim_dict["spim"], spim_dict["x"], spim_dict["y"])
    # spectrum.show()
    # hyperspex = Hyperspex(spim_dict["spim"]-198, spim_dict["x"], spim_dict["y"], spim_dict["wavelength"])