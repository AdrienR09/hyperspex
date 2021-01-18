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

from scipy.constants import c, h
import numpy as np




class ImageWindow(QMainWindow):

    _sigUpdateSpectrum = QtCore.Signal()

    def __init__(self, data, x_range, y_range):
        """ Create the laser scanner window.
        """
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'ui_image_window.ui')

        # Load it
        super().__init__()
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

    def init_image(self):

        self.my_colors = ColorScaleInferno()
        self._image = pg.ImageItem(image=self.get_image(), axisOrder='row-major')
        self._image.setLookupTable(self.my_colors.lut)
        self.image.addItem(self._image)

        self._colorbar = ColorbarWidget(self._image)
        self.colorbar.addWidget(self._colorbar)

        self._image_roi = pg.ROI(self._pos1, self._pos2,
                                 maxBounds=QRectF(QPoint(0, 0), QPoint(self._x_range.size, self._y_range.size)))
        self._image_roi.addScaleHandle((1, 0), (0, 1))
        self._image_roi.addScaleHandle((0, 1), (1, 0))
        self.image.addItem(self._image_roi)

        self._image_roi.sigRegionChanged.connect(self._update_roi)

    def _update_roi(self):

        (x1, x2), (y1, y2) = self._image_roi.getArraySlice(self.get_image(), self._image, returnSlice=False)[0]

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

        self.pos1, self.pos2 = (x1, y1), (x2, y2)

        self._sigUpdateSpectrum.emit()

    def _update_image(self, index):
        sender = self.sender()
        self._i_min, self._i_max = sender.roi_index(index)
        image = self._data[:, :, self._i_min:self._i_max+1].mean(axis=2)
        return image

class SpectrumWindow(QMainWindow):

    _sigUpdateImage = []
    for i in range(4):
        _sigUpdateImage.append(QtCore.Signal())

    def __init__(self, data, wavelength_range):
        """ Create the laser scanner window.
        """
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'ui_spectrum_window.ui')

        # Load it
        super().__init__()
        uic.loadUi(ui_file, self)

        with open(os.path.join(this_dir, "style/qdark.qss"), 'r') as stylesheetfile:
            stylesheet = stylesheetfile.read()
        self.setStyleSheet(stylesheet)

        self._data = data
        self._wavelength_range = wavelength_range
        self._energy_range = c*h/self._wavelength_range
        self._range_type = "wavelength"
        self._spectral_roi_wg = []
        self._spectral_roi_min = []
        self._spectral_roi_max = []
        self._spectral_roi = []
        self._spectral_roi_btn = [self.roi_btn_1, self.roi_btn_2, self.roi_btn_3, self.roi_btn_4]

        self._pos1 = (0, 0)
        self._pos2 = (data.shape[0], data.shape[1])

        self.init_spectrum()

    def init_spectrum(self):

        roi_wg_layout = [self.roi_1_layout, self.roi_2_layout, self.roi_3_layout, self.roi_4_layout]

        for i in range(4):

            _min = (self.range.max()-self.range.min())/4*i
            _max = (self.range.max()-self.range.min())/4*(i+1)

            self._spectral_roi.append(_min, _max)

            self._spectral_roi_min.append(ScienDSpinBox())
            self._spectral_roi_min[i].setSuffix("m")
            self._spectral_roi_min[i].setMinimumWidth(100)
            self._spectral_roi_min[i].setMinimum(self.range.min())
            self._spectral_roi_min[i].setMaximum(self.range.max())
            self._spectral_roi_min[i].setValue(_min)
            roi_wg_layout[i].addWidget(self._spectral_roi_min[i])

            self._spectral_roi_max.append(ScienDSpinBox())
            self._spectral_roi_max[i].setSuffix("m")
            self._spectral_roi_max[i].setMinimumWidth(100)
            self._spectral_roi_max[i].setMinimum(self.range.min())
            self._spectral_roi_max[i].setMaximum(self.range.max())
            self._spectral_roi_max[i].setValue(_max)
            roi_wg_layout[i].addWidget(self._spectral_roi_max[i])

            roi = pg.LinearRegionItem(values=[0, 100], orientation=pg.LinearRegionItem.Vertical,
                                brush="#fc03032f", bounds=[self.range.min(), self.range.max()])
            self._spectral_roi.append(roi)
            self._spectrum_window.spectrum.addItem(roi)

            if i > 0:
                roi.hide()

            self._spectral_roi_wg[i].sigRegionChanged.connect(partial(self._update_roi, False, i))
            self._spectral_roi_min[i].valueChanged.connect(partial(self._updatel_roi, True, i))
            self._spectral_roi_max[i].valueChanged.connect(partial(self._update_roi, True, i))
            self._spectral_roi_btn[i].valueChanged.connect(partial(self._show_roi, i))

        self._plot = self.spectrum.plot()

    def _update_roi(self, spin_widget, index):

        if spin_widget:
            _min, _max = self._spectral_roi_min[index].value(), self._spectral_roi_max[index].value()
            self._spectral_roi[index] = (_min, _max)
            self._spectral_roi[index].setRegion([_min, _max])

        elif spin_widget:
            _min, _max = self._spectral_roi[index].getRegion()
            self._spectral_roi[index] = (_min, _max)
            self._spectral_roi_min[index].setValue(_min)
            self._spectral_roi_max[index].setValue(_max)

        self._sigUpdateImage[index].emit()

    def _show_roi(self, index):

        btn_state = self._spectral_roi_btn[index].isVisible()
        if btn_state:
            self._spectral_roi_btn[index].hide()
        else:
            self._spectral_roi_btn[index].show()

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

    def _update_spectrum(self):
        image_wg = self.sender()
        self._pos1 = image_wg._pos1
        self._pos2 = image_wg._pos2
        self.plot()

    def range_index(self, value):
        index = np.abs(self.range - value).argmin()
        return index

    def roi_index(self, index):
        i_min = self.range_index(self._spectral_roi_min[index])
        i_max = self.range_index(self._spectral_roi_max[index])
        return i_min, i_max

    def plot(self):
        self._plot.setData(self.range, self.spectrum)

class Hyperspex:

    def __init__(self, data, x_range=None, y_range=None, wavelength_range=None):

        app = QApplication([])
        self._spectrum = SpectrumWindow(data, wavelength_range)
        self._images = []
        for i in range(4):
            image = ImageWindow(data, x_range, y_range)
            self._images.append(image)
            self._spectrum._sigUpdateImage[i].connect(partial(image._update_image(i)))
        sys.exit(app.exec_())

    def show(self):
        """Make window visible and put it above all other windows.
        """
        QMainWindow.show(self._spectrum)

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
    dirpath = r"C:\Users\L2C_USER\Desktop\Samples\LPCNO\Experimental data\S2\20210118\20210118-0959-39_xy.npz"
    spim_dict = load_spim(dirpath)
    app = QApplication([])
    spectrum = SpectrumWindow(spim_dict["spim"], spim_dict["wavelength"])
    sys.exit(app.exec_())
    # hyperspex = Hyperspex(spim_dict["spim"]-198, spim_dict["x"], spim_dict["y"], spim_dict["wavelength"])