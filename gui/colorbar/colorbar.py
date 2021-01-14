# -*- coding: utf-8 -*-
"""
This module contains a GUI for operating the spectrometer camera logic module.

Qudi is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Qudi is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Qudi. If not, see <http://www.gnu.org/licenses/>.

Copyright (c) the Qudi Developers. See the COPYRIGHT.txt file at the
top-level directory of this distribution and at <https://github.com/Ulm-IQO/qudi/>
"""

import os
import pyqtgraph as pg
from functools import partial

from style.colordefs import ColorScaleInferno, ColorScaleMagma
from gui.scientific_spinbox.scientific_spinbox import ScienDSpinBox, ScienSpinBox

from qtpy import QtWidgets
from qtpy import uic

import numpy as np

class ColorBar(pg.GraphicsObject):
    """ Create a ColorBar according to a previously defined color map.

    @param object pyqtgraph.ColorMap cmap: a defined colormap
    @param float width: width of the colorbar in x direction, starting from
                        the origin.
    @param numpy.array ticks: optional, definition of the relative ticks marks
    """

    def __init__(self, cmap, width, cb_min, cb_max):

        pg.GraphicsObject.__init__(self)

        # handle the passed arguments:
        self.stops, self.colors = cmap.getStops('float')
        self.stops = (self.stops - self.stops.min())/self.stops.ptp()
        self.width = width

        # Constructs an empty picture which can be altered by QPainter
        # commands. The picture is a serialization of painter commands to an IO
        # device in a platform-independent format.
        self.pic = pg.QtGui.QPicture()

        self.refresh_colorbar(cb_min, cb_max)

    def refresh_colorbar(self, cb_min, cb_max, width = None, height = None, xMin = None, yMin = None):
        """ Refresh the appearance of the colorbar for a changed count range.

        @param float cb_min: The minimal count value should be passed here.
        @param float cb_max: The maximal count value should be passed here.
        @param float width: optional, with that you can change the width of the
                            colorbar in the display.
        """

        if width is None:
            width = self.width
        else:
            self.width = width

#       FIXME: Until now, if you want to refresh the colorbar, a new QPainter
#              object has been created, but I think that it is not necassary.
#              I have to figure out how to use the created object properly.
        p = pg.QtGui.QPainter(self.pic)
        p.drawRect(self.boundingRect())
        p.setPen(pg.mkPen('k'))
        grad = pg.QtGui.QLinearGradient(width/2.0, cb_min*1.0, width/2.0, cb_max*1.0)
        for stop, color in zip(self.stops, self.colors):
            grad.setColorAt(1.0 - stop, pg.QtGui.QColor(*[255*c for c in color]))
        p.setBrush(pg.QtGui.QBrush(grad))
        if xMin is None:
            p.drawRect(pg.QtCore.QRectF(0, cb_min, width, cb_max-cb_min))
        else:
            # If this picture whants to be set in a plot, which is going to be
            # saved:
            p.drawRect(pg.QtCore.QRectF(xMin, yMin, width, height))
        p.end()

        vb = self.getViewBox()
        # check whether a viewbox is already created for this object. If yes,
        # then it should be adjusted according to the full screen.
        if vb is not None:
            vb.updateAutoRange()
            vb.enableAutoRange()

    def paint(self, p, *args):
        """ Overwrite the paint method from GraphicsObject.

        @param object p: a pyqtgraph.QtGui.QPainter object, which is used to
                         set the color of the pen.

        Since this colorbar object is in the end a GraphicsObject, it will
        drop an implementation error, since you have to write your own paint
        function for the created GraphicsObject.
        """
        # paint colorbar
        p.drawPicture(0, 0, self.pic)

    def boundingRect(self):
        """ Overwrite the paint method from GraphicsObject.

        Get the position, width and hight of the displayed object.
        """
        return pg.QtCore.QRectF(self.pic.boundingRect())

class ColorbarWidget(QtWidgets.QWidget):
    """ Create the SettingsDialog window, based on the corresponding *.ui file."""

    def __init__(self, image_widget):
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'ui_colorbar.ui')

        # Load it
        super(ColorbarWidget, self).__init__()
        uic.loadUi(ui_file, self)

        self._cb_min = 0
        self._cb_max = 100

        self.init_spin_box()
        self.init_colorbar()

        self.set_image(image_widget)

        self.percentile.clicked.emit()
        self.percentile.setChecked(True)

    def init_spin_box(self):

        self._min_percentile = ScienDSpinBox()
        self._min_manual = ScienDSpinBox()
        self._max_percentile = ScienDSpinBox()
        self._max_manual = ScienDSpinBox()

        self._min_percentile.setSuffix('%')
        self._min_percentile.setMinimum(0)
        self._min_percentile.setMaximum(100)
        self._min_percentile.setValue(0)

        self._min_manual.setSuffix('c/s')

        self._max_percentile.setSuffix('%')
        self._max_percentile.setMinimum(0)
        self._max_percentile.setMaximum(100)
        self._max_percentile.setValue(100)

        self._max_manual.setSuffix('c/s')

        self.min.addWidget(self._min_manual)
        self.min.addWidget(self._min_percentile)
        self.max.addWidget(self._max_percentile)
        self.max.addWidget(self._max_manual)

        self._min_percentile.valueChanged.connect(self.shortcut_to_cb_centiles)
        self._min_manual.valueChanged.connect(self.shortcut_to_cb_manual)
        self._max_percentile.valueChanged.connect(self.shortcut_to_cb_centiles)
        self._max_manual.valueChanged.connect(self.shortcut_to_cb_manual)

        self.manual.clicked.connect(self.update_cb_range)
        self.percentile.clicked.connect(self.update_cb_range)

    def init_colorbar(self):

        self.my_colors = ColorScaleInferno()
        self._cb = ColorBar(self.my_colors.cmap_normed, width=100, cb_min=self._cb_min, cb_max=self._cb_max)
        self.colorbar.addItem(self._cb)
        self.colorbar.hideAxis('bottom')
        self.colorbar.setLabel('left', 'Intensity', units='c/s')
        self.colorbar.setMouseEnabled(x=False, y=False)

    def set_image(self, image_widget):
        self._image = image_widget
        self._min_manual.setValue(np.min(self._image.image))
        self._min_percentile.setValue(0)
        self._max_manual.setValue(np.max(self._image.image))
        self._max_percentile.setValue(100)
        self.refresh_colorbar()

    def get_cb_range(self):
        """ Determines the cb_min and cb_max values for the image
        """
        # If "Manual" is checked, or the image data is empty (all zeros), then take manual cb range.
        if self.manual.isChecked() or np.count_nonzero(self._image.image) < 1:
            cb_min = self._min_manual.value()
            cb_max = self._max_manual.value()

        # Otherwise, calculate cb range from percentiles.
        else:
            # Exclude any zeros (which are typically due to unfinished scan)
            image_nonzero = self._image.image[np.nonzero(self._image.image)]

            # Read centile range
            low_centile = self._min_percentile.value()
            high_centile = self._max_percentile.value()

            cb_min = np.percentile(image_nonzero, low_centile)
            cb_max = np.percentile(image_nonzero, high_centile)

        cb_range = [cb_min, cb_max]

        return cb_range

    def refresh_colorbar(self):
        """ Adjust the colorbar.

        Calls the refresh method from colorbar, which takes either the lowest
        and higherst value in the image or predefined ranges. Note that you can
        invert the colorbar if the lower border is bigger then the higher one.
        """
        cb_range = self.get_cb_range()
        self._cb.refresh_colorbar(cb_range[0], cb_range[1])

    def refresh_image(self):
        """ Update the current Depth image from the logic.

        Everytime the scanner is scanning a line in depth the
        image is rebuild and updated in the GUI.
        """

        image_data = self._image.image
        cb_range = self.get_cb_range()

        # Now update image with new color scale, and update colorbar
        self._image.setImage(image=image_data, levels=(cb_range[0], cb_range[1]))
        self.refresh_colorbar()

    def update_cb_range(self):
        """Redraw colour bar and scan image."""
        self.refresh_colorbar()
        self.refresh_image()

    def shortcut_to_cb_manual(self):
        """Someone edited the absolute counts range for the xy colour bar, better update."""
        self.manual.setChecked(True)
        self.update_cb_range()

    def shortcut_to_cb_centiles(self):
        """Someone edited the centiles range for the xy colour bar, better update."""
        self.percentile.setChecked(True)
        self.update_cb_range()
