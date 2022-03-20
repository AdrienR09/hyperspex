import matplotlib.pyplot as plt
import pathlib
import sys
sys.path.insert(0, "/Users/adrien/Documents/software/hyperspex/")
from figurefirst import FigureLayout, mpl_functions
from IPython.display import display, SVG
from scipy import optimize
from scipy.interpolate import interp1d
from data_analysis_tools.data import rebin_xy

abs_path = str(pathlib.Path(__file__).parent.absolute())

def map_and_spectrum(**kwargs):

    layout = FigureLayout(abs_path+'/templates/map_and_spectrum.svg')
    layout.make_mplfigures()

    ax1 = layout.axes['img1']
    map = kwargs["map"]
    map -= map.min()
    im = ax1.imshow(map, vmin=kwargs['v_min'], vmax=kwargs['v_max'])
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    cax1 = layout.axes['cb1']
    plt.colorbar(im, cax=cax1, orientation='horizontal')
    cax1.xaxis.set_ticks_position("top")

    ax2 = layout.axes['spec1']
    spectrum = kwargs['spectrum']
    #spectrum = rebin_xy(spectrum)
    x_range = kwargs["range"]
    x_range, spectrum = rebin_xy(x_range, spectrum, )
    ax2.plot(x_range, spectrum)
    ax2.set_xlim(kwargs['x_min'], kwargs['x_max'])
    ax2.set_ylim(kwargs['y_min'], kwargs['y_max'])
    ax2.set_xlabel(kwargs['xlabel'])
    ax2.set_ylabel(kwargs['ylabel'])

    layout.apply_mpl_methods()
    layout.insert_figures("mpl_layer")
    layout.set_layer_visibility("Calque 1", False)
    layout.write_svg(abs_path+'/map_and_spectrum.svg')
    display(SVG(abs_path+'/map_and_spectrum.svg'))