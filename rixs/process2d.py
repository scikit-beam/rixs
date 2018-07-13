""" Functions for processing 2D Image data.

Typical command line workflow is executed in run_rest()

Parameters
----------
selected_image_name : string
    unique descriptor of the data that will be convereted
    into a rixs spectrum
photon_events : array
    three column x, y, Iph photon locations and intensities
curvature : dictionary
    n2d order polynominal coeficients defining image curvature
    {'x^2': coef, 'x1': coef, 'x0': coef}
image_meta : string
    container for metadata
spectrum: array
    binned spectrum two column defining
    pixel, intensity
resolution : array
    parameters defining gaussian from fit_resolution
"""

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import lmfit


def make_fake_image(x0=1000, x1=0.02, x2=0., sigma=2, noise=0.002):
    """Make a fake list of photon events.
    Parameters
    ----------
    x0, x1, x2 : float, float, float
        values defining image curvature
    noise : float
        probability of noise
        i.e. randomly located events
    Returns
    ----------
    photon_events : array
        three column x, y, Iph photon locations
    """
    randomy = 2**11*rand(1000000)
    choose = (np.exp(-(randomy-x0)**2/(2*sigma)) + noise) > rand(randomy.size)
    yvalues = randomy[choose]
    xvalues = 2**11 * rand(len(yvalues))
    Iph = np.ones(xvalues.shape)
    noise = (rand(xvalues.size) - 0.5)*0.4
    Iph += noise
    yvalues_curvature = yvalues + x1*xvalues + x2*xvalues**2
    return np.vstack((xvalues, yvalues_curvature, Iph)).transpose()


def plot_scatter(ax, photon_events, pointsize=1, **kwargs):
    """Make a scatterplot
    This is effcient when there are a low RIXS intensity

    Parameters
    ----------
    ax : matplotlib axes object
        axes for plotting on
    photon_events : array
        two column x, y, Iph photon locations and intensities
    pointsize : float
        multiply all point sizes by a fixed factor
    **kwargs : dictionary
        passed onto matplotlib.pyplot.scatter.
    Returns
    ----------
    image_artist : matplotlib artist object
        artist from image scatter plot
    """
    ax.cla()
    ax.set_facecolor('black')
    defaults = {'c': 'white', 'edgecolors': 'white', 'alpha': 0.5,
                'pointsize': pointsize}

    kwargs.update({key: val for key, val in defaults.items()
                   if key not in kwargs})

    pointsize = kwargs.pop('pointsize')
    image_artist = ax.scatter(photon_events[:, 0], photon_events[:, 1],
                              s=photon_events[:, 2]*pointsize, **kwargs)
    return image_artist


def plot_pcolor(ax, photon_events, cax=None, bins=None, **kwargs):
    """Make an pcolorfast image of the photon_events
    This is effcient when there are a large RIXS intensity

    Parameters
    ----------
    ax : matplotlib axes object
        axes for plotting on
    photon_events : array
        two column x, y, Iph photon locations and intensities
    cax : matplotlib axes object
        use to plot colorbar to a specific axis
    bins : int or array_like or [int, int] or [array, array]
        The bin specification in y then x order:
            * None chooses bin edges from int/2, int/2 +1 etc.
              with partially populated edges removed.
            * If int, the number of bins for the two dimensions (ny=nx=bins).
            * If array_like, the bin edges for the two dimensions
              (y_edges=x_edges=bins).
            * If [int, int], the number of bins in each dimension
              (ny, nx = bins).
            * If [array, array], the bin edges in each dimension
              (y_edges, x_edges = bins).
            * A combination [int, array] or [array, int], where int
              is the number of bins and array is the bin edges.

    **kwargs : dictionary
        passed onto matplotlib.pyplot.imshow.
    Returns
    ----------
    image_artist : matplotlib artist object
        artist from image plot
    cb_artist : matplotlib artist object
        artist from colorbar associated with plot
    """
    ax.cla()

    image_output = photon_events_to_image(photon_events, bins=bins)
    x_centers, y_centers, image, x_edges, y_edges = image_output

    defaults = {'cmap': 'gray',
                'vmin': np.nanpercentile(image, 1),
                'vmax': np.nanpercentile(photon_events[:, 2], 99)}

    kwargs.update({key: val for key, val in defaults.items()
                   if key not in kwargs})

    xlim = (x_edges.min(), x_edges.max())
    ylim = (y_edges.min(), y_edges.max())
    image_artist = ax.pcolorfast(xlim, ylim, image, **kwargs)
    cb_artist = plt.colorbar(image_artist, ax=ax, cax=cax)
    ax.axis('tight')

    return image_artist, cb_artist


def poly(x, x2=0., x1=0., x0=500.):
    """Second order polynominal function for fitting curvature.
    Returns x2*x**2 + x1*x + x0
    """
    return x2*x**2 + x1*x + x0


def get_curvature_offsets(photon_events, bins=None):
    """ Determine the offests that define the isoenergetic line.
    This is determined as the maximum of the cross correlation function with
    a reference taken from the center of the image.

    Parameters
    ------------
    photon_events : array
        three column x, y, Iph photon locations and intensities
    bins : int or array_like or [int, int] or [array, array]
        The bin specification in y then x order:
            * If None
            * If int, the number of bins for the two dimensions (nx=ny=bins).
            * If array_like, the bin edges for the two dimensions
              (y_edges=x_edges=bins).
            * If [int, int], the number of bins in each dimension
              (ny, nx = bins).
            * If [array, array], the bin edges in each dimension
              (y_edges, x_edges = bins).
            * A combination [int, array] or [array, int], where int
              is the number of bins and array is the bin edges.

    Returns
    -------------
    x_centers : array
        columns positions where offsets were determined
        i.e. binx/2, 3*binx/2, 5*binx/2, ...
    offsets : array
        np.array of row offsets defining curvature. This is referenced
        to the center of the image.
    """
    x = photon_events[:, 0]
    y = photon_events[:, 1]
    Iph = photon_events[:, 2]

    if bins is None:
        ybins = np.arange(y.min()//1 + 0.5, y.max()//1, 1)
        xstep = 50
        xbins = np.arange(x.min() + xstep/2, x.max() - xstep/2, xstep)
        bins = (ybins, xbins)

    image, y_edges, x_edges = np.histogram2d(y, x, bins=bins, weights=Iph)
    y_centers = (y_edges[:-1] + y_edges[1:])/2
    x_centers = (x_edges[:-1] + x_edges[1:])/2

    ref_column = image[:, image.shape[1]//2]

    offsets = np.array([])
    for column in image.T:
        cross_correlation = np.correlate(column, ref_column, mode='same')
        offsets = np.append(offsets, y_centers[np.argmax(cross_correlation)])
    return x_centers, offsets - offsets[offsets.shape[0]//2]


def fit_curvature(photon_events, bins=None,
                  guess_curvature={'x2': 0., 'x1': 0., 'x0': 500.},
                  vary_params={'x2': True, 'x1': True, 'x0': True}):
    """Get offsets, fit them and return polynomial that defines the curvature

    Parameters
    -------------
    photon_events : array
        two column x, y photon location coordinates
    bins : int or array_like or [int, int] or [array, array]
        The bin specification in y then x order:
            * If int, the number of bins for the two dimensions (nx=ny=bins).
            * If array_like, the bin edges for the two dimensions
              (y_edges=x_edges=bins).
            * If [int, int], the number of bins in each dimension
              (ny, nx = bins).
            * If [array, array], the bin edges in each dimension
              (y_edges, x_edges = bins).
            * A combination [int, array] or [array, int], where int
              is the number of bins and array is the bin edges.
    curvature : dictionary
    n2d order polynominal coeficients defining
    curvature
    {'x^2': coef, 'x1': coef, 'x0': coef}

    Returns
    -----------

    """
    x_centers, offsets = get_curvature_offsets(photon_events, bins)
    poly_model = lmfit.Model(poly, independent_vars=['x'])
    params = poly_model.make_params()
    for name, c in guess_curvature.items():
        params[name].set(value=c)
    for name, TF in vary_params.items():
        params[name].set(value=TF)
    result = poly_model.fit(offsets, x=x_centers, params=params)

    if result.success is not True:
        print(result.message)

    return result.best_values


def plot_curvature(ax1, curvature, photon_events, plot_offset=0.):
    """ Plot a red line defining curvature on ax1

    Parameters
    ----------
    ax1 : matplotlib axes object
        axes for plotting on
    curvature : dictionary
        n2d order polynominal coeficients defining image curvature
        {'x^2': coef, 'x1': coef, 'x0': coef}
    photon_events : array
        two column x, y photon location coordinates

    Returns
    ---------
    curvature_artist : matplotlib artist object
        artist from image scatter plot
    """
    x = np.arange(np.nanmax(photon_events[:, 0]))
    y = poly(x, **curvature)
    return plt.plot(x, y + plot_offset, 'r-')


def estimate_elastic_pos(photon_events, x_range=(0, 20), bins=None):
    """
    Estimate where the elastic line.


    Parameters
    ----------
    photon_events : array
        two column x, y, Iph photon locations and intensities
    x_range : (float, float)
        Select x region of image to use
        A tuple defines a range in x (minx, maxx)
    bins : int or sequence of scalars or str, optional
        Binning in the y direction.
        If 'bins' is None a step of 1 is assumed over the relavant range
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines the bin edges, including the rightmost
        edge, allowing for non-uniform bin widths.
    """
    x = photon_events[:, 0]
    y = photon_events[:, 1]
    Iph = photon_events[:, 2]

    choose = np.logical_and(x > x_range[0], x < x_range[1])
    y_choose = y[choose]
    I_choose = Iph[choose]

    if bins is None:
        bins = np.arange(y_choose.min()//1 + 0.5, y_choose.max()//1 - 0.5)
    I_profile, y_edges = np.histogram(y_choose, bins=bins, weights=I_choose)
    y_centers = (y_edges[:-1] + y_edges[1:])/2
    elastic_y_value = y_centers[np.argmax(I_profile)]
    return elastic_y_value


def extract(photon_events, curvature, bins=None):
    """Apply curvature to photon events to create pixel versus intensity spectrum

    Parameters
    ----------
    photon_events : array
        three column x, y , Iph photon locations and intensities
    curvature : dictionary
        n2d order polynominal coeficients defining image curvature
        {'x^2': coef, 'x1': coef, 'x0': coef}
    bins : int or sequence of scalars or str, optional
        Binning in the y direction.
        If 'bins' is None a step of 1 is assumed over the relavant range
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines the bin edges, including the rightmost
        edge, allowing for non-uniform bin widths.

    Returns
    -------
    spectrum : array
        two column: y,      , I
                    position, intensity
    """
    x = photon_events[:, 0]
    y = photon_events[:, 1]
    Iph = photon_events[:, 2]
    try:
        curvature.pop('x0')
    except KeyError:
        pass
    corrected_y = y - poly(x, x0=0., **curvature)

    if bins is None:
        bins = np.arange(corrected_y.min()//1 + 0.5,
                         corrected_y.max()//1 - 0.5)
    Ibin, y_edges = np.histogram(corrected_y, bins=bins, weights=Iph)
    y_centers = (y_edges[:-1] + y_edges[1:])/2
    spectrum = np.vstack((y_centers, Ibin)).transpose()
    return spectrum


def photon_events_to_image(photon_events, bins=None):
    """ Convert 1D photon events into 2D image
    Opposite of image_to_photon_events

    Parameters
    -----------
    photon_events : np.array
        three column x, y, Iph photon locations and intensities
    bins : int or array_like or [int, int] or [array, array]
        The bin specification in y then x order:
            * If int, the number of bins for the two dimensions (nx=ny=bins).
            * If array_like, the bin edges for the two dimensions
              (y_edges=x_edges=bins).
            * If [int, int], the number of bins in each dimension
              (ny, nx = bins).
            * If [array, array], the bin edges in each dimension
              (y_edges, x_edges = bins).
            * A combination [int, array] or [array, int], where int
              is the number of bins and array is the bin edges.
    Returns
    -----------
    x_centers : array
        1D vector describing column position
    y_centers : array
        1D vector describing row position
    image : array
        2D image
    x_centers : array
        1D vector describing bin edges for columns
    y_centers : array
        1D vector describing row position
    """
    x = photon_events[:, 0]
    y = photon_events[:, 1]
    Iph = photon_events[:, 2]

    if bins is None:
        ybins = np.arange(y.min()//1 + 0.5, y.max()//1, 1)
        xbins = np.arange(x.min()//1 + 0.5, x.max()//1, 1)
        bins = (ybins, xbins)

    image, y_edges, x_edges = np.histogram2d(y, x, bins=bins, weights=Iph)
    y_centers = (y_edges[:-1] + y_edges[1:])/2
    x_centers = (x_edges[:-1] + x_edges[1:])/2

    return x_centers, y_centers, image, x_edges, y_edges


def run_test():
    """Run at test of the code.
    This can also be used as a reference for command-line analysis.

    Returns
    -------------
    ax1 : matplotlib axis object
        axes used to plot the image
    ax2 : matplotlib axis object
        axes used to plot the spectrum
    """
    photon_events = make_fake_image()
    curvature = fit_curvature(photon_events)
    print("Curvature is\n{} \t {:.3e}\n{}\t{:.3e}\n"
          "{}\t{:.3e}".format(*sum(curvature.items(), ())))

    fig1, ax1 = plt.subplots()
    plt.title('Image')
    fig2, ax2 = plt.subplots()
    plt.title('Spectrum')

    plot_scatter(ax1, photon_events)
    plot_curvature(ax1, curvature, photon_events)

    spectrum = extract(photon_events, curvature)
    ax2.plot(spectrum[:, 0], spectrum[:, 0], '.')
    # Depreated function will be added back later
    # resolution = fit_resolution(spectrum)
    # plot_resolution(ax2, spectrum)
    # plot_resolution_fit(ax2, spectrum, resolution)
    # print("Resolution is {}".format(resolution[0]))

    return ax1, ax2


if __name__ == "__main__":
    """Run test of code"""
    print('Run a test of the code')
    run_test()

# DEPRECIATED FUNCTIONS HERE TO BE UPGRADED LATER
# REPLACE update with lmfit derived function
# from lmfit.lineshapes import gaussian
# def fit_resolution(spectrum, xmin=-np.inf, xmax=np.inf):
#    """Fit a Gaussian model to ['spectrum']
#    in order to determine resolution_values.
#
#    Parameters
#    ----------
#    spectrum: array
#        binned spectrum two column defining
#        pixel, intensity
#    xmin/xmax : float
#        minimum/maximum value for fitting range
#
#    Returns
#    ----------
#    resolution : array
#        values parameterizing gaussian function
#        ['FWHM', 'center', 'amplitude', 'offset']
#    """
#    allx = spectrum[:, 0]
#    choose = np.logical_and(allx > xmin, allx <= xmax)
#    x = allx[choose]
#    y = spectrum[:, 1][choose]
#
#    GaussianModel = lmfit.Model(gaussian)
#    params = GaussianModel.make_params()
#    params['center'].value = x[np.argmax(y)]
#    params['FWHM'].set(min=0)
#    result = GaussianModel.fit(y, x=x, params=params)
#
#    if result.success:
#        resolution = [result.best_values[arg]
#                      for arg in ['FWHM', 'center', 'amplitude', 'offset']]
#        return resolution


# TO BE INLCUDED LATER
# def plot_resolution(ax2, spectrum):
#     """ Plot blue points defining the spectrum on ax2
#
#     Parameters
#     ------------
#     ax2 : matplotlib axes object
#     spectrum: array
#         binned spectrum two column defining
#         pixel, intensity
#
#     Returns
#     -----------
#     spectrum_artist : matplotlib artist
#         Resolution plotting object
#     """
#     plt.sca(ax2)
#     spectrum_artist = plt.plot(spectrum[:, 0], spectrum[:, 1], 'b.')
#     plt.xlabel('pixels')
#     plt.ylabel('Photons')
#     return spectrum_artist
#
#
# def plot_resolution_fit(ax2, spectrum, resolution, xmin=None, xmax=None):
#     """Plot the gaussian fit to the resolution function
#
#     Parameters
#     -----------
#     ax2 : matplotlib axes object
#         axes to plot on
#     spectrum: array
#         binned spectrum two column defining
#         pixel, intensity
#     xmin/xmax : float/float
#         range of x values to plot over (the same as fitting range)
#     resolution_values : array
#         parameters defining gaussian from fit_resolution
#     """
#     plt.sca(ax2)
#     if xmin is None:
#         xmin = np.nanmin(spectrum[:, 0])
#     if xmax is None:
#         xmax = np.nanmax(spectrum[:, 0])
#     x = np.linspace(xmin, xmax, 10000)
#     y = gaussian(x, *resolution)
#     return plt.plot(x, y, 'r-')
#
#
#
# DECIDE WHETHER THIS GOES HERE
# def clean_image_threshold(photon_events, thHigh):
#     """ Remove cosmic rays and glitches using a fixed threshold count.
#
#     Parameters
#     ------------
#     photon_events : array
#         three column x, y, z with location coordinates (x,y)
#         and intensity (z)
#     thHigh: float
#         Threshold limit. Pixels with counts above/below +/- thHigh
#         will be removed from image.
#
#     Returns
#     -----------
#     clean_photon_events : array
#         Cleaned photon_events
#     changed_pixels: float
#         1 - ratio between of removed and total pixels.
#     """
#     bad_indices = np.logical_and(photon_events[:, 2] < thHigh,
#                                  photon_events[:, 2] > -1.*thHigh)
#     clean_photon_events = photon_events[bad_indices, :]
#     changed_pixels = 1.0 - (clean_photon_events.shape[0]
#                             / photon_events.shape[0]*1.0)
#     return clean_photon_events, changed_pixels
