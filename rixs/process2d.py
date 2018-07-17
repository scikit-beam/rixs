""" Functions for processing 2D Image data.

Parameters
----------
selected_image_name : string
    unique descriptor of the data that will be convereted
    into a rixs spectrum
photon_events : array
    three column x, y, Iph photon locations and intensities
curvature : array
    The polynominal coeffcients describing the image curvature.
    These are in decreasing order e.g.
    .. code-block:: python
       curvature[0]*x**2 + curvature[1]*x**1 + curvature[2]*x**0
image_meta : string
    container for metadata
spectrum: array
    binned spectrum two column defining
    pixel, intensity
resolution : array
    parameters defining gaussian from fit_resolution
"""

import numpy as np
from numpy.polynomial.polynomial import polyfit


def get_curvature_offsets(photon_events, bins=None):
    """ Determine the offests that define the isoenergetic line.
    The input data is divided into columns as defined by bins.
    Each column is then collaped into a spectrum assuming zero curvature
    The cross correlation betwen different columns is then calculations
    and the peak position relative to the center of the image is returned

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


def fit_curvature(photon_events, guess_curvature, bins=None):
    """Get offsets, fit them and return polynomial that defines the curvature

    Parameters
    -------------
    photon_events : array
        two column x, y photon location coordinates
    curvature : array
        The polynominal coeffcients describing the image curvature.
        These are in decreasing order e.g.
        .. code-block:: python
           curvature[0]*x**2 + curvature[1]*x**1 + curvature[2]*x**0
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

    """
    x_centers, offsets = get_curvature_offsets(photon_events, bins)
    reversed_curvature = polyfit(x_centers, offsets, len(guess_curvature) - 1)
    curvature = reversed_curvature[::-1]

    return curvature


def estimate_elastic_pos(photon_events, x_range=(0, 20), bins=None):
    """
    Estimate where the elastic line is.

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


def apply_curvature(photon_events, curvature, bins=None):
    """Apply curvature to photon events to create pixel versus intensity spectrum

    Parameters
    ----------
    photon_events : array
        three column x, y , Iph photon locations and intensities
    curvature : array
        The polynominal coeffcients describing the image curvature.
        These are in decreasing order e.g.
        .. code-block:: python
       curvature[0]*x**2 + curvature[1]*x**1 + curvature[2]*x**0
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
    curvature[-1] = 0
    corrected_y = y - np.polyval(curvature, x)

    if bins is None:
        bins = np.arange(corrected_y.min()//1 + 0.5,
                         corrected_y.max()//1 - 0.5)
    Ibin, y_edges = np.histogram(corrected_y, bins=bins, weights=Iph)
    y_centers = (y_edges[:-1] + y_edges[1:])/2
    spectrum = np.column_stack((y_centers, Ibin))
    return spectrum


def photon_events_to_image(photon_events, bins=None):
    """ Convert 1D photon events into 2D image
    In the default binning, data at the edges is excluded, so that
    problems with half empty bins are avoided. This might come at
    the cost of loosing half a pixel of data.

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
        +ve y is up convention is applied.
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

    # impose +ve y is up convention
    yorder = np.argsort(y_centers)[::-1]
    y_centers = y_centers[yorder]
    image = image[yorder, :]
    y_edges = y_edges[::-1]

    return x_centers, y_centers, image, x_edges, y_edges


def image_to_photon_events(image):
    """ Convert 2D image into 1D photon events
    This assumes integers define the centers of bins.
    Zeros are not included.

    Parameters
    -----------
    image : 2D np.array
        photon intensities

    Returns
    -----------
    photon_events : np.array
        three column x, y, Iph photon locations and intensities
    """
    x_centers = np.arange(image.shape[1])
    y_centers = np.arange(image.shape[0])[::-1]  # +ve y is up convention
    X_CENTERS, Y_CENTERS = np.meshgrid(x_centers, y_centers)

    choose = image > 0

    photon_events = np.column_stack((X_CENTERS[choose].ravel(),
                                     Y_CENTERS[choose].ravel(),
                                     image[choose].ravel()))

    return photon_events
