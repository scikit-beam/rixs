import numpy as np
import matplotlib.pyplot as plt
from rixs.process2d import photon_events_to_image


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

    image_artist = ax.pcolorfast(x_edges, y_edges, image, **kwargs)
    cb_artist = plt.colorbar(image_artist, ax=ax, cax=cax)
    ax.axis('tight')

    return image_artist, cb_artist


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
    y = np.polyval(curvature, x)
    return plt.plot(x, y + plot_offset, 'r-')
