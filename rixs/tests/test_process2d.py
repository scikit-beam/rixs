from rixs import process2d
import numpy as np
from numpy.random import rand
from numpy.testing import assert_array_almost_equal


def make_fake_image(curvature, elastic_y, sigma=2, noise=0.002):
    """Make a fake list of photon events.
    Parameters
    ----------
    curvature : array
        The polynominal coeffcients describing the image curvature.
        These are in decreasing order e.g.
        .. code-block:: python
           curvature[0]*x**2 + curvature[1]*x**1 + curvature[2]*x**0
    elastic_y : float
        height (y value) to locate the elastic line
    noise : float
        probability of noise
        i.e. randomly located events
    Returns
    ----------
    photon_events : array
        three column x, y, Iph photon locations
    """
    randomy = 2**11*rand(1000000)
    choose = (np.exp(-(randomy-elastic_y)**2/(2*sigma)) +
              noise) > rand(randomy.size)
    yvalues = randomy[choose]
    xvalues = 2**11 * rand(len(yvalues))
    Iph = np.ones(xvalues.shape)
    noise = (rand(xvalues.size) - 0.5)*0.4
    Iph += noise
    curvature[-1] = 0
    yvalues_curvature = yvalues + np.polyval(curvature, xvalues)
    return np.vstack((xvalues, yvalues_curvature, Iph)).transpose()


def test_curvature_fit():
    """Test curvature fit on simulated data."""
    fake_curvature = np.array([0.02, 1000.])
    photon_events = make_fake_image(fake_curvature, 1000., noise=0)
    curvature = process2d.fit_curvature(photon_events,
                                        np.array([0., 0]))

    val = curvature[0]
    guess = fake_curvature[0]
    ratio = np.abs(val - guess) / (val + guess)
    assert(ratio < 0.05)


def test_apply_curvature():
    """Test apply_curvature on simulated data."""
    y_edges = np.arange(-0.5, 100.5, 1)
    y_centers = (y_edges[:-1] + y_edges[1:])/2
    x_centers = 100*rand(y_centers.size)

    Iph = np.exp(-(y_centers-50)**2)

    photon_events = np.vstack((x_centers, y_centers, Iph)).transpose()
    spectrum = process2d.apply_curvature(photon_events,
                                         curvature=np.array([0., 0., 0.]),
                                         bins=y_edges)

    assert_array_almost_equal(y_centers, spectrum[:, 0])
    assert_array_almost_equal(Iph, spectrum[:, 1])


def test_photon_events_to_image():
    """Test conversion to image"""
    size = 10
    x = np.arange(0, size)
    y = np.copy(x)
    Iph = np.ones_like(x)

    photon_events = np.vstack((x, y, Iph)).T
    image_info = process2d.photon_events_to_image(photon_events)
    x_centers, y_centers, image, *_ = image_info

    corresponding_image = np.identity(image.shape[0])[::-1, :]

    assert_array_almost_equal(x[1:-1], x_centers)
    assert_array_almost_equal(y[1:-1][::-1], y_centers)
    assert_array_almost_equal(image, corresponding_image)


def test_image_to_photon_events():
    """Test conversion between image and photon_events"""
    size = 10
    image = np.identity(10)[::-1, :]

    photon_events = process2d.image_to_photon_events(image)
    assert_array_almost_equal(photon_events[:, 0], np.arange(size)[::-1])
    assert_array_almost_equal(photon_events[:, 1], np.arange(size)[::-1])
    assert_array_almost_equal(photon_events[:, 2], np.ones(size))


def test_estimate_elastic_pos():
    """Test estimate_elastic_pos"""
    elastic_ind = 20
    y = np.arange(0, 50)
    x = rand(y.size)
    Iph = np.zeros_like(y)
    Iph[elastic_ind] = 1.
    photon_events = np.vstack((x, y, Iph)).T

    elastic_est = process2d.estimate_elastic_pos(photon_events)
    assert_array_almost_equal(y[elastic_ind], elastic_est)
