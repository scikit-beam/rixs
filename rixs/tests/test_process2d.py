from rixs import process2d
import numpy as np
from numpy.random import rand
from numpy.testing import assert_array_almost_equal


def test_curvature_fit():
    """Test curvature fit on simulated data."""
    fake_curvature = dict(x0=1000., x1=0.02, x2=0.)
    photon_events = process2d.make_fake_image(**fake_curvature, noise=0)
    vary_params = dict(x0=True, x1=True, x2=False)
    curvature = process2d.fit_curvature(photon_events,
                                        vary_params=vary_params)

    val = curvature['x1']
    guess = fake_curvature['x1']
    ratio = np.abs(val - guess) / (val + guess)
    assert(ratio < 0.05)


def test_extract():
    """Test extract on simulated data."""
    y_edges = np.arange(-0.5, 100.5, 1)
    y_centers = (y_edges[:-1] + y_edges[1:])/2
    x_centers = 100*rand(y_centers.size)

    Iph = np.exp(-(y_centers-50)**2)

    photon_events = np.vstack((x_centers, y_centers, Iph)).transpose()
    spectrum = process2d.extract(photon_events,
                                 curvature=dict(x0=0, x1=0, x2=0),
                                 bins=y_edges)

    assert_array_almost_equal(y_centers, spectrum[:, 0])
    assert_array_almost_equal(Iph, spectrum[:, 1])
