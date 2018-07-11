from rixs import process1d
import numpy as np
import pandas as pd

spectra = process1d.make_fake_spectra()
align_min = 10
align_max = 70
first_spectrum_name = spectra.columns[0]
shifts = process1d.get_shifts(spectra, spectra[first_spectrum_name],
                              align_min=align_min, align_max=align_max,
                              background=0.5)

aligned_spectra = process1d.apply_shifts(spectra, shifts)

aligned_spectra_aligned_region = aligned_spectra.loc[align_min:align_max, :]

ref = aligned_spectra_aligned_region[first_spectrum_name].values 
for name, _ in aligned_spectra_aligned_region.iteritems():
    spec = aligned_spectra_aligned_region[name].values
    np.testing.assert_almost_equal(spec, ref)


#elastic_energies = np.linspace(-0.2, 0.2, 10)
#energy_loss = np.linspace(-2, 10, 5000)
#
#def make_spectrum(energy_loss, elastic_energy, name):
#    y = np.exp(-(energy_loss-elastic_energy)**2/(0.1))
#    return pd.Series(y, index=energy_loss, name=name)
#
#fake_spectra = pd.concat([make_spectrum(energy_loss, elastic_energy, "{}".format(i))
#                          for i, elastic_energy in enumerate(elastic_energies)],
#                         axis=1)
#                          
#                          
#shifts = process1d.get_shifts(fake_spectra, fake_spectra['0'])
#
#apply_shifts