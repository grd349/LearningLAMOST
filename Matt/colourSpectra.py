from fits import Spectrum
import matplotlib.pyplot as plt

spectrum = Spectrum('/data2/mrs493/DR1_3/spec-55860-B6001_sp12-198.fits')

fig, ax = plt.subplots(figsize = (18,12))

spectrum.plotFlux(log = False, ax = ax, colour = 'k')

cBands = {'cB':[3980,4920], 'c#820BBB':[5070,5950],'cR':[5890,7270],'c#2E0854':[7310,8810]}

ax.set_ylim(plt.ylim())
ax.set_xlim(plt.xlim())

ax.set_xlabel('Wavelength / Angstroms')
ax.set_ylabel('Flux')

for col in cBands:
    ax.fill_betweenx(plt.ylim(), cBands[col][0], cBands[col][1], facecolor = col[1:], alpha=0.5)