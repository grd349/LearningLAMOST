import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import norm
from matplotlib.patches import Rectangle
import matplotlib.mlab as mlab

#matplotlib.rcParams.update({'font.size': 15})

n = 1000

w = sp.linspace(-5, 5, n)

continuum = [10]*n

diff = mlab.normpdf(w, 0, 1)

#fig, ax = plt.subplots(figsize = (10,8))
fig, ax = plt.subplots()

S = ax.plot(w, continuum - diff, c = 'k', label = 'Spectrum')[0]
C = ax.plot(w, continuum, ls = '--', c = 'r', label = 'Continuum')[0]

ax.set_xlabel('Wavelength \ Angstroms')
ax.set_ylabel('Flux')
plt.xticks([]," ")
plt.yticks([]," ")
plt.ylim([9.5, 10.2])
ax.fill_between(w, continuum, continuum - diff, alpha = 0.5, color = 'y', label = 'Line Area')
ax.fill_between(w[400:600], 10, 0, alpha = 0.5, color = 'r', label = '')
#ax.set_yticklabels([])
#ax.get_yaxis().set_visible(False)
rect1 = Rectangle((0, 0), 1, 1, fc="y", alpha=0.5, label = 'Line Area')
rect2 = Rectangle((0, 0), 1, 1, fc="r", alpha=0.5, label = 'Rectangle')
plt.legend([S, C, rect1, rect2], ['Spectrum', 'Continuum', 'Line Area', 'Rectangle'], loc = 4)
plt.tight_layout()
plt.savefig('eqWex.pdf')
plt.show()
