import matplotlib.pyplot as plt
import scipy as sp

def relu(x):
    if x<0:
        return 0
    else:
        return x

x = sp.linspace(-10,10, 1000)

plt.subplots()
ax = plt.axes(frameon=False)

ax.axhline(0, c = 'k')
ax.axvline(0, c = 'k')

ax.plot(x, [relu(y) for y in x], lw = 2)
ax.set_xlabel('x')
ax.set_ylabel('f_a(x)')
plt.xticks([]," ")
plt.yticks([]," ")
l = ax.get_xlim()
ax.set_ylim(l)