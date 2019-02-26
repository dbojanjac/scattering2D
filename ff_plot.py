# Far Field Pattern plot

# Function call: python3 pplot.py folder_name far_field_file
# ie. python3 pplot.py results ff_isotropic

# Input variables
        # folder_name: folder containing far field file
        # file_name: far field pattern file vontainig two columns  in format
        #               (angle far_field_amplitude

# Plot axis are hardcoded as
    # ax.set_rmax(0);  ax.set_rmin(-10)
    # ax.set_rticks([-10, -5, 0, 2])

import matplotlib.pyplot as plt
import numpy as np
import sys

def file_len(fname):
    '''File length calculator'''

    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

folder_name = sys.argv[1]
file_name = sys.argv[2]
FF_n = file_len(folder_name + '/' + file_name)

ifile = open(folder_name + '/' + file_name, 'r')

phi = [0] * FF_n;   FF = [0] * FF_n;    m = 0

# Read line by line from far_field_file and split it into (angle far_field_amplitude)
for line in ifile:
    entry = line.split()
    phi[m] = float(entry[0])
    FF[m] = float(entry[1])
    m +=1

# Put maximum at 0dB
FF_max = abs(max(np.log10(FF)))
FF = 10 * (np.log10(FF) + FF_max)

fig = plt.figure()
ax = fig.add_subplot(111, projection = 'polar')
ax.plot(phi, FF, label = file_name, c = 'b')
plt.legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3,
           ncol = 1, mode = "expand", borderaxespad = 0.)
ax.set_rmax(0);  ax.set_rmin(-10)
ax.set_rticks([-10, -5, 0, 2])

plt.show()
