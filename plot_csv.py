import csv
import numpy as np
import matplotlib.pyplot as plt

results = []
with open("results1.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader, None)
    for row in reader:
        print (row)
        results.append([int(row[0]), float(row[1]), float(row[2])])

results = np.array(results)
fig, ax = plt.subplot()
ax.plot(results[:, 0], results[:, 1], "--", label="E_field_norm")
ax.plot(results[:, 0], results[:, 2], "-", label="FF_norm")
ax.yscale("log")
ax.xlabel("Broj elemenata u ćeliji")
ax.ylabel("Norma")
ax.legend()
ax.savefig("FFnorm.png")
