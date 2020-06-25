import csv
import numpy as np
import matplotlib.pyplot as plt

results = []
with open("results2.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader, None)
    for row in reader:
        print (row)
        results.append([int(row[1]), float(row[3])])

results = np.array(results)
plt.plot(results[:, 0], results[:, 1], label="FF_norm")
plt.xlabel("#cells")
plt.ylabel("Norma")
plt.legend()
plt.savefig("FFnorm2.png")
