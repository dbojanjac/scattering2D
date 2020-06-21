import csv
import numpy as np
import matplotlib.pyplot as plt

results = []
with open("results.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader, None)
    for row in reader:
        print (row)
        results.append([int(row[0]), float(row[1]), float(row[2])])

results = np.array(results)
plt.plot(results[:, 0], results[:, 1], label="E_field_norm")
plt.plot(results[:, 0], results[:, 2], label="FF_norm")
plt.xlabel("Broj elemenata u ćeliji")
plt.ylabel("Norma")
plt.legend()
plt.savefig("FFnorm.png")
