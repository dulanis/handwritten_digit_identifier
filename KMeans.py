import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
samples = digits.data
model = KMeans(n_clusters = 10, random_state = 42)
model.fit(samples)
fig = plt.figure(figsize=(8, 3))

fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')
for i in range(10):

  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()
new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,2.60,7.47,7.31,3.36,0.00,0.00,0.00,0.00,5.94,6.53,5.81,7.62,4.87,0.13,0.00,0.00,5.55,5.64,0.07,4.41,7.62,2.03,0.00,0.00,1.57,3.35,3.05,4.00,7.62,2.21,0.00,0.45,7.52,7.62,7.62,7.61,7.62,2.17,0.00,0.75,7.62,6.57,7.27,7.61,5.33,1.22,0.00,0.07,4.71,5.33,4.60,2.21,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.35,3.05,3.81,3.18,0.15,0.00,0.00,0.00,3.19,7.62,7.24,7.62,6.30,1.44,0.00,0.00,3.80,7.61,0.07,2.66,7.06,7.16,1.82,0.00,3.64,7.61,0.89,0.00,1.06,6.85,5.54,0.00,1.52,7.62,3.20,0.00,0.53,6.47,6.08,0.00,0.00,6.84,7.16,5.92,7.37,7.54,2.64,0.00,0.00,1.13,5.03,5.31,4.17,1.57,0.00,0.00],
[0.00,5.68,7.62,7.62,7.62,6.66,1.98,0.00,0.75,7.62,4.49,1.83,3.10,6.54,7.21,0.00,0.00,3.96,1.67,0.00,0.00,3.81,7.62,0.00,0.00,0.00,0.60,3.42,3.80,5.99,7.31,0.00,0.00,0.00,4.24,7.62,7.62,7.62,7.39,2.86,0.00,0.00,0.83,2.28,2.13,1.70,6.77,5.33,0.97,7.39,4.49,0.91,0.97,2.66,7.16,5.08,0.30,4.94,7.61,7.62,7.62,7.62,6.75,1.50],
[0.00,2.11,3.05,3.05,3.04,2.43,2.21,0.20,0.00,6.48,7.62,7.62,7.62,7.62,7.62,1.98,0.00,0.38,0.15,0.00,0.30,4.33,7.62,0.44,0.00,0.20,2.21,2.28,2.05,6.62,7.20,3.66,0.00,1.98,7.62,7.62,7.62,7.62,7.62,7.54,0.00,0.07,1.30,1.29,7.36,5.80,0.00,0.00,0.00,0.00,0.00,2.66,7.62,2.86,0.00,0.00,0.00,0.00,0.00,5.70,6.74,0.27,0.00,0.00]
])
new_labels = model.predict(new_samples)
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
print()
