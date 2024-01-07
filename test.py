import numpy as np
from matplotlib import pyplot as plt
data = np.loadtxt(open("results/res.csv","rb"),delimiter=",", skiprows=1)
print(data.shape)

plt.plot(data[:, 0], data[:, 1])
plt.title('loss')
plt.savefig('image/loss.png', bbox_inches='tight', dpi=500)

plt.cla()
plt.plot(data[:, 0], data[:, 2:])
plt.title('metrics')
plt.legend(labels=['precision', 'recall', 'f1 value'])
plt.savefig('image/acc.png', bbox_inches='tight', dpi=500)
