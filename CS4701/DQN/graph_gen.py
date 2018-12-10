import numpy as np
import matplotlib.pyplot as plt

c_natural = np.genfromtxt("/Users/MagdBayoumi/Downloads/c_natural.csv", delimiter=",")
r_natural = np.genfromtxt("/Users/MagdBayoumi/Downloads/r_natural.csv", delimiter=",")
c_dueling = np.genfromtxt("/Users/MagdBayoumi/Downloads/c_dueling.csv", delimiter=",")
r_dueling = np.genfromtxt("/Users/MagdBayoumi/Downloads/r_dueling.csv", delimiter=",")
c_PRmem = np.genfromtxt("/Users/MagdBayoumi/Downloads/c_PRmem.csv", delimiter=",")
r_PRmem = np.genfromtxt("/Users/MagdBayoumi/Downloads/r_PRmem.csv", delimiter=",")

plt.figure(1)
plt.plot(np.array(c_natural), c='r', label='natural')
plt.plot(np.array(c_dueling), c='b', label='dueling')
plt.plot(np.array(c_PRmem), c='k', label='PRmem')
plt.legend(loc='best')
plt.ylabel('cost')
plt.xlabel('training steps')
plt.grid()
plt.savefig("foo.png",bbox_inches='tight')

plt.figure(2)
plt.plot(np.array(r_natural), c='r', label='natural')
plt.plot(np.array(r_dueling), c='b', label='dueling')
plt.plot(np.array(r_PRmem), c='k', label='PRmem')
plt.legend(loc='best')
plt.ylabel('accumulated reward')
plt.xlabel('training steps')
plt.grid()

plt.savefig("foo2.png",bbox_inches='tight')