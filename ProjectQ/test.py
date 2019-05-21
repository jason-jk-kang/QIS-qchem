import matplotlib.pyplot as plt

one = [1,2,3,4]
two = [10,14,16,17]

f3 = plt.figure(4)
plt.plot(one, two, '-', color='orange')
plt.ylabel('Velocity')
plt.xlabel('Time in au')

plt.show()
