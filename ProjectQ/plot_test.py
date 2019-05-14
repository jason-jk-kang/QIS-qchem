import matplotlib.pyplot as plt


one = [1, 2, 3, 4, 5]
two = list(reversed(one))

three = [6,7,8,9,10]
four = list(reversed(three))


f1 = plt.figure(0)
plt.plot(one, one, 'x-')
plt.plot(one, two, 'o-')
plt.ylabel('Energy in Hartree')
plt.xlabel('Bond length in angstrom')

plt.show()

f2 = plt.figure(1)
plt.plot(one, three, 'x-')
plt.plot(one, four, 'o-')
plt.ylabel('Energy in Hartree')
plt.xlabel('Bond length in angstrom')

plt.show()
