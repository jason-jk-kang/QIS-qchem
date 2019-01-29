name = "jerry"
a=[0.1]
b=[0.1]
c=[0.1, 0.2]
d=[0.1]
e=[0.1]


f = open('test.txt', 'a')

f.write("Results for {}: \n".format(name))

f.write("Optimal UCCSD Singlet Energy: ")
for item in a:
    f.write("{}".format(str(item)))
f.write("\n")

f.write("Optimal UCCSD Singlet Amplitudes: ")
for item in b:
    f.write("{} ".format(str(item)))
f.write("\n")

f.write("Classical CCSD Energy: ")
for item in c:
    f.write("{} ".format(str(item)))
f.write("Hartrees \n")

f.write("Classical CCSD Energy: ")
for item in c:
    f.write("{} ".format(str(item)))
f.write("Hartrees \n")

f.write("Exact FCI Energy: ")
for item in d:
    f.write("{} ".format(str(item)))
f.write("Hartrees \n")

f.write("Initial Energy of UCCSD with CCSD amplitudes: ")
for item in e:
    f.write("{} ".format(str(item)))
f.write("Hartrees \n \n")

f.close
