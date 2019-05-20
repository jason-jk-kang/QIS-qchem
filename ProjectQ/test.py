fci_force_list = [1,2,3,4]
time = 0.1

clock = [time]
for x in range(len(fci_force_list) - 1):
    clock += [clock[-1] + time]

print(clock)
