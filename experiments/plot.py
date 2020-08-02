import matplotlib.pyplot as plt
filename = 'agent_reward_all.txt'

X,rew0,rew1,rew2 = [],[],[],[]

with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        value = [float(s) for s in line.strip().split()]
        rew0.append(value[0])
        rew1.append(value[1])
        rew2.append(value[2])
        
X = range(len(rew0))

plt.figure(0)
plt.plot(X, rew0)

plt.figure(1)
plt.plot(X, rew1)

plt.figure(2)
plt.plot(X, rew2)

plt.show()