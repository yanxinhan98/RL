import os
print(os.path.join(os.path.dirname(os.path.realpath(__file__)), "archive", "train", "1", "2", "3", "4")) #.. join as many items as desired
import numpy as np
b = np.arange(0,4)
print(b)
prob = [1/len(b) for i in b]
a = np.random.choice(b, p=prob)
print(a)

if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), "ResNet_flower_out")):
    os.mkdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "ResNet_flower_out"))

a = np.zeros((4,3))
print(np.argmax(a[2, :]))

actions = [0,1,2]
states = [0,1]
visited = [[False for i in range(0, len(actions))] for i in range(0, len(states))]
print(visited)

tableQ = np.zeros((2, 3))
tableQ[0][2] = 3
a = np.where(tableQ == np.amax(tableQ))[1][0]
print(a)

a = "RLMC"
b = a[:2]
print(b)