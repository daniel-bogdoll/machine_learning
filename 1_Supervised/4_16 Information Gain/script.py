# Compute information gains
from math import log, log2

import numpy as np
import pandas as pd

mobug = 10
lobug = 14

brown = 6
blue = 10
green = 8

entropy = []

brownA = np.empty((0, 3))
brownB = np.empty((0, 3))

blueA = np.empty((0, 3))
blueB = np.empty((0, 3))

greenA = np.empty((0, 3))
greenB = np.empty((0, 3))

smallA = np.empty((0, 3))
smallB = np.empty((0, 3))

mediumA = np.empty((0, 3))
mediumB = np.empty((0, 3))


def calculateEntropy(objects, size):
    entropy = 0
    for object in objects:
        p = object[0]/size  # probability
        entropy += p*log2(p)
    entropy = -1 * entropy
    print(objects, size, entropy)
    return entropy

def countAnimals(nparray):
    mobug = len(np.where(nparray == 'Mobug')[0])
    lobug = len(np.where(nparray == 'Lobug')[0])

    # Compute entropy
    animals = [[mobug, 'mobug'], [lobug, 'lobug']]
    entropy = calculateEntropy(animals, (mobug + lobug))
    return [entropy, mobug + lobug]

data = pd.read_csv('ml-bugs.csv').to_numpy()
for point in data:
    if point[1] == 'Brown':
        brownA = np.vstack([brownA, point])
    else:
        brownB = np.vstack([brownB, point])

    if point[1] == 'Blue':
        blueA = np.vstack([blueA, point])
    else:
        blueB = np.vstack([blueB, point])

    if point[1] == 'Green':
        greenA = np.vstack([greenA, point])
    else:
        greenB = np.vstack([greenB, point])

    if point[2] < 17:
        smallA = np.vstack([smallA, point])
    else:
        smallB = np.vstack([smallB, point])

    if point[2] < 20:
        mediumA = np.vstack([mediumA, point])
    else:
        mediumB = np.vstack([mediumB, point])

def entropyGain(parent, child1, child2):
    sizeTotal = parent[1]
    sizeChild1 = child1[1]
    sizeChild2 = child2[1]
    return (parent[0] - (sizeChild1/sizeTotal* child1[0] + sizeChild2/sizeTotal * child2[0]))
    #Slides are wrong: It#s not 0.5 * (childA + childB) but their respective set size!

entropyTotal = countAnimals(data)
print("MY TOTAL",entropyTotal)

entropyBrownA = countAnimals(brownA)
entropyBrownB = countAnimals(brownB)
entropyGainBrown = entropyGain(entropyTotal, entropyBrownA, entropyBrownB)

entropyBlueA = countAnimals(blueA)
entropyBlueB = countAnimals(blueB)
entropyGainBlue = entropyGain(entropyTotal, entropyBlueA, entropyBlueB)

entropyGreenA = countAnimals(greenA)
entropyGreenB = countAnimals(greenB)
entropyGainGreen = entropyGain(entropyTotal, entropyGreenA, entropyGreenB)

entropySmallA = countAnimals(smallA)
entropySmallB = countAnimals(smallB)
entropyGainSmall = entropyGain(entropyTotal, entropySmallA, entropySmallB)

entropyMediumA = countAnimals(mediumA)
entropyMediumB = countAnimals(mediumB)
entropyGainMedium = entropyGain(entropyTotal, entropyMediumA, entropyMediumB)

print(entropyGainBrown)
print(entropyGainBlue)
print(entropyGainGreen)
print(entropyGainSmall)
print(entropyGainMedium)

def two_group_ent(first, tot):                        
    return -(first/tot*np.log2(first/tot) + (tot-first)/tot*np.log2((tot-first)/tot))

tot_ent = two_group_ent(10, 24)                       
g17_ent = 15/24 * two_group_ent(11,15) + 9/24 * two_group_ent(6,9)                  

print("THEIR TOTAL",tot_ent)
print 

answer = tot_ent - g17_ent     