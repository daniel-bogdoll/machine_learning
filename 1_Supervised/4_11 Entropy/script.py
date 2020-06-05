from math import log,log2
import numpy as np

def entropy(objects):
    size = 0
    entropy = 0
    for object in objects:
        size += object[0]   #two loops are not so nice
    for object in objects:
        p = object[0]/size #probability
        entropy += p*log2(p)
    entropy = -1 * entropy
    print(entropy)

objects = [[8,'red'],[3,'blue'],[2,'yellow']]
entropy(objects)