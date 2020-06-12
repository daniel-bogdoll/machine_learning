import numpy as np
import tensorflow as tf

tf.random.set_seed(7) 

a = np.random.rand(4,3)
b = tf.convert_to_tensor(a)

print(type(a))
print(a)
print(b)

c = b.numpy()

print(type(c))
print(c)

b = b * 40

print(b)
print(a)
print(c)

a = a + 1

print(a)
print(b)
print(c)