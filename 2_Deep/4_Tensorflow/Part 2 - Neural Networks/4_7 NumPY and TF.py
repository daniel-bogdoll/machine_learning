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

#If you change the values of the Tensor, the values of the ndarray will not change, and vice-versa.

b = b * 40

print(b)
print(a)    # NumPy array stays the same
print(c)    # NumPy array stays the same

a = a + 1

print(a)
print(b)    # Tensor stays the same
print(c)    # NumPy array stays the same