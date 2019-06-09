import numpy as np


# mnist file format label, pix-11, pix-12, pix-13, ...

# define two numpy arrays
a = np.array([[1,2],[3,4]])
b = np.array([[1,1],[1,1]])
print(a)

print(b)


# train using mnist_train.csv
    # index [i]0 is y
    # index [i][1:] is x


# initialize netral net variables
    # m X 1 array     