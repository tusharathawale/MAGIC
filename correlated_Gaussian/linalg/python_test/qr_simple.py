import numpy as np
from numpy import linalg as LA
# this code is from https://www.andreinc.net/2021/01/25/computing-eigenvalues-and-eigenvectors-using-qr-decomposition
# we use this for testing the correctness of c code

# A is a square random matrix of size n
n = 4
#A = np.random.rand(n, n)
'''
A = np.array([[0.00001,  0.00001,   0.00001,   0.00001],
                [0.00001,   20.000,   60.000,   40.000],
                [0.00001,   60.000 , 180.000 , 120.000],
                [0.00001,   40.000,  120.000,   80.000]])
'''
A = np.array([[0,  0,   0,   0],
                [0,   20.000,   60.000,   40.000],
                [0,   60.000 , 180.000 , 120.000],
                [0,   40.000,  120.000,   80.000]])


B = np.array([[0.200,    0.600,    0.400,    0.800],
                [0.600,    1.800,    1.200,    2.400],
                [0.400,    1.200,    0.800,    1.600],
                [0.800,    2.400,    1.600,    3.200]])


C = np.array([[1,-1,4,1],
                [1,4,-2,1],
                [1,4,2,1],
                [1,-1,0,1]])

D = np.array([  [1, 1, 0, 1, 0, 1, 0, 1],
                [1, 2, 1, 0, 1, 0, 1, 0],
                [0, 1, 3, 1, 0, 1, 0, 1],
                [1, 0, 1, 4, 1, 0, 1, 0],
                [0, 1, 0, 1, 5, 1, 0, 1],
                [1, 0, 1, 0, 1, 6, 1, 0],
                [0, 1, 0, 1, 0, 1, 7, 1],
                [1, 0, 1, 0, 1, 0, 1, 8]] )


# Adding this one into the qr tests
# for the linear algorithm blog
E = np.array(
[[0.0012813, 0.0007226, 0.0012581, 0.0007015],
 [0.0007226, 0.0006682, 0.0006721, 0.0006144],
 [0.0012581, 0.0006721, 0.0012851, 0.0006792],
 [0.0007015, 0.0006144, 0.0006792, 0.0006132]]
)

F = np.array([
[0.20, 0.60, 0.40],
[0.60, 1.80, 1.20],
[0.40, 1.20, 0.80]
])

G = np.array([
[0.0, 0.0, 0.0],
[0.0, 20.0, 60.0],
[0.0, 60.0, 180.0]
])

np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})


inputM = G

print("inputM=")
print((inputM))

Q, R = np.linalg.qr(inputM)
print("eigen q")
print(Q)
print("eigen r")
print(R)

# check eigen values
w, v = LA.eig(inputM)
print("eigen values")
print(w)
print("eigen vectors")
print(v)

'''
print("A=")
print((A))


def eigen_qr_simple(A, iterations=5):
    Ak = np.copy(A)
    n = A.shape[0]
    QQ = np.eye(n)
    for k in range(iterations):
        print(Ak)
        Q, R = np.linalg.qr(Ak)
        print("eigen q")
        print(Q)
        print("eigen r")
        print(R)

        Ak = R @ Q
        QQ = QQ @ Q
        # we "peek" into the structure of matrix A from time to time
        # to see how it looks
        if k%10000 == 0:
            print("A",k,"=")
            print((Ak))
            print("\n")
    return Ak, QQ

# We call the function    
eigen_qr_simple(A)

# We compare our results with the official numpy algorithm
print(np.linalg.eigvals(A))

# compute the eigen vectors based on eigen value
'''