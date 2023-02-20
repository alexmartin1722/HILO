# 1. Input M(n,m) matrix, N a number less than n, M a number
# 2. Generate N random numbers less than n
# 3. Select N rows of M to generate a matrix A(N,m)
# 4. Calculate Sum(A(N,i)) for each i,
# 5. Record in Matrix J(M,m)
# 6. Calculate Sum(A(N,i)^2)
# 7. Record in Matrix K(M,m)
# 8. Go to 2
# 9. Output J and K

import random
import numpy as np
import argparse

def bootstrap(matrix, n: int, m: int):
    """
    Args:
        Matrix: A matrix of size n*m
        N: A number less than n
        M: A number, the number of times to run the bootstrap
    Returns:
        J: A matrix of size M*m
        K: A matrix of size M*m
    """

    assert(n < matrix.shape[0])
    assert(m > 0)

    J = np.zeros((m, matrix.shape[1]))
    K = np.zeros((m, matrix.shape[1]))

    for i in range(m):
        randoms = random.sample(range(0, matrix.shape[0]), n)
        print('Randoms', randoms, 'Iteration', i)
        A = matrix[randoms, :]
        print('A', A)

        # for each column, sum the values
        J[i, :] = np.sum(A, axis=0)
        print('J\n', J)

        # for each column, sum the squared values
        K[i, :] = np.sum(np.square(A), axis=0)
        print('K\n', K)
    return J, K

def testing():
    matrix = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20],[21,22,23,24,25,26,27,28,29,30],[31,32,33,34,35,36,37,38,39,40],[41,42,43,44,45,46,47,48,49,50],[51,52,53,54,55,56,57,58,59,60],[61,62,63,64,65,66,67,68,69,70],[71,72,73,74,75,76,77,78,79,80],[81,82,83,84,85,86,87,88,89,90],[91,92,93,94,95,96,97,98,99,100],[101,102,103,104,105,106,107,108,109,110]])
    print('Matrix=\n',matrix)
    
    n = 5
    m = 3
    print('N=',n)
    print('M=',m)
    J, K = bootstrap(matrix, n, m)
    print('J=\n',J)
    print('K=\n',K)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', type=str, required=True, help='Path to matrix file')
    parser.add_argument('--n', type=int, required=True, help='Number of rows to sample')
    parser.add_argument('--m', type=int, required=True, help='Number of times to run bootstrap')
    parser.add_argument('--out-dir', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()


    #matrix is csv or txt
    if args.matrix.endswith('.csv'):
        matrix = np.loadtxt(args.matrix, delimiter=',')
    else:
        matrix = np.loadtxt(args.matrix)
    
    J, K = bootstrap(matrix, args.n, args.m)
    
    #write J and K to file
    np.savetxt(args.out_dir + '/J.csv', J, delimiter=',')
    np.savetxt(args.out_dir + '/K.csv', K, delimiter=',')




if __name__ == "__main__":
    main()




