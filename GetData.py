import numpy as np
import smtplib

from sklearn.preprocessing import OneHotEncoder
def loadfile():
    mymatrix = np.loadtxt('/home/ehud/PycharmProjects/Goldbach/Datasets/primes1.txt')
    mymatrix = mymatrix.astype(int)
    return mymatrix.ravel().tolist()
def find_Ppairs(prime_list):
    pairs_counter = 0
    prime_set = set(prime_list)
    limit = prime_list[-1]
    dataset = [[4,1],[6,1]]
    print(dataset)
    # 4194304
    for num in range(8,4194304,2):
        div = divmod(num,41943)
        if div[1] == 0:
            mat = np.matrix(dataset)
            print("%d" %div[0])
            with open(r'Datasets/dataset.txt', 'a+') as f:
                np.savetxt(f, mat, fmt='%i')
            dataset = []
        for p in prime_list:
            if(p > num/2):
                break
            else:
                if num-p in prime_set:
                    pairs_counter += 1
        dataset.append([num ,pairs_counter])
        pairs_counter = 0
    mat = np.matrix(dataset)
    with open(r'Datasets/dataset.txt', 'a+') as f:
        np.savetxt(f, mat, fmt='%i')

def get_mod3_mod5_mod7(arr):
    """
    :param arr: array of positive integers
    :return: copy of arr with appended colums of,
    the one hot encoded resediue in mod3 , mod 5 and mod 7
    """
    # arr = [[divmod(x,3)[1] == 0,divmod(x,5)[1],divmod(x,7)[1]] for x in arr]
    R_3 = np.zeros([len(arr), 3])
    R_5 = np.zeros([len(arr), 5])
    R_7 = np.zeros([len(arr), 7])
    for i in range(0, len(arr)):
        r_3 = divmod(arr[i],3)
        if(r_3[1] == 0):
            R_3[i] = [0,0,1]
        if (r_3[1] == 1):
            R_3[i] = [0, 1, 0]
        if (r_3[1] == 2):
            R_3[i] = [1, 0, 0]

        r_5 = divmod(arr[i], 5)
        if (r_5[1] == 0):
            R_5[i] = [0, 0, 0, 0, 1]
        if (r_5[1] == 1):
            R_5[i] = [0, 0, 0, 1, 0]
        if (r_5[1] == 2):
            R_5[i] = [0, 0, 1, 0, 0]
        if (r_5[1] == 3):
            R_5[i] = [0, 1, 0, 0, 0]
        if (r_5[1] == 4):
            R_5[i] = [1, 0, 0, 0, 0]

        r_7 = divmod(arr[i], 7)
        if (r_7[1] == 0):
            R_7[i] = [0, 0, 0, 0, 0, 0, 1]
        if (r_7[1] == 1):
            R_7[i] = [0, 0, 0, 0, 0, 1, 0]
        if (r_7[1] == 2):
            R_7[i] = [0, 0, 0, 0, 1, 0, 0]
        if (r_7[1] == 3):
            R_7[i] = [0, 0, 0, 1, 0, 0, 0]
        if (r_7[1] == 4):
            R_7[i] = [0, 0, 1, 0, 0, 0, 0]
        if (r_7[1] == 5):
            R_7[i] = [0, 1, 0, 0, 0, 0, 0]
        if (r_7[1] == 6):
            R_7[i] = [1, 0, 0, 0, 0, 0, 0]
    return np.c_[np.c_[np.c_[arr,R_3],R_5],R_7]
find_Ppairs(loadfile())
