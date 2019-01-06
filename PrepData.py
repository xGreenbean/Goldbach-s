import numpy as np
def vec_bin_array(arr, m):
    """
    Arguments:
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int8's.
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[...,bit_ix] = fetch_bit_func(strs).astype("int8")

    return ret

def get_mod3_mod5_mod7(arr):
    """
    :param arr: array of positive integers
    :return: copy of arr with appended colums of,
    the one hot encoded resediue in mod3 , mod 5 and mod 7
    """
    # arr = [[divmod(x/2,3)[1] == 0,divmod(x,5)[1],divmod(x,7)[1],divmod(x,11)[1],divmod(x,13)[1],divmod(x,17)[1]] for x in arr]
    R_3 = np.zeros([len(arr), 3])
    R_5 = np.zeros([len(arr), 5])
    R_7 = np.zeros([len(arr), 7])
    R_11 = np.zeros([len(arr), 11])
    R_13 = np.zeros([len(arr), 13])
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

        r_11 = divmod(arr[i], 11)
        if (r_11[1] == 0):
            R_11[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        if (r_11[1] == 1):
            R_11[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        if (r_11[1] == 2):
            R_11[i] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        if (r_11[1] == 3):
            R_11[i] = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        if (r_11[1] == 4):
            R_11[i] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        if (r_11[1] == 5):
            R_11[i] = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        if (r_11[1] == 6):
            R_11[i] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        if (r_11[1] == 7):
            R_11[i] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        if (r_11[1] == 8):
            R_11[i] = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        if (r_11[1] == 9):
            R_11[i] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if (r_11[1] == 10):
            R_11[i] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    return np.c_[np.c_[np.c_[R_3,R_5],R_7],R_11]

def append_bias_reshape(features,labels):
    n_training_samples = features.shape[0]
    n_dim = features.shape[1]
    f = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim + 1])
    l = np.reshape(labels,[n_training_samples,1])
    return f, l


class DataGeneratorSeq(object):

    def __init__(self, partitions, batch_size, num_unroll):
        self._partitions = partitions
        self._partitions_length = len(self._partitions) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._partitions_length // self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):

        batch_data = np.zeros((self._batch_size), dtype=np.float32)
        batch_labels = np.zeros((self._batch_size), dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b] + 1 >= self._partitions_length:
                # self._cursor[b] = b * self._segments
                self._cursor[b] = np.random.randint(0, (b + 1) * self._segments)

            batch_data[b] = self._partitions[self._cursor[b]]
            batch_labels[b] = self._partitions[self._cursor[b] + np.random.randint(0, 5)]

            self._cursor[b] = (self._cursor[b] + 1) % self._partitions_length

        return batch_data, batch_labels

    def unroll_batches(self):

        unroll_data, unroll_labels = [], []
        init_data, init_label = None, None
        for ui in range(self._num_unroll):
            data, labels = self.next_batch()

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0, min((b + 1) * self._segments, self._partitions_length - 1))