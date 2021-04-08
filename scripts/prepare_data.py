import numpy as np
import pdb
from tqdm import tqdm
f = open('/content/drive/My Drive/COMP 991/Levine_32_matrix.csv')
data = f.readlines()
data = data[1:]
numpy_array = np.zeros((len(data), 33))
for i in tqdm(range(len(data))):
    split_data = data[i].strip('\n').split(',')
    if split_data[-1] == 'NaN':
        split_data[-1] = '0'
    array_data = np.array(split_data).astype('float')
    numpy_array[i] = array_data


inds = np.random.permutation(numpy_array.shape[0])
numpy_array = numpy_array[inds]
label = numpy_array[:, -1]

train_data = numpy_array[np.where(label==0)]
test_data = numpy_array[np.where(label>0)]
data_mean = np.mean(numpy_array, 0)
data_std = np.std(numpy_array, 0)
train_data = (train_data - data_mean[None,:]) / data_std[None, :]
test_data = (test_data - data_mean[None,:]) / data_std[None, :]
train_data[:, -1] = label[np.where(label==0)]
test_data[:, -1] = label[np.where(label>0)]


np.save('/content/drive/My Drive/COMP 991/Levine_32_matrix_train.npy', train_data)
np.save('/content/drive/My Drive/COMP 991/Levine_32_matrix_test.npy', test_data)