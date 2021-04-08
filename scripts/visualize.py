import numpy as np
import matplotlib 
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')
import matplotlib.pyplot as plt
train_data = np.load('/playpen1/scribble/ssy/data/Levine_32_matrix_train.npy')
train_data = train_data[:, :-1]
inds = np.random.permutation(train_data.shape[0])
from tqdm import tqdm
#train_data = torch.Tensor(train_data[inds]).to(device)

test_data = np.load('/playpen1/scribble/ssy/data/Levine_32_matrix_test.npy')
test_label = test_data[:, -1] - 1
test_data = test_data[:, :-1]
#test_data = torch.Tensor(test_data).to(device)
n_clusters = int(np.max(test_label)) + 1
D = test_data.shape[1]
for i in range(D):
    for j in range(i+1, D):
        for k in range(j+1, D):
            print(i,j,k)
            fig = plt.figure(figsize=(15,10))
            for c in range(n_clusters):
                ax = fig.add_subplot(3,5,c+1, projection='3d')
                idx = np.where(test_label == c)
                ax.scatter(test_data[idx,i],test_data[idx,j], test_data[idx,k])
                ax.set_xlim(-7, 7)
                ax.set_ylim(-7, 7)
                ax.set_zlim(-7, 7)
            ax = fig.add_subplot(3,5,15, projection='3d')
            ax.scatter(test_data[:,i],test_data[:,j], test_data[:,k])
            ax.set_xlim(-7, 7)
            ax.set_ylim(-7, 7)
            ax.set_zlim(-7, 7)
            plt.savefig(f'/playpen1/scribble/ssy/log/cell_viz/{i}_{j}_{k}.png')
            plt.close()



