import numpy as np
import pdb
from PIL import Image

path2groundtruth = './ground_truth/200_130/shape01.png'
path2last_phi_mat = 'last_phi_mat_1.npy'

img = Image.open(path2groundtruth).convert('LA')
ground_truth = np.asarray(img)[:,:,0] / 255
last_phi_mat = np.load(path2last_phi_mat)

ind = np.zeros_like(last_phi_mat)
ind[last_phi_mat < 0] = 1.0

# relative error calculate in percent ( % )
error_area = np.abs(ground_truth - ind)
relative_error = 100 * np.sum(error_area) / np.sum(ground_truth)
msg = f'Relative error: {np.round(relative_error,3):>3} %'
print(msg)

import matplotlib.pyplot as plt 
plt.figure()
plt.title(msg)
plt.imshow(error_area, cmap='bone_r')
plt.show()
