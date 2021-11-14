import scipy.io
import scipy.ndimage
import numpy as np
import math
from libtiff import TIFF
import tifffile
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from matplotlib import pyplot as plt
from testing import quantitative_assess
import random

patch_size ,stride =40,40 
noise_sigma=100

aug_times = 1
scales = [2.0,1.5,1,0.5]
# scales = [1]

dict = r'train_data/dc.tif'
dict_Pavia = r'Pavia.mat'
dict_Center = r'PaviaU.mat'
dict_train = r'train_data/'
dict_test = r'test_data/'


def data_aug(img,rot_time,filp_mode):

  if filp_mode == -1:
    return np.rot90(img, k=rot_time)
  else:
    return np.flip(np.rot90(img, k=rot_time), axis=filp_mode)


def truncated_linear_stretch(image, truncated_value=2, maxout=1, min_out=0):
    
    def gray_process(gray, maxout = maxout, minout = min_out):
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray_new = (gray - truncated_down) / ((truncated_up - truncated_down) / (maxout - minout))
        gray_new[gray_new < minout] = minout
        gray_new[gray_new > maxout] = maxout
        return np.float32(gray_new)

    image= np.float32(image)
    height,width,band = image.shape
    out =np.zeros((height,width,band))
    
    for b in range(band):
        out[:,:,b] = gray_process(image[:,:,b])
    return out

def noise_add_by_bands(data):

    h,w,c=data.shape
    noise=[]
    for channel in range(c):
        sigma=random.randint(1,noise_sigma)
        noise.append(np.random.normal(scale=(sigma / 255), size=[h,w]).astype(np.float32))
    noise=np.array(noise)
    noise=noise.transpose(1, 2, 0)
    noise_image=np.clip(data + noise, 0,
                           1).astype(np.float32)
    return noise_image

def aug(data_train, count):
    for s in scales:
        print(s)
        data_scaled = scipy.ndimage.zoom(data_train, (s, s, 1)).astype(np.float32)
        data_scaled[data_scaled<0]=0
        data_scaled[data_scaled>1]=1


        print("data_scaled:",data_scaled.shape)
        h_scaled, w_scaled,band_scaled = data_scaled.shape
        noise_scaled = np.clip(data_scaled + np.random.normal(scale=(noise_sigma / 255), size=data_scaled.shape), 0, 1).astype(np.float32)
        # noise_scaled = noise_add_by_bands(data_scaled)
        for i in range(0, h_scaled- patch_size + 1, stride):
            for j in range(0, w_scaled- patch_size + 1, stride):
              for k in range(0, aug_times):
                count += 1
                x = data_scaled[i:i+patch_size, j:j+patch_size , :]
                y = noise_scaled[i:i+patch_size, j:j+patch_size , :]
                rot_time = np.random.randint(0, 4)
                filp_mode = np.random.randint(-1, 2)
                x_aug = data_aug(x,rot_time,filp_mode)
                y_aug = data_aug(y,rot_time,filp_mode)
                x_np = np.array(x_aug, dtype='float32')
                y_np = np.array(y_aug, dtype='float32')
                clean.append(x_np)
                noise.append(y_np)
        print(count)
    return  count
data_all = tifffile.imread(dict)
data_all = data_all.transpose(1,2,0)
data_all = data_all.astype(np.float32)
data_all = truncated_linear_stretch(data_all)





plt.imshow(data_all[600:800,50:250, (57, 27, 17)])
plt.show()
data_test = data_all[600:800,50:250,:]


noise_test = (data_test + np.random.normal(scale=(noise_sigma/255), size=data_test.shape)).astype(np.float32)

print('quantitative_assess', quantitative_assess(data_test, noise_test))

fig = plt.figure()
fig.add_subplot(121)
plt.imshow(data_test[:, :, (57, 27, 17)])
fig.add_subplot(122)
plt.imshow(noise_test[:, :, (57, 27, 17)])
plt.show()

data_train1 = data_all[0:600,:,:]
data_train2 = data_all[800:1280,:,:]
data_train_DC =np.concatenate((data_train1,data_train2),axis=0)


clean = []
noise = []
count = 0
count=aug(data_train_DC,count)
count=aug(data_train_Paiva,count)
print('augment finished')
print('there are:'+str(count)+'in total')

# scipy.io.savemat(dict_train+'train_data_sigma'+str(noise_sigma)+'_aug_ls.mat', {'clean': clean,'noise': noise})
scipy.io.savemat(dict_test+'test'+'.mat', {'clean': data_test,'noise': noise_test})

def generate_rand():
    for i in range(10):
        noise_test = noise_add_by_bands(data_test)
        scipy.io.savemat(dict_test + 'DC_test_sigma_rand' + str(noise_sigma) +'_'+str(i+1)+ '.mat',
                         {'clean': data_test, 'noise': noise_test})

# generate_rand()