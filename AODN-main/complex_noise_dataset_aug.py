import scipy.io
import scipy.ndimage
import numpy as np
import math
from libtiff import TIFF
import tifffile
from matplotlib import pyplot as plt
from testing import quantitative_assess
import random

# normal setting
aug_times = 1
scales = [2.0, 1.5, 1, 0.5]
patch_size, stride = 40, 40  

dict = r'train_data/dc.tif'
dict_train = r'train_data/'
dict_test = r'/home/scuse/database/sp/'

noise_sigma = 100

# add Poisson noise if need
Poisson_noise = 0
# add stripe noise if need
stripe_noise = 0
# add dead line if need
dead_line = 0

# add salt and paper if need
salt_paper = 0

PEAK_max_value = 5

min_amount = 0.05
max_amount = 0.15
list = range(191)
band_number_DL = 10
band_number_stripe = 10

noise_str = ''

if Poisson_noise == 1:
    noise_str += "P_"
if dead_line == 1:
    noise_str += "DL_"
if stripe_noise == 1:
    noise_str += "S_"

if salt_paper ==1:
    noise_str += "sp_"

def data_aug(img, rot_time, filp_mode):
    if filp_mode == -1:
        return np.rot90(img, k=rot_time)
    else:
        return np.flip(np.rot90(img, k=rot_time), axis=filp_mode)



def truncated_linear_stretch(image, truncated_value=2, maxout=1, min_out=0):

    def gray_process(gray, maxout=maxout, minout=min_out):
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray_new = (gray - truncated_down) / ((truncated_up - truncated_down) / (maxout - minout))
        gray_new[gray_new < minout] = minout
        gray_new[gray_new > maxout] = maxout
        return np.float32(gray_new)

    image = np.float32(image)
    height, width, band = image.shape
    out = np.zeros((height, width, band))

    for b in range(band):
        out[:, :, b] = gray_process(image[:, :, b])
    return out

def sp_noise(image):
    prob = 0.025
    thres = 1 - prob
    for i in range(image.shape[0]):

        for j in range(image.shape[1]):

            rdn = random.random()
            if rdn < prob:
                image[i][j] = 0

            elif rdn > thres:
                image[i][j] = 1

    return image


def noise_add_by_bands(data):
    h, w, c = data.shape
    noise = []
    if stripe_noise == 1:
        S_band = random.sample(list, band_number_stripe)
        if 57 not in S_band:
            S_band.append(57)
        if 17 not in S_band:
            S_band.append(17)
        if 27 not in S_band:
            S_band.append(27)
    if dead_line == 1:
        DL_band = random.sample(list, band_number_DL)
        if 2 not in DL_band:
            DL_band.append(2)

    for channel in range(c):
        sigma = random.randint(1, noise_sigma)

        # add gaussian noise
        gaussian_band = np.random.normal(scale=(sigma / 255), size=[h, w])
        noise_band = gaussian_band
        # add Poisson noise if need
        if Poisson_noise == 1:
            PEAK = random.randint(1, PEAK_max_value)
            image = data[:, :, channel]
            poisson_band = np.random.poisson(image  * PEAK) / PEAK 
            noise_band += poisson_band.astype(np.float32)
        # add stripe noise if need
        if stripe_noise == 1:
            if channel in S_band:
                number = random.randint(int(0.05 * w), int((0.15 * w)))
                loc = random.sample(range(h), number)

                stripe_noise_band = np.zeros((h, w))
                for i in loc:
                    stripe = random.random()
                    stripe_noise_band[i, :] = stripe_noise_band[i, :] - stripe
                noise_band += stripe_noise_band
        noise.append(noise_band)

        # add dead line if need
        if dead_line == 1:
            if channel in DL_band:
                number = random.randint(int(0.05 * w), int((0.15 * w)))
                loc = random.sample(range(h), number)
                DL_noise_band = np.zeros((h, w))
                for i in loc:
                    DL_noise_band[i, :] = -1
                noise_band += DL_noise_band

    noise = np.array(noise)
    noise = noise.transpose(1, 2, 0)
    if salt_paper ==1:
        for channel in range(c):
            data[:, :, channel]= sp_noise(data[:, :, channel])
    #noise_image = np.clip(data + noise, 0, 1).astype(np.float32)
    noise_image=(data + noise).astype(np.float32)
    return noise_image


def aug(data_train, count):
    for s in scales:
        print(s)
        data_scaled = scipy.ndimage.zoom(data_train, (s, s, 1)).astype(np.float32)
        data_scaled[data_scaled < 0] = 0
        data_scaled[data_scaled > 1] = 1

        print("data_scaled:", data_scaled.shape)
        h_scaled, w_scaled, band_scaled = data_scaled.shape
        # noise_scaled = noise_add_by_bands(data_scaled)
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                for k in range(0, aug_times):
                    count += 1
                    x = data_scaled[i:i + patch_size, j:j + patch_size, :]

                    rot_time = np.random.randint(0, 4)
                    filp_mode = np.random.randint(-1, 2)
                    x_aug = data_aug(x, rot_time, filp_mode)
                    
                    y_aug = noise_add_by_bands(x_aug)
                    x_np = np.array(x_aug, dtype='float32')
                    y_np = np.array(y_aug, dtype='float32')
                    clean.append(x_np)
                    noise.append(y_np)
        print(count)
    return count


data_all = tifffile.imread(dict)
data_all = data_all.transpose(1, 2, 0)
data_all = data_all.astype(np.float32)
data_all = truncated_linear_stretch(data_all)
data_test = data_all[600:800, 50:250, :].astype(np.float32)

# noise_test = noise_add_by_bands(data_test)
noise_test =data_test+np.clip(data_clean + np.random.normal(scale=(50 / 255), size=data_clean.shape), 0,
                                 1).astype(np.float32)
print('quantitative_assess', quantitative_assess(data_test, noise_test))
exit()
fig = plt.figure()
fig.add_subplot(121)
plt.imshow(data_test[:, :, (2)],cmap='gray')
fig.add_subplot(122)
plt.imshow(noise_test[:, :, (2)],cmap='gray')
plt.show()

data_train1 = data_all[0:600, :, :]
data_train2 = data_all[800:1280, :, :]
data_train_DC = np.concatenate((data_train1, data_train2), axis=0)


def generate_rand():
    for i in range(10):
        noise_test = noise_add_by_bands(data_test)
        scipy.io.savemat(dict_test + 'DC_test_sigma_rand' + str(noise_sigma) + '_' + noise_str + str(i + 1) + '.mat',
                         {'clean': data_test, 'noise': noise_test})

def generate_visual():
    for i in range(10):
        noise_test = noise_add_by_bands(data_test)
        scipy.io.savemat(dict_test + 'visual'  + '_' + noise_str + str(i + 1) + '.mat',
                         {'clean': data_test, 'noise': noise_test})
# generate_rand()
clean = []
noise = []
count = 0
# count=aug(data_train_DC,count)
print('augment finished')
print('there are:' + str(count) + 'in total')

scipy.io.savemat(dict_train + 'train_data_sigma' + str(noise_sigma) + noise_str + '.mat',
                 {'clean': clean, 'noise': noise})