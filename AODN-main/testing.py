import argparse
import os

import numpy as np
import scipy.io
import torch
import time
import matplotlib.pyplot as plt
import random
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from block.SAM import sam


parser = argparse.ArgumentParser(description='PyTorch casaoct')
parser.add_argument('--model', default='AODN', type=str, help='choose a type of model')
# parser.add_argument('--test_data', default='Indian_pines_truncated.mat', type=str, help='path of test data')
# parser.add_argument('--test_data', default='test_data/Indian_pines_corrected.mat', type=str, help='path of test data')
parser.add_argument('--test_data', default='test_data/test.mat', type=str, help='path of test data')
parser.add_argument('--result_path', default='result/rand100', type=str, help='path of result data ')
parser.add_argument('--save', default=0, type=int, help='save denoise image or not')
parser.add_argument('--show', default=1, type=int, help='show denoise image or not')
parser.add_argument('--K', default=128, type=int, help='K adjacent bands')
# parser.add_argument('--sigma', default=100, type=int, help='noise level')
parser.add_argument('--sigma', default=100, type=int, help='noise level')
parser.add_argument('--CPU_ONLY', default=0, type=int, help='only use CPU')
args = parser.parse_args()

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


def AODN():

    models = torch.load('model/proposed(K=64)/sigma_100/model.pth')

    print('AODN')
    models.cuda()
    models.eval()
    return models


def testmodel(input_key):
    function_map={

        'AODN': AODN,

    }
    return function_map[input_key]()

def noise_add_by_bands(data):

    h,w,c=data.shape
    noise=[]
    for channel in range(c):
        sigma=random.randint(0,args.sigma)
        noise.append(np.random.normal(scale=(sigma / 255), size=[h,w]))
    noise=np.array(noise)
    noise=noise.transpose(1, 2, 0)
    noise_image=np.clip(data + noise, 0,
                           1).astype(np.float32)
    return noise_image


def denoise(data_noise,model,K=24,cuda=1):
    hight,width,cs=data_noise.shape
    data_noise = data_noise.astype(np.float32)
    data_noise = data_noise.transpose((2, 1, 0))
    # print(data_noise.shape)
    test_out = np.zeros(data_noise.shape).astype(np.float32)

    if cuda:
        model = model.cuda()
    k = int(K / 2)
    start_time = time.time()
    for channel_i in range(cs):
        x = data_noise[channel_i, :, :]

        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x)
        if channel_i < k:
            # print(channel_i)
            y = data_noise[0:K, :, :]


        elif channel_i < cs - k:

            y = np.concatenate((data_noise[channel_i - k:channel_i, :, :],
                                data_noise[channel_i + 1:channel_i + k + 1, :, :]))

            # print(y.shape)
        else:
            y = data_noise[cs - K:cs, :, :]

        y = np.expand_dims(y, axis=0)
        y = np.expand_dims(y, axis=0)

        y = torch.from_numpy(y)
        if cuda:
            x=x.cuda()
            y=y.cuda()
        with torch.no_grad():

            # out = generator(x.cuda(), y.cuda())
            out = model(x, y)
            # out = model(x)
        out = out.squeeze(0)

        out = out.data.cpu().numpy()

        test_out[channel_i, :, :] = out
    test_out = test_out.transpose((2, 1, 0))

    end_time =time.time()
    timetotal =end_time-start_time
    print('denoise time: %2.4f' % timetotal)
    return [test_out,timetotal]

def quantitative_assess(data_clean,test_out):
    psnrs=[]
    ssims=[]

    height,width,band =data_clean.shape
    for b in range(band):
        psnr1 = peak_signal_noise_ratio(data_clean[:, :, b], test_out[:, :, b],data_range=1)
        ssim1 = structural_similarity(data_clean[:, :, b], test_out[:, :, b],win_size=11,data_range=1,gaussian_weights=1)

        psnrs.append(psnr1)
        ssims.append(ssim1)
    avg_psnr = np.mean(psnrs)
    avg_ssim = np.mean(ssims)
    Sam=sam(data_clean,test_out)
    return [avg_psnr,avg_ssim,Sam]


def show_image(data_noise,test_out,data_clean):
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    plt.imshow(data_noise[:, :, (57, 27, 17)])
    plt.axis('off')  
    plt.title('noise image')  

    ax2 = fig.add_subplot(132)
    plt.imshow(test_out[:, :, (57, 27, 17)])
    plt.axis('off') 
    plt.title('denoise image') 

    ax3 = fig.add_subplot(133)
    plt.imshow(data_clean[:, :, (57, 27, 17)])
    plt.axis('off')  
    plt.title('clean image')  
    plt.show()

def save_result(test_out,path,modelname):
    scipy.io.savemat(path + '/' + modelname + '_sigma' + str(args.sigma)+'.mat'.format('result_' + modelname),
                     {'data': test_out})
    #  scipy.io.savemat(path + '/' + modelname + '_sigma_rand' + str(args.sigma)+'.mat'.format('result_' + modelname),
    #                   {'data': test_out})
    # scipy.io.savemat(path + '/' + modelname + '_IP_real.mat'.format('result_' + modelname),
    #                  {'data': test_out})
    # scipy.io.savemat(path + '/' + modelname + '_PU_real.mat'.format('result_' + modelname),
    #                  {'paviaU': test_out})

if __name__ == '__main__':

    args = parser.parse_args()

    K = args.K
    CPU_ONLY = args.CPU_ONLY
    data_path = args.test_data
    result_path =args.result_path
    data = scipy.io.loadmat(data_path)
    data_clean = data['clean']
    # data_clean = data['paviaU']
    # data= data['indian_pines']
    # data = np.insert(data, 0, values=0, axis=1)
    # data_clean = np.insert(data, 0, values=0, axis=0)

    # data_clean= truncated_linear_stretch(data['indian_pines_corrected'])
    # data_clean = data['clean'][400:600, 50:250, :]
    height, width, band = data_clean.shape
    cuda = (torch.cuda.is_available()) & (CPU_ONLY == 0)
    model = testmodel(args.model)

    if cuda == 0:
        print('cpu only')
    psnrs=[]
    ssims=[]
    sams=[]
    times=[]
    print('sigma:',args.sigma)


    for i in range(1):
        #
        # data_noise= data_clean
        if args.sigma == -1:
            rand_path = 'test_data/rand'
            data_noise = scipy.io.loadmat(os.path.join(rand_path,'DC_test_sigma_rand100_' + str(i + 1) + '.mat'))['noise']
        # data_noise = noise_add_by_bands(data_clean)
        # data_noise = data['noise']
        else:
            data_noise = np.clip(data_clean + np.random.normal(scale=(args.sigma / 255), size=data_clean.shape), 0,
                                 1).astype(np.float32)
        print(data_noise.shape)
        test_out,totaltime= denoise(data_noise,model,K,cuda)

        print(data_clean.shape)
        print(test_out.shape)
        results = quantitative_assess(data_clean, test_out)

        print(' the %d result:' % (i+1))
        print(' PSNR after denoising: %.4f' % results[0])
        print(' SSIM after denoising: %.4f' % results[1])
        print(' SAM after denoising: %.4f' % results[2])
        psnrs.append(results[0])
        ssims.append(results[1])
        sams.append(results[2])
        times.append(totaltime)
        if args.sigma == -1:
            rand_path = 'test_data/rand'
            save_result(test_out,result_path,args.model+'_'+str(i))
    mpsnr=np.mean(psnrs)
    mssim=np.mean(ssims)
    msam =np.mean(sams)
    mtime=np.mean(times)
    stdpsnr=np.std(psnrs,ddof=1)
    stdssim=np.std(ssims,ddof=1)
    stdsam = np.std(sams, ddof=1)
    stdtime = np.std(times, ddof=1)
    print('mean and std')
    print(' mPSNR : %.4f' % mpsnr)
    print(' stdPSNR : %.4f' % stdpsnr)
    print(' mSSIM : %.4f' % mssim)
    print(' stdSSIM : %.4f' % stdssim)
    print(' mSAM : %.4f' % msam)
    print(' stdSAM : %.4f' % stdsam)
    print(' mTime : %.4f' % mtime)
    print(' stdTime : %.4f' % stdtime)
    if args.save==1:
        print('start saving')
        # test_out = np.delete(test_out, 0, 0)
        # test_out = np.delete(test_out, 0, 1)
        # save_result(test_out,result_path,args.model)
        print('saving complete')
    if args.show==1:
        show_image(data_noise,test_out,data_clean)



