import argparse
import re
import os, glob, datetime, time
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter
from tqdm import tqdm

from HSIDCNN import HSIDCNN
from AODN import AODN
from testing import quantitative_assess, denoise

parser = argparse.ArgumentParser(description='PyTorch dense att')
parser.add_argument('--model', default='AODN_att_both', type=str, help='choose a type of model')

parser.add_argument('--batch_size', default=64, type=int, help='batch size')

parser.add_argument('--train_data', default='train_data/train_data_sigma_rand100_aug_ls.mat', type=str, help='path of train data')
parser.add_argument('--test_data_DC', default='test_data/rand/DC_test_sigma_rand100_1.mat', type=str, help='path of test data DC')
# parser.add_argument('--test_data_Paiva', default='test_data/Paiva_test_sigma100.mat', type=str, help='path of test data Paiva')

# parser.add_argument('--sigma', default=50, type=int, help='noise level')
parser.add_argument('--sigma', default='rand100', type=str, help='noise level')
parser.add_argument('--epoch', default=800, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-5, type=float, help='initial learning rate for Adam')
parser.add_argument('--K', default=64, type=int, help='K adjacent bands')
parser.add_argument('--dense_num', default=4, type=int, help='number of conv in one densenet block ')
parser.add_argument('--b_num', default=6, type=int, help='number of conv in one densenet block ')

parser.add_argument('--alpha', default=0.2, type=float, help='alpha of OCT')



args = parser.parse_args()

batch_size = args.batch_size
n_epoch = args.epoch
sigma = args.sigma
k = int(args.K/2)
dense_num =args.dense_num

cuda = torch.cuda.is_available()
print("cuda is",cuda)
save_dir = os.path.join('model', 'sigma_'+str(args.sigma)+args.model+'_al_'+str(args.alpha)+'d_num'+str(dense_num)+
                        'b_num'+str(args.b_num)+'_K'+str(args.K)+"_lr_"+str(args.lr)+"_bs_"+str(batch_size))
writer = SummaryWriter('runs/'+'sigma_'+str(args.sigma)+args.model+'_al_'+str(args.alpha)+'d_num'+str(dense_num)+
                       'b_num' + str(args.b_num) +'_K'+str(args.K)+"_lr_"+str(args.lr)+"_bs_"+str(batch_size))
# milestone=[5,20 ,40, 60, 80, 100]
milestone=[80,160,240]
# milestone=[800]
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

#寻找到最终的一个检查点
def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

#显示训练时间等等
def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


#主函数
if __name__ == '__main__':
    # model selection
    print('linear stretch 2%')
    print('===> Building model')
    print(args.model,'sigma',args.sigma, ' alpha: ',args.alpha,' dense_num: ',
          dense_num,' block_num: ', args.b_num, ' K: ', args.K, ' lr ', args.lr," bs ", batch_size )

    elif args.model=='AODN':
        model=AODN(dense_num=dense_num, alpha=args.alpha, block_num=args.b_num, K=args.K, feature_Channel=20, layer_output=80)


    #TEST DATA
    test_data_DC = scipy.io.loadmat(args.test_data_DC)
    test_data_clean_DC = test_data_DC['clean']
    test_data_noise_DC = test_data_DC['noise']

    data = scipy.io.loadmat(args.train_data)
    data_clean = data['clean']
    data_noise = data['noise']

    number, heigth, width, band = data_clean.shape
    data = np.zeros((2, number, heigth, width, band))
    for n in range(number):
        data[0, n, :, :, :] = data_clean[n, :, :, :]
        data[1, n, :, :, :] = data_noise[n, :, :, :]
    data = torch.from_numpy(data.transpose((1, 0, 4, 2, 3)))

    criterion = nn.MSELoss(reduction='sum')



    #CONTINUE
    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
        if os.path.isfile(os.path.join(save_dir, 'checkpoint_%03d.pth' % initial_epoch)):
            checkpoint = torch.load(os.path.join(save_dir, 'checkpoint_%03d.pth' % initial_epoch))
            optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr)
            start_epoch = checkpoint['epoch']  
            scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=0.4, last_epoch=checkpoint['epoch'])
            scheduler.load_state_dict(checkpoint['lr_schedule'])
    model.train()

    if cuda:
        model = model.cuda()
        # vgg_p = vgg_p.cuda()
        # device_ids = [0]
        # model = nn.DataParallel(model, device_ids=device_ids).cuda()
        # criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=0.4)  # learning rates
    
    for epoch in range(initial_epoch, n_epoch):

        model.train()
        lr = optimizer.state_dict()['param_groups'][0]['lr']

        DLoader = DataLoader(dataset=data, num_workers=0, drop_last=False, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()
        try:
            with tqdm(DLoader,ncols=100,ascii=True) as t:
                for n_count, batch_yx in enumerate(t):
                    # print(n_count)
                    optimizer.zero_grad()
                    if cuda:
                        batch_x, batch_y = batch_yx[:,0,:,:].cuda(), batch_yx[:,1,:,:].cuda()
                        

                    iter_band = np.arange(band)
                    np.random.shuffle(iter_band)
                    for b in iter_band:

                        x = batch_y[:,b, :, :]
                        noise_free = batch_x[:,b, :, :]

                        x = torch.unsqueeze(x, dim=1).float()
                        noise_free = torch.unsqueeze(noise_free, dim=1).float()
                        # print(x.shape)
                        # print(noise_free.shape)
                        if b < k:

                            y = batch_y[:,0:args.K, :, :]

                        elif b < band - k:

                            y = torch.cat((batch_y[:,b - k:b, :, :],
                                                batch_y[:,b + 1:b + k + 1, :, :]),1)

                        else:
                            y = batch_y[:,band - args.K:band, :, :]
                        y = torch.unsqueeze(y, dim=1).float()

                        learned_image=model(x, y)
                        # learned_image = model(x)
                        mse_loss = criterion(learned_image, noise_free)
                        loss = mse_loss
                        epoch_loss += loss.item()
                        loss.backward()
                        optimizer.step()
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()
        scheduler.step()

        elapsed_time = time.time() - start_time
        writer.add_scalar('loss', epoch_loss, epoch + 1)
        writer.add_scalar('lr', lr, epoch + 1)
        if (epoch + 1) % 5 == 0 or (epoch<100):
            testmodel=model.eval()

            img_out,denoise_time_DC = denoise(test_data_noise_DC,testmodel,args.K)
            aft_psnr_DC,aft_ssim_DC,sam_dc = quantitative_assess(test_data_clean_DC, img_out)

            writer.add_scalar('psnr_DC', aft_psnr_DC, epoch + 1)
            writer.add_scalar('ssim_DC', aft_ssim_DC, epoch + 1)

            aft_psnr =  aft_psnr_DC
            aft_ssim =  aft_ssim_DC
            writer.add_scalar('psnr', aft_psnr, epoch + 1)
            writer.add_scalar('ssim', aft_ssim, epoch + 1)

            batch_number = data.size(0) // batch_size
            log('epcoh = %4d ,train loss = %d ,lr=%2.8f,psnr=%2.4f ssim=%2.4f, time = %4.2f s' %
                    (epoch + 1, epoch_loss / batch_number,lr,aft_psnr,aft_ssim, elapsed_time))


            torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))

            checkpoint = {
                          'optimizer': optimizer.state_dict(),
                          "epoch": epoch,
                          'lr_schedule': scheduler.state_dict()
                        }
            torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_%03d.pth' % (epoch + 1)))
            # torch.save(scheduler.state_dict(), os.path.join(save_dir, 'scheduler.pth'))



