from __future__ import division
import os
import logging
import time
import glob
import datetime

import numpy as np
from scipy.io import loadmat, savemat

import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import utils as util
from collections import OrderedDict

from config import opt
from DataLoader import DataLoader_Imagenet_val

from model.noise_model import AugmentNoise
from model.Mask import Masker
from model.arch_unet import UNet

# 获取系统时间、初始化计数器变量、设置可见的GPU设备和设置PyTorch的线程数
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
torch.set_num_threads(6)

# 创建保存路径
opt.save_path = os.path.join(opt.save_model_path, opt.log_name, systime)
os.makedirs(opt.save_path, exist_ok=True)

# 配置日志记录以及创建日志记录器
util.setup_logger(
    "train",
    opt.save_path,
    "train_" + opt.log_name,
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("train")


def save_network(network, epoch, name):
    save_path = os.path.join(opt.save_path, 'models')
    os.makedirs(save_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_path = os.path.join(save_path, model_name)
    if isinstance(network, nn.DataParallel) or isinstance(
        network, nn.parallel.DistributedDataParallel
    ):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)
    logger.info('Checkpoint saved to {}'.format(save_path))


def load_network(load_path, network, strict=True):
    assert load_path is not None
    logger.info("Loading model from [{:s}] ...".format(load_path))
    if isinstance(network, nn.DataParallel) or isinstance(
        network, nn.parallel.DistributedDataParallel
    ):
        network = network.module
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith("module."):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)
    return network

def save_state(epoch, optimizer, scheduler):
    """Saves training state during training, which will be used for resuming"""
    save_path = os.path.join(opt.save_path, 'training_states')
    os.makedirs(save_path, exist_ok=True)
    state = {"epoch": epoch, "scheduler": scheduler.state_dict(), 
                                            "optimizer": optimizer.state_dict()}
    save_filename = "{}.state".format(epoch)
    save_path = os.path.join(save_path, save_filename)
    torch.save(state, save_path)

def resume_state(load_path, optimizer, scheduler):
    """Resume the optimizers and schedulers for training"""
    resume_state = torch.load(load_path)
    epoch = resume_state["epoch"]
    resume_optimizer = resume_state["optimizer"]
    resume_scheduler = resume_state["scheduler"]
    optimizer.load_state_dict(resume_optimizer)
    scheduler.load_state_dict(resume_scheduler)
    return epoch, optimizer, scheduler

def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


class DataLoader_SIDD_Medium_Raw(Dataset):
    def __init__(self, data_dir):
        super(DataLoader_SIDD_Medium_Raw, self).__init__()
        self.data_dir = data_dir
        # get images path
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = loadmat(fn)["x"]
        # random crop
        H, W = im.shape
        CSize = 256
        rnd_h = np.random.randint(0, max(0, H - CSize))
        rnd_w = np.random.randint(0, max(0, W - CSize))
        im = im[rnd_h : rnd_h + CSize, rnd_w : rnd_w + CSize]
        im = im[np.newaxis, :, :]
        im = torch.from_numpy(im)
        return im

    def __len__(self):
        return len(self.train_fns)


def get_SIDD_validation(dataset_dir):
    val_data_dict = loadmat(
        os.path.join(dataset_dir, "ValidationNoisyBlocksRaw.mat"))
    val_data_noisy = val_data_dict['ValidationNoisyBlocksRaw']
    val_data_dict = loadmat(
        os.path.join(dataset_dir, 'ValidationGtBlocksRaw.mat'))
    val_data_gt = val_data_dict['ValidationGtBlocksRaw']
    num_img, num_block, _, _ = val_data_gt.shape
    return num_img, num_block, val_data_noisy, val_data_gt


def validation_kodak(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def validation_bsd300(dataset_dir):
    fns = []
    fns.extend(glob.glob(os.path.join(dataset_dir, "test", "*")))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def validation_Set14(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref, data_range=255.0):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(data_range**2 / np.mean(np.square(diff)))
    return psnr


# Training Set
TrainingDataset = DataLoader_Imagenet_val(opt.data_dir, patch=opt.patchsize)
TrainingLoader = DataLoader(dataset=TrainingDataset,
                            num_workers=8,
                            batch_size=opt.batchsize,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)

# Validation Set
Kodak_dir = os.path.join(opt.val_dirs, "Kodak24")
BSD300_dir = os.path.join(opt.val_dirs, "BSD300")
Set14_dir = os.path.join(opt.val_dirs, "Set14")
# valid_dict = {
#     "Kodak24": validation_kodak(Kodak_dir),
#     "BSD300": validation_bsd300(BSD300_dir),
#     "Set14": validation_Set14(Set14_dir)
# }
valid_dict = {
    "Kodak24": validation_kodak(Kodak_dir)
}

# Noise adder
noise_adder = AugmentNoise(style=opt.noisetype)

# Masker
masker = Masker(width=4, mode='interpolate', mask_type='all')

# Network
network = UNet(in_channels=opt.n_channel,
                out_channels=opt.n_channel,
                wf=opt.n_feature)
if opt.parallel:
    network = torch.nn.DataParallel(network)
network = network.cuda()

# about training scheme
num_epoch = opt.n_epoch
ratio = num_epoch / 100
optimizer = optim.Adam(network.parameters(), lr=opt.lr,
                       weight_decay=opt.w_decay)
scheduler = lr_scheduler.MultiStepLR(optimizer,
                                     milestones=[
                                         int(20 * ratio) - 1,
                                         int(40 * ratio) - 1,
                                         int(60 * ratio) - 1,
                                         int(80 * ratio) - 1
                                     ],
                                     gamma=opt.gamma)
print("Batchsize={}, number of epoch={}".format(opt.batchsize, opt.n_epoch))

# Resume and load pre-trained model
epoch_init = 1
if opt.resume is not None:
    epoch_init, optimizer, scheduler = resume_state(opt.resume, optimizer, scheduler)
if opt.checkpoint is not None:
    network = load_network(opt.checkpoint, network, strict=True)

# temp
if opt.checkpoint is not None:
    epoch_init = 60
    for i in range(1, epoch_init):
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
        logger.info('----------------------------------------------------')
        logger.info("==> Resuming Training with learning rate:{}".format(new_lr))
        logger.info('----------------------------------------------------')

print('init finish')

if opt.noisetype in ['gauss25', 'poisson30']:
    Thread1 = 0.8
    Thread2 = 1.0
else:
    Thread1 = 0.4
    Thread2 = 1.0

Lambda1 = opt.Lambda1
Lambda2 = opt.Lambda2
increase_ratio = opt.increase_ratio

for epoch in range(epoch_init, opt.n_epoch + 1):
    cnt = 0

    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

    network.train()
    for iteration, clean in enumerate(TrainingLoader):
        st = time.time()
        clean = clean / 255.0
        clean = clean.cuda()
        noisy = noise_adder.add_train_noise(clean)

        optimizer.zero_grad()

        net_input, mask = masker.train(noisy)
        noisy_output = network(net_input)
        n, c, h, w = noisy.shape
        noisy_output = (noisy_output*mask).view(n, -1, c, h, w).sum(dim=1)
        diff = noisy_output - noisy

        with torch.no_grad():
            exp_output = network(noisy)
        exp_diff = exp_output - noisy

        # g25, p30: 1_1-2; frange-10
        # g5-50 | p5-50 | raw; 1_1-2; range-10
        Lambda = epoch / opt.n_epoch
        if Lambda <= Thread1:
            beta = Lambda2
        elif Thread1 <= Lambda <= Thread2:
            beta = Lambda2 + (Lambda - Thread1) * \
                (increase_ratio-Lambda2) / (Thread2-Thread1)
        else:
            beta = increase_ratio
        alpha = Lambda1

        revisible = diff + beta * exp_diff
        loss_reg = alpha * torch.mean(diff**2)
        loss_rev = torch.mean(revisible**2)
        loss_all = loss_reg + loss_rev

        loss_all.backward()
        optimizer.step()
        logger.info(
            '{:04d} {:05d} diff={:.6f}, exp_diff={:.6f}, Loss_Reg={:.6f}, Lambda={}, Loss_Rev={:.6f}, Loss_All={:.6f}, Time={:.4f}'
            .format(epoch, iteration, torch.mean(diff**2).item(), torch.mean(exp_diff**2).item(),
                    loss_reg.item(), Lambda, loss_rev.item(), loss_all.item(), time.time() - st))

    scheduler.step()

    if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch:
        network.eval()
        # save checkpoint
        save_network(network, epoch, "model")
        save_state(epoch, optimizer, scheduler)
        # validation
        save_model_path = os.path.join(opt.save_model_path, opt.log_name,
                                       systime)
        validation_path = os.path.join(save_model_path, "validation")
        os.makedirs(validation_path, exist_ok=True)
        np.random.seed(101)
        valid_repeat_times = {"Kodak24": 10, "BSD300": 3, "Set14": 20}

        for valid_name, valid_images in valid_dict.items():
            avg_psnr_dn = []
            avg_ssim_dn = []
            avg_psnr_exp = []
            avg_ssim_exp = []
            avg_psnr_mid = []
            avg_ssim_mid = []
            save_dir = os.path.join(validation_path, valid_name)
            os.makedirs(save_dir, exist_ok=True)
            repeat_times = valid_repeat_times[valid_name]
            for i in range(repeat_times):
                for idx, im in enumerate(valid_images):
                    origin255 = im.copy()
                    origin255 = origin255.astype(np.uint8)
                    im = np.array(im, dtype=np.float32) / 255.0
                    noisy_im = noise_adder.add_valid_noise(im)
                    if epoch == opt.n_snapshot:
                        noisy255 = noisy_im.copy()
                        noisy255 = np.clip(noisy255 * 255.0 + 0.5, 0,
                                           255).astype(np.uint8)
                    # padding to square
                    H = noisy_im.shape[0]
                    W = noisy_im.shape[1]
                    val_size = (max(H, W) + 31) // 32 * 32
                    noisy_im = np.pad(
                        noisy_im,
                        [[0, val_size - H], [0, val_size - W], [0, 0]],
                        'reflect')
                    transformer = transforms.Compose([transforms.ToTensor()])
                    noisy_im = transformer(noisy_im)
                    noisy_im = torch.unsqueeze(noisy_im, 0)
                    noisy_im = noisy_im.cuda()
                    with torch.no_grad():
                        n, c, h, w = noisy_im.shape
                        net_input, mask = masker.train(noisy_im)
                        noisy_output = (network(net_input) *
                                        mask).view(n, -1, c, h, w).sum(dim=1)
                        dn_output = noisy_output.detach().clone()
                        # Release gpu memory
                        del net_input, mask, noisy_output
                        torch.cuda.empty_cache()
                        exp_output = network(noisy_im)
                    pred_dn = dn_output[:, :, :H, :W]
                    pred_exp = exp_output.detach().clone()[:, :, :H, :W]
                    pred_mid = (pred_dn + beta*pred_exp) / (1 + beta)

                    # Release gpu memory
                    del exp_output
                    torch.cuda.empty_cache()

                    pred_dn = pred_dn.permute(0, 2, 3, 1)
                    pred_exp = pred_exp.permute(0, 2, 3, 1)
                    pred_mid = pred_mid.permute(0, 2, 3, 1)

                    pred_dn = pred_dn.cpu().data.clamp(0, 1).numpy().squeeze(0)
                    pred_exp = pred_exp.cpu().data.clamp(0, 1).numpy().squeeze(0)
                    pred_mid = pred_mid.cpu().data.clamp(0, 1).numpy().squeeze(0)

                    pred255_dn = np.clip(pred_dn * 255.0 + 0.5, 0,
                                         255).astype(np.uint8)
                    pred255_exp = np.clip(pred_exp * 255.0 + 0.5, 0,
                                          255).astype(np.uint8)
                    pred255_mid = np.clip(pred_mid * 255.0 + 0.5, 0,
                                          255).astype(np.uint8)

                    # calculate psnr
                    psnr_dn = calculate_psnr(origin255.astype(np.float32),
                                             pred255_dn.astype(np.float32))
                    avg_psnr_dn.append(psnr_dn)
                    ssim_dn = calculate_ssim(origin255.astype(np.float32),
                                             pred255_dn.astype(np.float32))
                    avg_ssim_dn.append(ssim_dn)

                    psnr_exp = calculate_psnr(origin255.astype(np.float32),
                                              pred255_exp.astype(np.float32))
                    avg_psnr_exp.append(psnr_exp)
                    ssim_exp = calculate_ssim(origin255.astype(np.float32),
                                              pred255_exp.astype(np.float32))
                    avg_ssim_exp.append(ssim_exp)

                    psnr_mid = calculate_psnr(origin255.astype(np.float32),
                                              pred255_mid.astype(np.float32))
                    avg_psnr_mid.append(psnr_mid)
                    ssim_mid = calculate_ssim(origin255.astype(np.float32),
                                              pred255_mid.astype(np.float32))
                    avg_ssim_mid.append(ssim_mid)

                    # visualization
                    if i == 0 and epoch == opt.n_snapshot:
                        save_path = os.path.join(
                            save_dir,
                            "{}_{:03d}-{:03d}_clean.png".format(
                                valid_name, idx, epoch))
                        Image.fromarray(origin255).convert('RGB').save(
                            save_path)
                        save_path = os.path.join(
                            save_dir,
                            "{}_{:03d}-{:03d}_noisy.png".format(
                                valid_name, idx, epoch))
                        Image.fromarray(noisy255).convert('RGB').save(
                            save_path)
                    if i == 0:
                        save_path = os.path.join(
                            save_dir,
                            "{}_{:03d}-{:03d}_dn.png".format(
                                valid_name, idx, epoch))
                        Image.fromarray(pred255_dn).convert(
                            'RGB').save(save_path)
                        save_path = os.path.join(
                            save_dir,
                            "{}_{:03d}-{:03d}_exp.png".format(
                                valid_name, idx, epoch))
                        Image.fromarray(pred255_exp).convert(
                            'RGB').save(save_path)
                        save_path = os.path.join(
                            save_dir,
                            "{}_{:03d}-{:03d}_mid.png".format(
                                valid_name, idx, epoch))
                        Image.fromarray(pred255_mid).convert(
                            'RGB').save(save_path)

            avg_psnr_dn = np.array(avg_psnr_dn)
            avg_psnr_dn = np.mean(avg_psnr_dn)
            avg_ssim_dn = np.mean(avg_ssim_dn)

            avg_psnr_exp = np.array(avg_psnr_exp)
            avg_psnr_exp = np.mean(avg_psnr_exp)
            avg_ssim_exp = np.mean(avg_ssim_exp)

            avg_psnr_mid = np.array(avg_psnr_mid)
            avg_psnr_mid = np.mean(avg_psnr_mid)
            avg_ssim_mid = np.mean(avg_ssim_mid)

            log_path = os.path.join(validation_path,
                                    "A_log_{}.csv".format(valid_name))
            with open(log_path, "a") as f:
                f.writelines("epoch:{},dn:{:.6f}/{:.6f},exp:{:.6f}/{:.6f},mid:{:.6f}/{:.6f}\n".format(
                    epoch, avg_psnr_dn, avg_ssim_dn, avg_psnr_exp, avg_ssim_exp, avg_psnr_mid, avg_ssim_mid))
