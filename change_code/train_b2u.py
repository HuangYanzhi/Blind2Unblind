from __future__ import division
import os
import logging
import time

import datetime

import numpy as np

from PIL import Image
import torch

import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader

import utils as util

from config import opt
from data_preprocess.DataLoader import DataLoader_Imagenet_val

from model.noise_model import AugmentNoise
from model.Mask import Masker
from model.arch_unet import UNet
from model.utils import save_network, save_state, resume_state, load_network

from val.val_function import validation_kodak, validation_bsd300, validation_Set14
from val.val_loss import calculate_psnr, calculate_ssim


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

valid_dict = {
    "Kodak24": validation_kodak(Kodak_dir)
#    "BSD300": validation_bsd300(BSD300_dir),
#    "Set14": validation_Set14(Set14_dir)
}

# Noise adder
noise_adder = AugmentNoise(style=opt.noisetype)

# Masker
masker = Masker(width=4, mode='interpolate', mask_type='all')

# Network
network = UNet(in_channels=opt.n_channel,
                out_channels=opt.n_channel,
                wf=opt.n_feature)

network = network.cuda()

# 设置训练的学习率调度方案（training scheme）
# 将在指定的里程碑处更新学习率，从而在训练过程中动态地调整学习率的大小
ratio = opt.n_epoch / 100
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

# 恢复并加载预训练模型
epoch_init = 1
if opt.resume is not None:
    epoch_init, optimizer, scheduler = resume_state(opt.resume, optimizer, scheduler)
if opt.checkpoint is not None:
    network = load_network(logger, opt.checkpoint, network, strict=True)

# 从检查点（checkpoint）恢复训练时，从指定的训练轮数开始，并更新学习率
if opt.checkpoint is not None:
    epoch_init = 60
    for i in range(1, epoch_init):
        scheduler.step() # # 更新学习率
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

        # 计算可逆项（revisible）以及相关的损失函数
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

    scheduler.step() # 更新学习率

    if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch:
        network.eval()
        # save checkpoint
        save_network(logger,network, epoch, "model")
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
