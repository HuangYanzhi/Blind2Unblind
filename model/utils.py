import os
import torch
from config import opt
import torch.nn as nn
from collections import OrderedDict


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator

def save_network(logger, network, epoch, name):
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


def load_network(logger, load_path, network, strict=True):
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


def checkpoint(net, epoch, name, systime):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))