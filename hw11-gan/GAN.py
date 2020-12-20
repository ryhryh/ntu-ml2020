

import numpy as np
import os
import random
import cv2
from matplotlib import pyplot as plt
from glob import glob
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

os.environ["CUDA_VISIBLE_DEVICES"]="0"



class FaceDataset(Dataset):
    def __init__(self, img_dir='/workdir/home/feynman52/NTU-ML2020/hw11-gan/datasets/faces', ):
        self.img_dir = img_dir
        self.setup()

    def setup(self):
        self.setup_data()
        self.setup_transform()

    def setup_data(self):
        self.data = glob(os.path.join(self.img_dir, '*'))[:]

    def setup_transform(self):
        transform_steps = [transforms.ToPILImage(),
                            transforms.Resize((64, 64)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ]
        self.transform_fn = transforms.Compose(transform_steps)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform_fn(img)
        return img


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class NetG(nn.Module):
    def __init__(self, in_dim, dim=64):
        super().__init__()

        self.dim = dim
        self.in_dim = in_dim
        self.setup_layer()  
        self.apply(weights_init)
    
    def setup_layer(self):
        self.l1 = nn.Sequential(nn.Linear(self.in_dim, self.dim * 8 * 4 * 4, bias=False),
                                nn.BatchNorm1d(self.dim * 8 * 4 * 4),
                                nn.ReLU())
        
        self.l2_5 = nn.Sequential(self.dconv_bn_relu(self.dim * 8, self.dim * 4),
                                    self.dconv_bn_relu(self.dim * 4, self.dim * 2),
                                    self.dconv_bn_relu(self.dim * 2, self.dim),
                                    nn.ConvTranspose2d(self.dim, 3, 5, 2, padding=2, output_padding=1),
                                    nn.Tanh())

    def dconv_bn_relu(self, in_dim, out_dim):
        convTranspose2d = nn.ConvTranspose2d(in_dim, out_dim, 
                                            5, 2,
                                            padding=2, 
                                            output_padding=1, 
                                            bias=False)
        net = nn.Sequential(convTranspose2d,
                            nn.BatchNorm2d(out_dim),
                            nn.ReLU())
        return net

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class NetD(nn.Module):
    def __init__(self, in_dim, dim=64):
        super().__init__()

        self.dim = dim
        self.in_dim = in_dim
        self.setup_layer()
        self.apply(weights_init)  
    
    def setup_layer(self):
        self.ls = nn.Sequential(nn.Conv2d(self.in_dim, self.dim, 5, 2, 2), 
                                nn.LeakyReLU(0.2),
                                self.conv_bn_lrelu(self.dim, self.dim * 2),
                                self.conv_bn_lrelu(self.dim * 2, self.dim * 4),
                                self.conv_bn_lrelu(self.dim * 4, self.dim * 8),
                                nn.Conv2d(self.dim * 8, 1, 4),
                                nn.Sigmoid())
        
    def conv_bn_lrelu(self, in_dim, out_dim):
        net = nn.Sequential(nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                            nn.BatchNorm2d(out_dim),
                            nn.LeakyReLU(0.2))
        return net

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y



from torch.autograd import Variable


class TrainAndTest:
    def __init__(self, option, ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.option = option
        self.setup()

    def setup(self):
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_criterion()
        self.setup_history()

    def setup_data(self):
        faceDataset = FaceDataset()
        self.train_loader = DataLoader(faceDataset, batch_size=self.option.batch_size, shuffle=True)

    def setup_model(self):
        self.netG = NetG(in_dim=self.option.z_dim).to(self.device) #!
        self.netD = NetD(3).to(self.device) #!

    def setup_optimizer(self):
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.option.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.option.lr, betas=(0.5, 0.999))

    def setup_criterion(self):
        self.criterion = nn.BCELoss()

    def setup_history(self,):
        self.df_history = pd.DataFrame()
        
    def train_one_epoch(self):
        loss_G = 0.0
        loss_D = 0.0
        self.netG.train()
        self.netD.train()

        for i, x_y in enumerate(self.train_loader):
            
            """ Train D """
            # img
            x = x_y.to(self.device)
            batch_size = len(x)
            z = Variable(torch.randn(batch_size, self.option.z_dim)).to(self.device)
            
            r_imgs = Variable(x)
            f_imgs = self.netG(z)
            
            # label
            r_label = torch.ones((batch_size)).to(self.device)
            f_label = torch.zeros((batch_size)).to(self.device)
            
            # D
            r_logit = self.netD(r_imgs)
            f_logit = self.netD(f_imgs)
            
            # compute loss
            r_loss = self.criterion(r_logit, r_label)
            f_loss = self.criterion(f_logit, f_label)
            batch_loss_D = (r_loss + f_loss) / 2

            self.optimizer_D.zero_grad()
            batch_loss_D.backward()
            self.optimizer_D.step()
            
            loss_D += batch_loss_D.item()
            
            
            """ train G """
            # leaf
            z = Variable(torch.randn(batch_size, self.option.z_dim)).to(self.device)
            f_imgs = self.netG(z)

            # dis
            f_logit = self.netD(f_imgs)

            # compute loss
            batch_loss_G = self.criterion(f_logit, r_label)

            # update model
            self.netG.zero_grad()
            batch_loss_G.backward()
            self.optimizer_G.step()
            
            loss_G += batch_loss_G.item()

        loss_G /= len(self.train_loader)
        loss_D /= len(self.train_loader)
        return loss_G, loss_D

    def train_one_batch(self):
        self.netG.train()
        self.netD.train()
        
        x_y = next(iter(self.train_loader))  

        """ Train D """
        # img
        x = x_y.to(self.device)
        batch_size = len(x)
        z = Variable(torch.randn(batch_size, self.option.z_dim)).to(self.device)

        r_imgs = Variable(x)
        f_imgs = self.netG(z)

        # label
        r_label = torch.ones((batch_size)).to(self.device)
        f_label = torch.zeros((batch_size)).to(self.device)

        # D
        r_logit = self.netD(r_imgs)
        f_logit = self.netD(f_imgs)

        # compute loss
        r_loss = self.criterion(r_logit, r_label) 
        # r_logit = [0.2, 0.6, 0.9], r_label = [1, 1, 1] 
        # => [[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]]
        # => [[1, 0], [1, 0], [1, 0]]
        # -1*log(0.2)-0*log(0.8)
        
        f_loss = self.criterion(f_logit, f_label)
        batch_loss_D = (r_loss + f_loss) / 2

        self.optimizer_D.zero_grad()
        batch_loss_D.backward()
        self.optimizer_D.step()


        """ train G """
        # leaf
        z = Variable(torch.randn(batch_size, self.option.z_dim)).to(self.device)
        f_imgs = self.netG(z)

        # dis
        f_logit = self.netD(f_imgs)

        # compute loss
        batch_loss_G = self.criterion(f_logit, r_label)

        # update model
        self.netG.zero_grad()
        batch_loss_G.backward()
        self.optimizer_G.step()
        
        batch_loss_G = batch_loss_G.item()
        batch_loss_D = batch_loss_D.item()
        return batch_loss_G, batch_loss_D
    
    def sample_imgs(self, epoch=None):
        file_name = 'epoch-%s---model-netG.pth'%(epoch)
        path = os.path.join('.', 'weights', file_name)
        
        model = self.netG
        model.load_state_dict(torch.load(path))
        model.eval()
        
        batch_size = 20
        z_sample = Variable(torch.randn(batch_size, self.option.z_dim)).to(self.device)
        imgs_sample = model(z_sample)
        
        imgs = imgs_sample.detach().cpu().numpy() # detach gradient, gpu to cpu, tensor to numpy
        imgs = (imgs + 1) / 2.0 #?
        '''
        -1 ~ 1
        0 ~ 2
        0 ~ 1
        '''
        return imgs
        
    
    def save_model(self, epoch, loss_G, loss_D):
        file_name = 'epoch-%s---model-netG.pth'%(epoch)
        path = os.path.join('.', 'weights', file_name)
        torch.save(self.netG.state_dict(), path)
        
        file_name = 'epoch-%s---model-netD.pth'%(epoch)
        path = os.path.join('.', 'weights', file_name)
        torch.save(self.netD.state_dict(), path)
        
        # history
        row = pd.DataFrame()
        row['epoch'] = [epoch]
        row['loss_G'] = [loss_G]
        row['loss_D'] = [loss_D]
        self.df_history = self.df_history.append(row)
        self.df_history.to_csv(os.path.join('.', 'weights', 'df_history.csv'))



class Option:
    def __init__(self):
        self.batch_size = 64
        self.z_dim = 100
        self.lr = 1e-4
        self.n_epoch = 10

option = Option()

trainAndTest = TrainAndTest(option)

for epoch in range(10000+1):
    loss_G, loss_D = trainAndTest.train_one_batch()
    loss_G, loss_D = round(loss_G, 3), round(loss_D, 3)
    print(epoch, loss_G, loss_D)
    
    if epoch%100==0:
        trainAndTest.save_model(epoch, loss_G, loss_D)

