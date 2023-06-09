import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from tqdm import tqdm

class Trainer(nn.Module):
    def __init__(self, model, train_dataloader, multi_scale, 
                       only_recon_epoch=15, dual_transform='flip'):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.train_dataloader = train_dataloader
        self.multi_scale = multi_scale
        self.only_recon_epoch = only_recon_epoch
        self.dual_transform =dual_transform
        self.to(self.device)
    def init_optimzer(self, lr=0.0005, betas=(0.9, 0.999)):
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=betas)

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch > 199:
            self.lr = 0.00005
        print('lr:', self.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr


    def warp(self, x, disp):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            vgrid = grid.cuda()
        else:
            vgrid = grid
        vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, align_corners=True)
        return output

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(x, 3, 1, 0)
        mu_y = F.avg_pool2d(y, 3, 1, 0)
        
        #(input, kernel, stride, padding)
        sigma_x  = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
        sigma_y  = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y , 3, 1, 0) - mu_x * mu_y
        
        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        
        SSIM = SSIM_n / SSIM_d
        
        return torch.clamp((1 - SSIM) / 2, 0, 1)


    def gradient(self, pred):
        D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    def compute_grad2_smoothness_loss(self, flo, image, beta):
        """
        Calculate the image-edge-aware second-order smoothness loss
        """
        
        img_grad_x, img_grad_y = self.gradient(image)
        weights_x = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_x), 1, keepdim=True))
        weights_y = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_y), 1, keepdim=True))

        dx, dy = self.gradient(flo)
        dx2, dxdy = self.gradient(dx)
        dydx, dy2 = self.gradient(dy)

        return (torch.mean(beta*weights_x[:,:, :, 1:]*torch.abs(dx2)) + torch.mean(beta*weights_y[:, :, 1:, :]*torch.abs(dy2))) / 2.0

    def reconstruction_loss(self, x, y):
        ssim = torch.mean(self.SSIM(x, y))
        l1 = torch.mean(torch.abs(x - y))
        return 0.85*ssim + 0.15*l1

    def set_loss_weight(self, smth_loss_weight=10, lr_loss_weight=0.01 ):
        self.smth_loss_weight = smth_loss_weight
        self.lr_loss_weight = lr_loss_weight

    def compute_loss(self, imgL, imgR, epoch):

        # estimate left disparity
        if self.multi_scale:
            l_disps = self.model(imgL, imgR)
        else:
            l_disps = self.model(imgL, imgR)

        # estimate right disparity
        if self.dual_transform == 'flip':
            imgL_ast = imgL.flip(3)
            imgR_ast = imgR.flip(3)
        elif self.dual_transform == 'rotate':
            imgL_ast = imgL.flip(2).flip(3)
            imgR_ast = imgR.flip(2).flip(3)
        if self.multi_scale:
            r_disps_ast = self.model(imgR_ast, imgL_ast)
        else:
            r_disps_ast = self.model(imgR_ast, imgL_ast)

        loss = 0
        for l_disp,r_disp_ast in zip(l_disps, r_disps_ast):
            # compute left reconstruction loss
            recon_imgL = self.warp(imgR, l_disp)
            left_recon_loss = self.reconstruction_loss(
                                        recon_imgL[:,:,:, 75:575], imgL[:,:,:, 75:575])

            # compte right reconstruction loss
            recon_imgR_ast = self.warp(imgL_ast, r_disp_ast)
            if self.dual_transform == 'flip':
                recon_imgR = recon_imgR_ast.flip(3)
                r_disp = r_disp_ast.flip(3)
            elif self.dual_transform == 'rotate':
                recon_imgR = recon_imgR_ast.flip(2).flip(3)
                r_disp = r_disp_ast.flip(2).flip(3)
            right_recon_loss = self.reconstruction_loss(
                                            recon_imgR[:,:,:, 0:500], imgR[:,:,:, 0:500])

            loss += left_recon_loss + right_recon_loss

            # compute total loss
            if epoch > self.only_recon_epoch:
                # compute left disparity smoothness loss
                left_smth_loss = self.compute_grad2_smoothness_loss(l_disp/20, imgL, 1.0)
                # compute right disparity smoothness loss
                right_smth_loss = self.compute_grad2_smoothness_loss(r_disp/20, imgR, 1.0)

                # compute left-right consistency loss
                r2l_disp = self.warp(r_disp, l_disp)
                if self.dual_transform == 'flip':
                    l_disp_ast = l_disp.flip(3)
                elif self.dual_transform == 'rotate':
                    l_disp_ast = l_disp.flip(2).flip(3)
                l2r_disp_ast = self.warp(l_disp_ast, r_disp_ast)

                if self.dual_transform == 'flip':
                    l2r_disp = l2r_disp_ast.flip(3)
                elif self.dual_transform == 'rotate':
                    l2r_disp = l2r_disp_ast.flip(2).flip(3)

                lr_right_loss = torch.mean(torch.abs(l_disp[:, :, :, 75:575] - r2l_disp[:, :, :, 75:575]))
                lr_left_loss = torch.mean(torch.abs(r_disp[:, :, :, 0:500] - l2r_disp[:, :, :, 0:500]))
                lr_loss = lr_left_loss + lr_right_loss

                loss += self.smth_loss_weight * (left_smth_loss + right_smth_loss) + self.lr_loss_weight * lr_loss
            
        return loss

    def fit(self, start_epoch ,epochs, cpt_dir, save_freq=1):
        print('start training')
        start_full_time = time.time()
        epoch_pbar = tqdm(range(start_epoch, epochs))
        for epoch in epoch_pbar: 
            print(f'This is {epoch}-th epoch')
            self.model.train()
            total_train_loss = 0
            self.adjust_learning_rate(self.optimizer, epoch)

            ## training ##
            iter_pbar = tqdm(self.train_dataloader)
            for batch_idx, (imgL_crop, imgR_crop) in enumerate(iter_pbar):
                imgL_crop, imgR_crop = imgL_crop.to(self.device), imgR_crop.to(self.device)
                self.optimizer.zero_grad()
                loss = self.compute_loss(imgL_crop, imgR_crop, epoch)
                loss.backward()
                self.optimizer.step()
                loss = loss.item()
                iter_pbar.set_description('E:{}|loss{:.3f}'.format(epoch, loss))
                total_train_loss += loss

            total_train_loss /= len(self.train_dataloader)
            epoch_pbar.set_description('E{}|loss{:.3f}'.format(epoch, total_train_loss))

            ## save checkpoint ##
            if epoch % save_freq == 0:
                os.makedirs(cpt_dir, exist_ok=True)
                total_loss_str = str(total_train_loss).replace('.','_')[0:7]
                cpt_path = os.path.join(cpt_dir, f'checkpoint_{epoch}_{total_loss_str}.cpt')                
                torch.save({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'train_loss': total_train_loss,
                }, cpt_path)
                
                print(f'Checkpoint saved to {cpt_path}')

        print('Done :)')
        print('full training time: %.2f hours' % ((time.time() - start_full_time) / 3600))
