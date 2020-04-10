import numpy as np
import os, glob
from skimage.io import imread, imsave
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import flow_utils

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

        
device = torch.device("cuda:3")
torch.cuda.set_device(device)

frames_folder = '/homes/grail/xiaojwan/flownet2_validation_examples'
frames = sorted(glob.glob(os.path.join(frames_folder, '*.pred.flo')))
print(len(frames))
for frame in frames:
    flow_file = frame
    frame_name = frame[:-9]
    image_with_noised_camera_file=frame_name+'_noised_sigma_5.png'
    image_file=frame_name+'_frame.png'
    output_prefix=frame_name

    image_with_noised_camera = imread(image_with_noised_camera_file)
    image=imread(image_file)[...,:3]
    flow=flow_utils.readFlow(flow_file)
    flow_zero_mask = np.all((flow==0), axis=2, keepdims=True)
    flow_zero_mask = np.expand_dims(flow_zero_mask, axis=0)

    image_size = image.shape[:2]
    render_size = flow.shape[:2]
    print(image_size)
    print(render_size)
    cropper = StaticCenterCrop(image_size, render_size)
    image_with_noised_camera = cropper(image_with_noised_camera)
    image = cropper(image)

    
    
    image_with_noised_camera = np.expand_dims(image_with_noised_camera, axis=0)
    image_with_noised_camera = image_with_noised_camera.transpose(0, 3, 1, 2)
    image_with_noised_camera = torch.from_numpy(image_with_noised_camera).float().to(device)

    B, C, H, W = image_with_noised_camera.size() ## B=1
    
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float().to(device)

    flow = np.expand_dims(flow, axis=0)
    flow = flow.transpose(0, 3, 1, 2)
    flow = torch.from_numpy(flow).float().to(device)
    vgrid = Variable(grid)+flow

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(image_with_noised_camera, vgrid)



    output = output.cpu().numpy()
    output = output.transpose(0,2,3,1)

    # for flow value = 0, use original pxiel value
    output = output * (1-flow_zero_mask) + image * flow_zero_mask
    imsave(output_prefix+'.pred.warp.png', output[0])

