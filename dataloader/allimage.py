import torch.utils.data as data
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath, mode='train'):

  all_left_img=[]
  all_right_img=[]

  kitti_dir = filepath + '/train/'
  kitti = os.listdir(kitti_dir)

  for ff in kitti:
    imm_l = os.listdir(kitti_dir+ff+'/RGB_left/')
    for im in imm_l:
      if is_image_file(kitti_dir+ff+'/RGB_left/'+im):
        all_left_img.append(kitti_dir+ff+'/RGB_left/'+im)

    imm_r = os.listdir(kitti_dir+ff+'/RGB_right/')
    for im in imm_r:
      if is_image_file(kitti_dir+ff+'/RGB_right/'+im):
        all_right_img.append(kitti_dir+ff+'/RGB_right/'+im)


  if mode == 'train':
    return all_left_img, all_right_img
  if mode != 'train':
    test_left_img=[]
    test_right_img=[]
    disp_img=[]

    kitti_dir = filepath+'/test/'
    kitti = os.listdir(kitti_dir)

    for ff in kitti:
      imm_l = os.listdir(kitti_dir+ff+'/RGB_left/')
      for im in imm_l:
        if is_image_file(kitti_dir+ff+'/RGB_left/'+im):
          test_left_img.append(kitti_dir+ff+'/RGB_left/'+im)

      imm_r = os.listdir(kitti_dir+ff+'/RGB_right/')
      for im in imm_r:
        if is_image_file(kitti_dir+ff+'/RGB_right/'+im):
          test_right_img.append(kitti_dir+ff+'/RGB_right/'+im)

      imm_d = os.listdir(kitti_dir+ff+'/disp_occ_0/')
      for im in imm_d:
        if is_image_file(kitti_dir+ff+'/disp_occ_0/'+im):
          disp_img.append(kitti_dir+ff+'/disp_occ_0/'+im)

    return all_left_img, all_right_img, test_left_img, test_right_img, disp_img


