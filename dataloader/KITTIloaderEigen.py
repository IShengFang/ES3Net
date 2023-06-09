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

def dataloader(txt_file, file_path):
    all_left_img=[]
    all_right_img=[]
    
    with open(txt_file, 'r') as f_txt:
        lines  = f_txt.read().splitlines()
        for line in lines:
            img_L = line.split(' ')[0][:-3]+'png'
            img_R = line.split(' ')[1][:-3]+'png'
            all_left_img.append(file_path+img_L)
            all_right_img.append(file_path+img_R)

    return all_left_img, all_right_img

