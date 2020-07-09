# adapted from http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html

import cv2
import numpy as np

from model_bisenet import BiSeNet
import torch
from torchvision import transforms
from PIL import Image

def bisenet_transform():
    return transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -1 ~ +1 # Newly Added
])

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        print(save_path[:-4])
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels = 6):
    # assume mask is float32 [0,1]

    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    return ls_

if __name__ == '__main__':

    A = cv2.imread("./img_align/37.png")
    B = cv2.imread("./img_align/2.png")

    # Mask
    if True:
        # BiseNet
        bisenet = BiSeNet(n_classes=19).cuda()
        bisenet.load_state_dict(torch.load('./checkpoint/79999_iter.pth'))
        bisenet.eval()

        m_target = Image.open("./img_align/37.png")
        m_target2 = bisenet_transform()(m_target).unsqueeze(0).cuda()
        m_parse = bisenet(m_target2)[0]
        m_lb = torch.argmax(m_parse,dim=1)

        # Mask Change
        # 0:Back 1:Face 2,3:Eyebrow 7,8:Ear 10:Nose 11,12,13:Mouth 14:Neck 17:Hair
        m_lb[m_lb == ((11 or 12 or 13))] = 255
        m_lb[m_lb != (255)] = 0
        m_lb/=255 # Make it 0/1 Mask
        m = m_lb.repeat(3,1,1).permute(1,2,0).cpu().numpy().astype('float32')

        # Mask Dilation(Expand) or Erosion(Diminish)
        kernel = np.ones((5,5),np.uint8)
        m = cv2.dilate(m,kernel,iterations=10)
        #m = cv2.erode(m,kernel,iterations = 15)
        m = cv2.blur(m, (50, 50))

        # Visualize Mask
        m_parse2 = m_parse.squeeze(0).detach().cpu().numpy().argmax(0)
        vis_parsing_maps(m_target, m_parse2, stride=1, save_im=True,save_path=f'lpb_parse.png')
    else:
        m = np.zeros([1024,1024,3], dtype='float32')
        half = 512
        m[:,half:] = 1 # make the mask half-and-half


    lpb = Laplacian_Pyramid_Blending_with_mask(A, B, m, 10)
    cv2.imwrite("lpb2.png",lpb)
