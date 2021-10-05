import cv2
import argparse
from openpose import Openpose, draw_person_pose
import matplotlib.pyplot as plt
import os.path
import numpy as np

def contrast_plot(imgs,labels=None,process=None, \
                  save_path=None,dpi=None):
    n=len(imgs)
    h=int(np.sqrt(n))
    w=int(np.ceil(n/h))
    for i,img in enumerate(imgs,1):
        plt.subplot(h,w,i)
        if process is not None:
            img=process(img)
        if len(img.shape)==2:
            plt.imshow(img,cmap='gray')
        else:
            plt.imshow(img[:,:,::-1])
        if labels is not None:
            plt.xlabel(labels[i-1])
        plt.xticks([])
        plt.yticks([])
    if save_path is not None:
        if not os.path.isabs(save_path):
            parent_path=os.path.join( \
                os.path.dirname(os.path.realpath(__file__)),'./data/')
            save_path=os.path.normpath(os.path.join(parent_path,save_path))
        plt.savefig(save_path,dpi=dpi)
        print('(contrast_plot)Save img into %s'%save_path)
    plt.show()

if __name__ == '__main__':
    openpose = Openpose(weights_file = os.path.join(os.path.dirname( \
        __file__),'./models/posenet.pth'), training = False)

    imgs_dir = os.path.join(os.path.dirname(__file__),'./data/reid')
    imgs_list = [i for i in sorted(os.listdir(imgs_dir)) if i.endswith(('.jpg','.png','.bmp'))]

    pose_imgs = []
    for img_name in imgs_list:
        img_path = os.path.join(imgs_dir, img_name)
        img = cv2.imread(img_path)
        poses, _ = openpose.detect(img, precise=True)
        pose_img = draw_person_pose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), poses)
        pose_imgs.append(pose_img)
    contrast_plot(pose_imgs, save_path='reid_pose_imgs.png',dpi=1600)
    