import cv2
import argparse
from openpose import Openpose, draw_person_pose
import matplotlib.pyplot as plt
import os.path
import numpy as np
import pickle
from entity import JointType
import matplotlib.cm as cm
from matplotlib.colors import Normalize

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

def calc_roi_pos(poses,roi,shape,padding=5,only=True): #if only is True, only calc the person who has the most detected joints
    eps = 1
    h, w = shape
    if len(poses) > 1 and only:
        only_ind = np.argmax([len(np.where(pose[:,-1]==2)[0]) \
            for pose in poses]) #poses[only_ind] is the sole one
        poses = poses[[only_ind]]
    all_parts_pos = {}
    for i,pose in enumerate(poses):
        t = {}
        for part_name, part_joints_inds in roi.items():
            part_joints = pose[part_joints_inds]
            part_joints = part_joints[np.where(part_joints[:,-1]==2)[0]][:,:2] \
                .round().astype(int)
            if part_joints.size > 0:
                maxx,maxy = np.max(part_joints,axis=0)
                minx,miny = np.min(part_joints,axis=0)
                if not(np.abs(maxx-minx)<=eps or np.abs(maxy-miny)<=eps):
                    t[part_name] = [max(minx-padding,0),max(miny-padding,0), \
                        min(maxx+padding,w),min(maxy+padding,h)] #[bbox.left,bbox.top,bbox.right,bbox.bottom]
        all_parts_pos[i] = t
    return all_parts_pos

def draw_person_parts(img,poses,roi,padding=5,only=True, \
                      colors=None,text=False,aligncolors=True): #padding膨胀，aligncolors使不同图像相同部位以同种颜色标注
    all_parts_pos = calc_roi_pos(poses,roi,img.shape[:2],padding,only) #e.g. {0:{'head':[x1,y1,x2,y2],'upper':[x1,y1,x2,y2],'lower':[x1,y1,x2,y2]},1:{...},...}
    canvas = img.copy()
    if colors is None:
        if aligncolors:
            L = len(roi)*len(all_parts_pos) #当only为False时，应乘以一个固定值（图片中行人的最大数量）
                #，aligncolors才会正确起作用，否则很可能颜色还是无法对齐，此时建议自行提供colors参数
        else:
            L = sum([len(_) for _ in all_parts_pos.values()])
        colors = cm.gist_rainbow(Normalize()(range(L)))*255
    i=0
    if aligncolors:
        part_index = dict(zip(*list(zip(*enumerate(roi.keys())))[::-1]))
    for parts_pos in all_parts_pos.values():
        if parts_pos:
            for part_name, part_pos in parts_pos.items():
                if aligncolors:
                    color = colors[i+part_index[part_name]]
                else:
                    color=colors[i]
                cv2.rectangle(canvas, (part_pos[0], part_pos[1]), \
                    (part_pos[2], part_pos[3]), color, 1)
                if text:
                    cv2.putText(canvas,part_name,(part_pos[0],part_pos[1]+6),\
                        cv2.FONT_HERSHEY_COMPLEX,0.3,color,1)
                if not aligncolors:
                    i+=1
        if aligncolors:
            i+=len(roi)
    return canvas, all_parts_pos

if __name__ == '__main__':
    roi = {
        'head': [JointType.Nose,JointType.Neck,JointType.RightShoulder, \
            JointType.LeftShoulder,JointType.RightEye,JointType.LeftEye, \
            JointType.RightEar,JointType.LeftEar], 
        'upper': [JointType.RightShoulder,JointType.RightElbow, \
            JointType.RightHand,JointType.LeftShoulder,JointType.LeftElbow, \
            JointType.LeftHand,JointType.RightWaist,JointType.LeftWaist], 
        'lower': [JointType.RightWaist,JointType.RightKnee,JointType.RightFoot, \
            JointType.LeftWaist,JointType.LeftKnee,JointType.LeftFoot]
    }
    openpose = Openpose(weights_file = os.path.join(os.path.dirname( \
        __file__),'./models/posenet.pth'), training = False)

    imgs_dir = os.path.join(os.path.dirname(__file__),'./data/reid')
    imgs_list = [i for i in sorted(os.listdir(imgs_dir)) if i.endswith(('.jpg','.png','.bmp'))]

    pose_imgs = []
    parts_pos_data = {}
    for img_name in imgs_list:
        img_path = os.path.normpath(os.path.join(imgs_dir, img_name))
        img = cv2.imread(img_path)
        poses, _ = openpose.detect(img, precise=True)
        img = draw_person_pose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), poses)
        img, parts_pos = draw_person_parts(img, poses, roi, \
            colors=[(0,0,255),(255,0,0),(0,255,0)], text=True)
        pose_imgs.append(img)
        parts_pos_data[img_path] = parts_pos

    with open(os.path.join(os.path.dirname(__file__),'./data/reid_poses.pickle'), 'wb') as f:
        pickle.dump(parts_pos_data, f, pickle.HIGHEST_PROTOCOL)
    contrast_plot(pose_imgs, save_path='reid_pose_imgs.png',dpi=1600)
