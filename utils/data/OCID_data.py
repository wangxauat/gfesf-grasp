import os
import glob

import cv2
import numpy as np

from utils.data.grasp_data import GraspDatasetBase
from utils.dataset_processing import grasp, image

# # Step 1: Open and read the txt file
# with open('grasping/code/FG-sr-edge-self/utils/data/training_0.txt', 'r') as file:
#     lines = file.readlines()

# # Step 2: Strip newline characters and store the paths in a list
# image_paths = [line.strip() for line in lines]
# print(image_paths)
# -------------------设置固定随机种子---------------------#
# def setup_seed(seed):
#     np.random.seed(seed)


# # 设置随机数种子
# setup_seed(54)


# ------------------------------------------------------#

class OCIDDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Jacquard dataset.
    """

    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, **kwargs):
        """
        :param file_path: Jacquard Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(OCIDDataset, self).__init__(**kwargs)

        # Step 1: Open and read the txt file
        self.file_path = os.path.join(file_path,'ocid.txt')
        with open(self.file_path, 'r') as file:
            lines = file.readlines()

        # Step 2: Strip newline characters and store the paths in a list
        RGB_paths = [line.strip() for line in lines]
        RGB_paths.sort()
        l = len(RGB_paths)
        self.length = l
        print(l)
        RGB_paths = [(file_path + f) for f in RGB_paths]
        rgb_paths = [f.replace(',', '/rgb/') for f in RGB_paths]
        depth_paths = [f.replace('rgb', 'depth') for f in rgb_paths]
        grasp_paths = [f.replace('depth', 'Annotations') for f in depth_paths]
        grasp_paths = [f.replace('.png', '.txt') for f in grasp_paths]
        # graspf = glob.glob(os.path.join(file_path,'*','*', '*.txt'))
        # graspf.sort()
        # l = len(graspf)
        # self.length = l
        # print(l)
        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        # if ds_rotate:
        #     graspf = graspf[int(l * ds_rotate):] + graspf[:int(l * ds_rotate)]

        # seq_path, im_name = RGB_paths.split(',')
        # sample_path = os.path.join(file_path, seq_path)
        # RGB_path = os.path.join(sample_path, 'rgb', im_name)
        # depth_path = os.path.join(sample_path , 'depth', im_name)
        # grasp_path = os.path.join(sample_path, 'Annotations', im_name[:-4] + '.txt')
        # depthf = [f.replace('grasps.txt', 'perfect_depth.tiff') for f in graspf]
        # rgbf = [f.replace('perfect_depth.tiff', 'RGB.png') for f in depthf]
        self.grasp_files = grasp_paths[int(l * start):int(l * end)]
        self.depth_files = depth_paths[int(l * start):int(l * end)]
        self.rgb_files = rgb_paths[int(l * start):int(l * end)]

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top
    
    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.Image.from_file(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        # cv2.imshow("img", depth_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        # cv2.imshow("img", rgb_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_rgb_b(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        rgb_img = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_BGR2GRAY)
        rgb_img_b = cv2.Canny(rgb_img, 80, 150, (3, 3))
        # cv2.imshow("img", rgb_img_b)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # rgb_img_b =cv2.cvtColor(numpy.asarray(rgb_img_b),cv2.COLOR_BGR2RGB)
        # if normalise:
        #     rgb_img_b.normalise()
        #     rgb_img_b.img = rgb_img_b.img.transpose((2, 0, 1))
        return rgb_img_b


    # def get_jname(self, idx):
    #     return '_'.join(self.grasp_files[idx].split(os.sep)[-1].split('_')[:-1])
