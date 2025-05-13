import argparse
import os
import random
import time
from os.path import isfile, join, split

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import yaml
import cv2

from torch.optim import lr_scheduler
from logger import Logger

from dataloader import get_loader
from model.network import Net
from skimage.measure import label, regionprops
from utils import reverse_mapping, visulize_mapping, edge_align, get_boundary_point

parser = argparse.ArgumentParser(description='PyTorch Semantic-Line Training')
# arguments from command line
parser.add_argument('--config', default="./config.yml", help="path to config file")
parser.add_argument('--model', required=True, help='path to the pretrained model')
parser.add_argument('--align', default=False, action='store_true')
parser.add_argument('--tmp', default="", help='tmp')
args = parser.parse_args()

assert os.path.isfile(args.config)

with open(args.config) as f:
    CONFIGS = yaml.load(f, Loader=yaml.FullLoader)

# merge configs
if args.tmp != "" and args.tmp != CONFIGS["MISC"]["TMP"]:
    CONFIGS["MISC"]["TMP"] = args.tmp

os.makedirs(CONFIGS["MISC"]["TMP"], exist_ok=True)
logger = Logger(os.path.join(CONFIGS["MISC"]["TMP"], "log.txt"))

def main():

    logger.info(args)

    model = Net(numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"], backbone=CONFIGS["MODEL"]["BACKBONE"])
    model = model.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])

    if args.model:
        if isfile(args.model):
            logger.info("=> loading pretrained model '{}'".format(args.model))
            checkpoint = torch.load(args.model, weights_only=False)
            if 'state_dict' in checkpoint.keys():
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info("=> loaded checkpoint '{}'"
                  .format(args.model))
        else:
            logger.info("=> no pretrained model found at '{}'".format(args.model))
    # dataloader
    test_loader = get_loader(CONFIGS["DATA"]["TEST_DIR"], CONFIGS["DATA"]["TEST_LABEL_FILE"], 
                                batch_size=1, num_thread=CONFIGS["DATA"]["WORKERS"], test=True)

    logger.info("Data loading done.")

    
    
    logger.info("Start testing.")
    total_time = test(test_loader, model, args)
    
    logger.info("Test done! Total %d imgs at %.4f secs without image io, fps: %.3f" % (len(test_loader), total_time, len(test_loader) / total_time))

def test(test_loader, model, args):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        bar = tqdm.tqdm(test_loader)
        iter_num = len(test_loader.dataset)
        ftime = 0
        ntime = 0
        for i, data in enumerate(bar):
            t = time.time()
            images, names, size_batch = data
            # print(f"Loading image: {names[0]}")

            images = images.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
            key_points = model(images)

            key_points = torch.sigmoid(key_points)
            ftime += (time.time() - t)
            t = time.time()
            visualize_save_path = os.path.join(CONFIGS["MISC"]["TMP"], 'visualize_test')
            os.makedirs(visualize_save_path, exist_ok=True)

            key_points_np = key_points.squeeze().cpu().numpy()

            binary_kmap = key_points_np > CONFIGS['MODEL']['THRESHOLD']
            kmap_label = label(binary_kmap, connectivity=1)

            props = regionprops(kmap_label, intensity_image=key_points_np) 

            detections = []
            for prop in props:
                confidence = prop.mean_intensity 
                centroid = prop.centroid      
                detections.append({'confidence': confidence, 'centroid': centroid})

            detections.sort(key=lambda x: x['confidence'], reverse=True)

            top_detections = detections
            # top_detections = detections[:2]

            top_plist = [d['centroid'] for d in top_detections]

            b_points = [] 
            scaled_b_points = [] 

            # size_tuple = (size[0][0].item(), size[0][1].item()) 
            size_tuple = size_batch[0]
            scale_w = size_tuple[1] / 400.0
            scale_h = size_tuple[0] / 400.0

            if top_plist: # Only proceed if we found at least one point
                b_points = reverse_mapping(top_plist, numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"], size=(400, 400))

                scale_w = size_tuple[1] / 400.0
                scale_h = size_tuple[0] / 400.0

                for i in range(len(b_points)):
                    y1 = int(np.round(b_points[i][0] * scale_h))
                    x1 = int(np.round(b_points[i][1] * scale_w))
                    y2 = int(np.round(b_points[i][2] * scale_h))
                    x2 = int(np.round(b_points[i][3] * scale_w))

                    y1 = max(0, min(y1, size_tuple[0] - 1))
                    x1 = max(0, min(x1, size_tuple[1] - 1))
                    y2 = max(0, min(y2, size_tuple[0] - 1))
                    x2 = max(0, min(x2, size_tuple[1] - 1))

                    if x1 == x2 and y1 == y2: 
                        continue

                    if x1 == x2:
                        angle = -np.pi / 2
                    else:
                        angle = np.arctan(float(y1-y2) / float(x1-x2)) 

                    (x1_b, y1_b), (x2_b, y2_b) = get_boundary_point(y1, x1, angle, size_tuple[0], size_tuple[1])
                    scaled_b_points.append((y1_b, x1_b, y2_b, x2_b)) 

            image_filename = names[0]
            base_image_dir = CONFIGS["DATA"]["TEST_DIR"]
            full_image_path = os.path.join(base_image_dir, image_filename)

            # print(f"DEBUG: Attempting to load image for visualization: '{full_image_path}'")
            # print(f"DEBUG: File exists? {os.path.isfile(full_image_path)}")

            vis = visulize_mapping(scaled_b_points, size_tuple[::-1], image_filename) # size_tuple[::-1] is (W, H)

            base_filename = os.path.basename(image_filename)
            output_basename = os.path.splitext(base_filename)[0]

            cv2.imwrite(join(visualize_save_path, base_filename), vis)

            np_data = np.array(scaled_b_points)
            np.save(join(visualize_save_path, output_basename), np_data)


            # --- Optional: Handle Edge Alignment for Top Lines ---
            if CONFIGS["MODEL"]["EDGE_ALIGN"] and args.align:
                b_points_aligned = []
                if scaled_b_points: 
                    for i in range(len(scaled_b_points)):
                         aligned_point = edge_align(scaled_b_points[i], full_image_path, size_tuple, division=5)
                         b_points_aligned.append(aligned_point)

                    vis_aligned = visulize_mapping(b_points_aligned, size_tuple[::-1], full_image_path)
                    cv2.imwrite(join(visualize_save_path, output_basename + '_align.png'), vis_aligned)
                    np_data_aligned = np.array(b_points_aligned)
                    np.save(join(visualize_save_path, output_basename + '_align'), np_data_aligned)
                else:
                    logger.warning(f"No points to align for {image_filename}")

            ntime += (time.time() - t)

            # heatmap_save_path = os.path.join(CONFIGS["MISC"]["TMP"], 'visualize_heatmaps')
            # os.makedirs(heatmap_save_path, exist_ok=True)

            # key_points_np = key_points.squeeze().cpu().numpy()

            # image_filename = names[0] # names is ('filename.png',)
            # base_filename = os.path.basename(image_filename)
            # output_basename_no_ext = os.path.splitext(base_filename)[0]
            # heatmap_filename = output_basename_no_ext + "_heatmap.png"
            # full_heatmap_path = os.path.join(heatmap_save_path, heatmap_filename)

            # # Plot and save the heatmap
            # plt.figure() # Create a new figure
            # plt.imshow(key_points_np, cmap='viridis', interpolation='nearest') # Use a colormap like viridis
            # plt.title(f'Heatmap for {base_filename}')
            # plt.colorbar() # Add a colorbar to show intensity scale
            # plt.savefig(full_heatmap_path)
            # plt.close() # Close the figure to free memory


            visualize_save_path = os.path.join(CONFIGS["MISC"]["TMP"], 'visualize_test')
            os.makedirs(visualize_save_path, exist_ok=True)

            binary_kmap = key_points_np > CONFIGS['MODEL']['THRESHOLD']

    print('forward time for total images: %.6f' % ftime)
    print('post-processing time for total images: %.6f' % ntime)
    return ftime + ntime
        
# def test(test_loader, model, args):
#     # switch to evaluate mode
#     model.eval()
#     with torch.no_grad():
#         bar = tqdm.tqdm(test_loader)
#         iter_num = len(test_loader.dataset)
#         ftime = 0
#         ntime = 0
#         for i, data in enumerate(bar):
#             t = time.time()
#             images, names, size = data
            
#             images = images.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
#             # size = (size[0].item(), size[1].item())       
#             key_points = model(images)
            
#             key_points = torch.sigmoid(key_points)
#             ftime += (time.time() - t)
#             t = time.time()
#             visualize_save_path = os.path.join(CONFIGS["MISC"]["TMP"], 'visualize_test')
#             os.makedirs(visualize_save_path, exist_ok=True)

#             binary_kmap = key_points.squeeze().cpu().numpy() > CONFIGS['MODEL']['THRESHOLD']
#             kmap_label = label(binary_kmap, connectivity=1)
#             props = regionprops(kmap_label)
#             plist = []
#             for prop in props:
#                 plist.append(prop.centroid)

#             size = (size[0][0], size[0][1])
#             b_points = reverse_mapping(plist, numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"], size=(400, 400))
#             scale_w = size[1] / 400
#             scale_h = size[0] / 400
#             for i in range(len(b_points)):
#                 y1 = int(np.round(b_points[i][0] * scale_h))
#                 x1 = int(np.round(b_points[i][1] * scale_w))
#                 y2 = int(np.round(b_points[i][2] * scale_h))
#                 x2 = int(np.round(b_points[i][3] * scale_w))
#                 if x1 == x2:
#                     angle = -np.pi / 2
#                 else:
#                     angle = np.arctan((y1-y2) / (x1-x2))
#                 (x1, y1), (x2, y2) = get_boundary_point(y1, x1, angle, size[0], size[1])
#                 b_points[i] = (y1, x1, y2, x2)

#             vis = visulize_mapping(b_points, size[::-1], names[0])

            

#             cv2.imwrite(join(visualize_save_path, names[0].split('/')[-1]), vis)
#             np_data = np.array(b_points)
#             np.save(join(visualize_save_path, names[0].split('/')[-1].split('.')[0]), np_data)

#             if CONFIGS["MODEL"]["EDGE_ALIGN"] and args.align:
#                 for i in range(len(b_points)):
#                     b_points[i] = edge_align(b_points[i], names[0], size, division=5)
#                 vis = visulize_mapping(b_points, size, names[0])
#                 cv2.imwrite(join(visualize_save_path, names[0].split('/')[-1].split('.')[0]+'_align.png'), vis)
#                 np_data = np.array(b_points)
#                 np.save(join(visualize_save_path, names[0].split('/')[-1].split('.')[0]+'_align'), np_data)
#             ntime += (time.time() - t)
#     print('forward time for total images: %.6f' % ftime)
#     print('post-processing time for total images: %.6f' % ntime)
#     return ftime + ntime

if __name__ == '__main__':
    main()
