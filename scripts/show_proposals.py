#!/usr/bin/env python
import os
import re
import sys
import argparse
import pickle
import json
import cv2
import numpy as np

#BASEPATH = '~/Downloads/data/grocery'
#EXT_REGEX = '^IMG_.*$'
BASEPATH = '~/Downloads/data/grocery(peter)/images_resize'
EXT_REGEX = '^Wegmans-.*$'
IMG_EXT = '.jpg'
REGEX_FLAGS = re.IGNORECASE
MDL = True

def nms(boxes, threshold):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], \
            np.where(overlap > threshold)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def main(argv):
    base_path = os.path.expanduser(BASEPATH)
    if MDL is True:
        proposal_path = os.path.join(base_path, 'MDL_Proposals')
    else:
        proposal_path = os.path.join(base_path, 'Proposals')

    for filename in os.listdir(proposal_path):
        if re.match(EXT_REGEX, filename, REGEX_FLAGS):
            im = cv2.imread(os.path.join(base_path, filename + IMG_EXT))
            im2 = im
            if MDL is True:
                ifs = open(os.path.join(proposal_path, filename), 'r')
                proposals = json.loads(ifs.read())
                proposals = np.asarray(proposals)
                proposals[:, 2] = proposals[:, 0] + proposals[:, 2]
                proposals[:, 3] = proposals[:, 1] + proposals[:, 3]

                #proposals = nms(proposals, 0.85)
                for idx in range(0, proposals.shape[0]):
                    pt1 = (proposals[idx, 0], proposals[idx, 1])
                    pt2 = (proposals[idx, 2], proposals[idx, 3])
                    im2 = cv2.rectangle(im2, pt1, pt2, (0, 255, 0), 1)

                im3 = im2
            else:
                proposals = pickle.load(open(os.path.join(proposal_path, filename)))
                for idx in range(0, 100):
                    im2 = cv2.rectangle(im2, (proposals[idx][1], proposals[idx][0]), \
                                           (proposals[idx][3], proposals[idx][2]), \
                                           (0, 255, 0), 5)
                im3 = cv2.resize(im2, (im2.shape[1]/5, im2.shape[0]/5))

            cv2.imshow('foo', im3)
            cv2.waitKey(0)
            cv2.imwrite(os.path.join(base_path, filename + '_res_top100.JPG'), im3)

if __name__ == '__main__':
    main(sys.argv)
