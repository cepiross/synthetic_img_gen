#!/usr/bin/env python
"""
Version History
    v0.3 exception handling (just an item, or absence in object)
    v0.2 add categories to annotation (label migration)
    v0.1 initial
"""
import os
import re
import sys
import argparse
import json
from datetime import datetime
import random
from pycocotools import mask

EXT_REGEX = '.*.json$'
REGEX_FLAGS = re.IGNORECASE
RATIO_HOLDUP = 1.0/9
TYPE = ['train', 'val']

def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments: input file.
    parser.add_argument(
        "--basepath",
        help="Input data base path."
    )

    args = parser.parse_args()
    if not args.basepath:
        print("No Base Path!")
        return

    base_path = os.path.expanduser(args.basepath)
    fout_train = open(os.path.join(base_path, '../train.json'), 'w')
    if fout_train is None:
        print("Error in opiening train.json")
    fout_val = open(os.path.join(base_path, '../val.json'), 'w')
    if fout_val is None:
        print("Error in opiening val.json")

    # reset random generator
    now = datetime.now()
    random.seed(now)

    # coco xml format
    # info (obj)
    #   description, version, year, contributor, date_created
    jsoninfo = dict()
    jsoninfo['description'] = 'This is initial version of grocery item dataset ' \
                                'generated by unreal engine.'
    jsoninfo['version'] = '0.3'
    jsoninfo['contributor'] = 'Jinhang Choi and PennState MDL group'
    jsoninfo['date_created'] = str(now)

    # coco xml format
    # images (array)
    #   file_name, height, width, id
    jsonimgs = []

    # coco xml format
    # licenses (array)

    # coco xml format
    # annotations (array)
    #   segmentation, area, iscrowd, image_id, bbox[x,y,w,h], category_id, id
    jsonantns_train = []
    objstrain_cnt = 0
    jsonantns_val = []
    objsval_cnt = 0

    # coco xml format
    # categories (array)
    #   supercategory, id, name
    jsoncategs = []

    labels = []
    base_objid = 0
    for fid, filename in enumerate(os.listdir(base_path)):
        if re.match(EXT_REGEX, filename, REGEX_FLAGS):
            ifs = open(os.path.join(base_path, filename), 'r')

            # unreal xml format
            # annotation
            #   filename
            #   imagesize
            #       nrows / ncols
            #   object (array)
            #       name
            #       polygon
            #           pt (array)
            #               x / y
            currjson = json.loads(ifs.read())['annotation']
            jsonimg = dict()
            jsonimg['file_name'] = currjson['filename']
            jsonimg['height'] = currjson['imagesize']['nrows']
            jsonimg['width'] = currjson['imagesize']['ncols']
            jsonimg['id'] = fid
            jsonimgs.append(jsonimg)

            print json.dumps(jsonimg)

            objlist = []
            if 'object' in currjson:
                if isinstance(currjson['object'], list):
                    for obj in currjson['object']:
                        objlist.append(obj)
                else:
                    objlist.append(currjson['object'])
            else:
                print ('object absence in %s' % filename)
                continue

            for objid, obj in enumerate(objlist):
                categname = obj['name'].split()
                categid = -1
                labelid = -1
                for itemid, item in enumerate(labels):
                    if item == categname:
                        labelid = itemid
                        break

                for item in jsoncategs:
                    if item['supercategory'] == categname[0] and item['name'] == categname[1]:
                        categid = item['id']
                        break

                if labelid == -1:
                    labels.append(categname)
                    print (categname, len(labels))

                if categid == -1:
                    jsoncateg = dict()
                    jsoncateg['supercategory'] = categname[0]
                    jsoncateg['name'] = categname[1]
                    categid = len(jsoncategs)
                    jsoncateg['id'] = categid
                    jsoncategs.append(jsoncateg)
                    print categname[0], categname[1], categid

                if 'polygon' not in obj:
                    continue

                segmentation = []
                for point in obj['polygon']['pt']:
                    ptx = float(point['x'])
                    pty = float(point['y'])
                    segmentation.append(ptx)
                    segmentation.append(pty)

                jsonantn = dict()
                jsonantn['segmentation'] = [segmentation]
                rle = mask.frPyObjects([segmentation], \
                                        int(jsonimg['height']), \
                                        int(jsonimg['width']))
                jsonantn['area'] = float(mask.area(rle)[0])
                jsonantn['iscrowd'] = 0
                jsonantn['image_id'] = fid
                jsonantn['bbox'] = mask.toBbox(rle)[0].tolist()
                jsonantn['category_id'] = categid
                jsonantn['id'] = base_objid + objid
                rand = random.random()
                if rand > RATIO_HOLDUP:
                    jsonantns_train.append(jsonantn)
                    objstrain_cnt += 1
                else:
                    jsonantns_val.append(jsonantn)
                    objsval_cnt += 1

            base_objid = base_objid + len(objlist)

    jsonoutput = dict()
    jsonoutput['info'] = jsoninfo
    jsonoutput['images'] = jsonimgs
    jsonoutput['categories'] = jsoncategs

    jsonoutput['annotations'] = jsonantns_train
    total_cnt = objstrain_cnt + objsval_cnt
    print ('writing train.json (%d/%d objects)..' % (objstrain_cnt, total_cnt))
    json.dump(jsonoutput, fout_train)

    jsonoutput['annotations'] = jsonantns_val
    print ('writing val.json (%d/%d objects)..' % (objsval_cnt, total_cnt))
    json.dump(jsonoutput, fout_val)

    fout_train.close()
    fout_val.close()

if __name__ == '__main__':
    main(sys.argv)