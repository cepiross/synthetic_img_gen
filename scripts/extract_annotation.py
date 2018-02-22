"""
Image Annotation Extractor using UnrealCV (11/07/2017)

Author: jpc5731@cse.psu.edu (Jinhang Choi),
        jdh5706@cse.psu.edu (Justin Hardin)

Example:
    from unrealcv import client
    client.connect()
    if not client.isconnected(): # Check if the connection is successfully established
        print 'UnrealCV server is not running. Run the game from http://unrealcv.github.io first.'
    else:
        filename = client.request('vget /camera/0/object_mask')
        print filename
"""
from unrealcv import client
from skimage import io
from skimage.measure import find_contours
import os
import numpy as np
import re
import xml.etree.ElementTree as ET


def main():
    client.connect() # Connect to the game
    if not client.isconnected(): # Check if the connection is successfully established
        print 'UnrealCV server is not running. Run the game from http://unrealcv.github.io first.'
        return
    label = 'unrealPic_detail_'
    #label = 'unrealPic_front_'
    #label = 'unrealPic_oblique_'
    iterations = 1300
    displamentVal = 2

    camLocations = [#'-955.886', '339', '286.650',
    #'-1235.886', '339', '286.650',
    # detailed view
    #['-900', '-200', '320'],
    #['-900', '-200', '250'],
    #['-900', '-200', '180'],
    #['-900', '-200', '90'],
    #['-900', '-200', '10']
    # front view
    #['-1100', '-200', '280'],
    #['-1100', '-200', '140']
    #['-1160', '-455', '280'] # side view test
    # oblique view
    #['-1300', '-200', '200']

    # test view
    ['-1000', '-200', '280'],
    ['-1000', '-200', '140']
    #'-1035.886', '339', '286.650']#,
    #'-1135.886', '339', '286.650',
    #'-955.886', '339', '386.650',
    #'-1135.886', '339', '16.650']
    ]

    # front view
    # rotation : 0 0 0
    #            340.871 80.879 360.000
    camRotations = ['0 0 0']#, '353.586 10.463 360.000', '353.586 -10.463 360.000', '340.586 0.463 360.000','340.586 0.463 10.000', '353.586 10.463 -10.000']
    # Get a list of all objects in the scene
    ret = client.request('vget /objects')
    if ret is not None:
        scene_objects = ret.split(' ')
        print 'There are %d objects in this scene' % len(scene_objects)
        color_mapping = get_color_mapping(client, scene_objects)
        startFileNum = 0
        for rot in range(len(camRotations)):
            client.request('vset /camera/0/rotation ' + camRotations[rot])
            for loc in range(len(camLocations)):
                loc_x = float(camLocations[loc][0])
                loc_y = float(camLocations[loc][1])
                loc_z = float(camLocations[loc][2])
                for i in range(iterations):
                    client.request('vset /camera/0/location ' + str(loc_x)  + ' ' + str(loc_y) + ' ' + str(loc_z))
                    filename = client.request('vget /camera/0/object_mask')
                    img = io.imread(filename)
                    print img.shape
                
                    anno = ET.Element('annotation')
                    fileN = ET.SubElement(anno, 'filename')
                    FileId = startFileNum + (i + iterations*(rot*len(camLocations)+loc))
                    fileN.text = 'ScreenShot'+ '{0:05d}'.format(FileId) + '.png'
                    # fileN = ET.SubElement(anno, 'depth_filename')
                    # fileN.text = 'ScreenShot'+ '{0:05d}'.format(startFileNum + (loc + iterations*rot) * 2 + 1) + '.png'
                    folder = ET.SubElement(anno, 'folder')
                    source = ET.SubElement(anno, 'source')
                    source = ET.SubElement(source, 'submittedBy')
                    source.text = 'Unreal Engine'
                    imagesize = ET.SubElement(anno, 'imagesize')
                    nrows = ET.SubElement(imagesize, 'nrows')
                    nrows.text = str(img.shape[0])
                    ncols = ET.SubElement(imagesize, 'ncols')
                    ncols.text = str(img.shape[1])

                    for scene_obj in range(len(scene_objects)):
                        color = color_mapping[scene_objects[scene_obj]]
                        poly = getPoly(img, color.R, color.G, color.B)
                        nameParts = scene_objects[scene_obj].split("_")
                        if poly != [] and len(poly[0]) > 100 and \
                            ('chips' in scene_objects[scene_obj] or 'pop' in scene_objects[scene_obj]) and \
                            'shelf_chips' not in scene_objects[scene_obj]:#nameParts[0] == 'chips':
                                obj = ET.SubElement(anno, 'object')
                                name = ET.SubElement(obj, 'name')
                                if nameParts[0] != 'shelf':
                                    name.text = nameParts[0]
                                else:
                                    name.text = ''
                                if len(nameParts) > 1:
                                    for k in range(1, len(nameParts)-1):
                                        name.text += ' ' + nameParts[k]
                                deletion = ET.SubElement(obj, 'deleted')
                                verification = ET.SubElement(obj, 'verified')
                                occlusion = ET.SubElement(obj, 'occluded')
                                reverse = ET.SubElement(obj, 'reversed')
                                if 'BACK' in scene_objects[scene_obj]:
                                    reverse.text = 'true'
                                else:
                                    reverse.text = 'false'
                                parts = ET.SubElement(obj, 'parts')
                                polygon = ET.SubElement(obj, 'polygon')
                                username = ET.SubElement(polygon, 'username')
                                username.text = 'Unreal Engine'
                                fullPoly = np.concatenate((poly[0:]))
                                for ply in range(len(fullPoly)):
                                    pt = ET.SubElement(polygon, 'pt')
                                    ptx = ET.SubElement(pt, 'x')
                                    ptx.text = str(fullPoly[ply, 1])
                                    pty = ET.SubElement(pt, 'y')
                                    pty.text = str(fullPoly[ply, 0])
                    loc_y += displamentVal
                    f = open(label + '{0:05d}'.format(i + iterations*(rot*len(camLocations)+loc)) + '.xml', 'w')
                    f.write(ET.tostring(anno))
                    f.close()
                    # ScreenShot Location from vrun shot
                    # ${UnrealCV Project}/Saved/Screenshots/Linux/ScreenShot*.png
                    client.request('vset /viewmode lit')
                    print FileId, client.request('vrun shot')
                    # client.request('vset /viewmode depth')
                    # print client.request('vrun shot')
        return

def getPoly(img, r, g, b):
	img2 = np.copy(img)
	mask = (img2[:, :, 0] != r) + (img2[:, :, 1] != g) + (img2[:, :, 2] != b)
	mask2 = (img2[:, :, 0] == r) & (img2[:, :, 1] == g) & (img2[:, :, 2] == b)
	img2[mask] = [0, 0, 0, 255]
	img2[mask2] = [255, 255, 255, 255]
	img2 = np.lib.pad(img2[:, :, 0], (1,1), 'constant')
	return find_contours(img2, 1)

class Color(object):
	''' A utility class to parse color value '''
	regexp = re.compile('\(R=(.*),G=(.*),B=(.*),A=(.*)\)')
	def __init__(self, color_str):
		self.color_str = color_str
		match = self.regexp.match(color_str)
		(self.R, self.G, self.B, self.A) = [int(match.group(i)) for i in range(1,5)]

	def __repr__(self):
		return self.color_str

def get_color_mapping(client, object_list):
	''' Get the color mapping for specified objects '''
	color_mapping = {}
	for objname in object_list:
		color_mapping[objname] = Color(client.request('vget /object/%s/color' % objname))
	return color_mapping

main()
