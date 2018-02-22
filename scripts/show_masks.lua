require 'image'

local cjson = require 'cjson'
local coco = require 'coco'
local maskApi = coco.MaskApi

local base_path = 'data/unreal/view_front/'
fh, err = io.open(base_path .. 'annotations/train.json')

str = fh:read()
json = cjson.decode(str)
imgs = json['images']
annotations = json['annotations']

print(json['info'])
local img = nil
local imgid = -1
local h,w
for i, obj in ipairs(annotations) do
    if imgid ~= obj['image_id'] then
        if img ~= nil then
            rs = maskApi.frPoly(segmentation, h, w)
            print (imgs[imgid+1]['file_name'])
            masks = maskApi.decode(rs)
            maskApi.drawMasks(img, masks)
            image.save(base_path .. imgs[imgid+1]['file_name'], img)
            img = nil
        end
        imgid = obj['image_id']
        img = image.load(base_path .. 'images/' .. imgs[imgid+1]['file_name'])
        h, w = img:size(2), img:size(3)
        segmentation = {}
        idx = 1
    end
    segmentation[idx] = torch.reshape(torch.DoubleTensor{obj['segmentation']}, #obj['segmentation'][1], 1)
    idx = idx + 1
end

if img ~= nil then
    image.save(base_path .. imgs[imgid+1]['file_name'], img)
    img = nil
end