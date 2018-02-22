require 'image'
require 'lfs'

local cjson = require 'cjson'
local coco = require 'coco'
local maskApi = coco.MaskApi

local base_path = 'data/unreal/view_side/'
local targets = {'train', 'val'}
local imfile_prefix = 'side_'
local bbox_shapes = {
    -- chips
    {
        -- polygon area should be at least larger than 65% of bbox area
        area_th=0.65,
        -- assuming that bbox is 90 x 90
        --area_abs=8100,
        -- assuming that bbox is 60 x 60
        area_abs=3600,
        -- maximum side ratio = 2.5:1
        side_th=2.5
    },
    -- pop
    {
        -- polygon area should be at least larger than 65% of bbox area
        area_th=0.65,
        -- assuming that bbox is 30 x 30
        --area_abs=900,
        -- assuming that bbox is 40 x 20
        area_abs=800,
        -- maximum side ratio = 2.5:1
        side_th=2.5
    }
}
local area_th, area_abs, side_th

for key, target in ipairs(targets) do
    labeljson = {}
    fh, err = io.open(base_path .. 'annotations/' .. target .. '.json')
    target_path = target .. '/'

    str = fh:read()
    json = cjson.decode(str)
    imgs = json['images']
    categs = json['categories']
    annotations = json['annotations']

    print(json['info'])

    function mkdir(name)
        pwd = lfs.currentdir()
        ret = lfs.chdir(name)
        if ret ~= true then
            ret, err = lfs.mkdir(name)
        end
        lfs.chdir(pwd)

        return ret
    end

    local img = nil
    local imgid = -1
    local h,w

    if mkdir(base_path .. target_path) then
        for i, obj in ipairs(annotations) do
            if imgid ~= obj['image_id'] then
                imgid = obj['image_id']
                img = image.load(base_path .. 'images/' .. imgs[imgid+1]['file_name'])
                h, w = img:size(2), img:size(3)

                print('Processing ' .. imgs[imgid+1]['file_name'])
            end

            if img ~= nil then
                segmentation = {}
                segmentation[1] = torch.reshape(torch.DoubleTensor{obj['segmentation']}, #obj['segmentation'][1], 1)
                rs = maskApi.frPoly(segmentation, h, w)
                bbox = maskApi.toBbox(rs)

                -- bbox : x, y, w, h
                bbox = bbox:storage()

                categ = categs[obj['category_id']+1]
                if categ['supercategory'] == 'chips' then
                    side_th = bbox_shapes[1].side_th
                    area_th = bbox_shapes[1].area_th
                    area_abs = bbox_shapes[1].area_abs
                else
                    side_th = bbox_shapes[2].side_th
                    area_th = bbox_shapes[2].area_th
                    area_abs = bbox_shapes[2].area_abs                
                end

                side = {}
                if bbox[3] > bbox[4] then
                    side[1] = bbox[3]
                    side[2] = bbox[4]
                else
                    side[1] = bbox[4]
                    side[2] = bbox[3]
                end
                area = maskApi.area(rs)[1] 
                if side[1] <= side_th * side[2] and area >= area_abs and area >= area_th * bbox[3] * bbox[4] then
                    cropped_img = image.crop(img, bbox[1], bbox[2], bbox[1]+bbox[3], bbox[2]+bbox[4])

                    obj_label = string.format('%s_%s', categ['supercategory'], categ['name'])
                    dst_path = base_path .. target_path .. obj_label .. '/'
                    mkdir(dst_path)
                    image.save(dst_path .. imfile_prefix .. string.format('%05d.png', obj['id']), cropped_img)
                    
                    elem = { id=obj['id'], img=imgs[imgid+1]['file_name'], label=obj_label, minx=bbox[1], miny=bbox[2], maxx=bbox[1]+bbox[3], maxy=bbox[2]+bbox[4]}
                    table.insert(labeljson, elem)
                end
            end
        end
    end
    fh:close()

    fh, err = io.open(base_path .. 'annotations/' .. target .. '_bbox.json', 'w')
    fh:write(cjson.encode(labeljson))
    fh:close()
end