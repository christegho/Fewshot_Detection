from darknet_meta import Darknet
import dataset
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import *
from cfg import cfg
from cfg import parse_cfg
import os
import pickle
import sys

def valid(valid_images, darknetcfg, learnetcfg, weightfile, dynamic_weight_file, 
          prefix, conf_thresh, nms_thresh):
    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]
    
    m = Darknet(darknetcfg, learnetcfg)
    m.print_network()
    m.load_weights(weightfile)
    m.cuda()
    m.eval()
    
    print('===> Loading dynamic weights from {}...'.format(dynamic_weight_file))
    with open(dynamic_weight_file, 'rb') as f:
        rws = pickle.load(dynamic_weight_file)
        dynamic_weights = [Variable(torch.from_numpy(rw)).cuda() for rw in rws]
        
    valid_dataset = dataset.listDataset(valid_images, shape=(m.width, m.height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))
    valid_batchsize = 2

    kwargs = {'num_workers': 4, 'pin_memory': True}
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs) 

    n_cls = 20
    lineId = -1
    
    for batch_idx, (data, target) in enumerate(valid_loader):
        data = data.cuda()
        data = Variable(data, volatile = True)
        output = m.detect_forward(data, dynamic_weights)

        if isinstance(output, tuple):
            output = (output[0].data, output[1].data)
        else:
            output = output.data

        batch_boxes = get_region_boxes_v2(output, n_cls, 
            conf_thresh, m.num_classes, m.anchors, m.num_anchors, 0, 1)

        if isinstance(output, tuple):
            bs = output[0].size(0)
        else:
            assert output.size(0) % n_cls == 0
            bs = output.size(0) // n_cls
        for b in range(bs):
            lineId = lineId + 1
            imgpath = valid_dataset.lines[lineId].rstrip()
            print(imgpath)
            imgid = os.path.basename(imgpath).split('.')[0]
            width, height = get_image_size(imgpath)
            i = 2
            oi = b * n_cls + i
            boxes = batch_boxes[oi]
            boxes = nms(boxes, nms_thresh)
            boxes_ = []
            for box in boxes:
                x1 = (box[0] - box[2]/2.0) * width
                y1 = (box[1] - box[3]/2.0) * height
                x2 = (box[0] + box[2]/2.0) * width
                y2 = (box[1] + box[3]/2.0) * height

                det_conf = box[4]
                for j in range((len(box)-5)/2):
                    cls_conf = box[5+2*j]
                    cls_id = box[6+2*j]
                    prob =det_conf * cls_conf
                boxes_ += [[prob, x1, y1, x2, y2]]

            plot_boxes_(imgpath, boxes_, '{}/{}.jpg'.format(prefix, imgid))



def plot_boxes_(imgpath, boxes, savename):
    img = Image.open(imgpath).convert('RGB')
    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = box[1:5]
        cls_conf = box[0]
        rgb = (0, 255, 127)
        draw.text((x1, y1), 'canister, {}'.format(cls_conf), fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline = rgb)
    print("save plot results to %s" % savename)
    img.save(savename)
    return img

if __name__ == '__main__':
    
    if len(sys.argv) in [5,6,7]:
        datacfg = sys.argv[1]
        darknet = parse_cfg(sys.argv[2])
        learnet = parse_cfg(sys.argv[3])
        weightfile = sys.argv[4]
        prefix = 'preds_2'
        conf_thresh = 0.2
        nms_thresh = 0.45
        valid_images = '/home/chris/safariland-element/val_can.txt'
        dynamic_weight_file = 'dynamic_weights.pkl'

        
        if len(sys.argv) >= 6:
            gpu = sys.argv[5]
        else:
            gpu = '0'
        if len(sys.argv) == 7:
            use_baserw = False
        else:
            use_baserw = False

        data_options  = read_data_cfg(datacfg)
        net_options   = darknet[0]
        meta_options  = learnet[0]
        data_options['gpus'] = gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        # Configure options
        cfg.config_data(data_options)
        cfg.config_meta(meta_options)
        cfg.config_net(net_options)

        if not os.path.exists(prefix):
            os.makedirs(prefix)
        valid(valid_images, darknet, learnet, weightfile, dynamic_weight_file, 
              prefix, conf_thresh, nms_thresh)
    else:
        print('Usage:')
        print(' python valid.py datacfg cfgfile weightfile')
