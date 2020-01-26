from PIL import Image, ImageDraw
from utils import *
import sys
namesfile = 'data/voc.names'
class_names = load_class_names(namesfile)

imgfile_path = '../safariland-element/CanistersRealTrainVal/JPEGImages/{}.jpg'

def plot_boxes_(img, boxes, savename=None, class_names=None):
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]);
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = box[1]
        y1 = box[2]
        x2 = box[3]
        y2 = box[4]

        rgb = (255, 0, 0)
#         if len(box) >= 7 and class_names:
        cls_conf = box[0]
#             cls_id = box[6]
#             print('%s: %f' % (class_names[cls_id], cls_conf))
        classes = len(class_names)
        offset = 2 * 123457 % classes
        red   = get_color(2, offset, classes)
        green = get_color(1, offset, classes)
        blue  = get_color(0, offset, classes)
        rgb = (red, green, blue)
        draw.text((x1, y1), 'canister, {}'.format(cls_conf), fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline = rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img

def save_images(exp):
    with open('results/metatunetest{}_novel0_neg0/ene000020/comp4_det_test_bird.txt'.format(exp)) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
    im_bxs = {}
    threshs = []
    for line in content:
        ctn = line.split(' ')
        img_name = ctn[0]
        if img_name not in im_bxs:
            im_bxs[img_name] = []
    #     import pdb; pdb.set_trace()
        threshs += [ctn[1]]
        if float(ctn[1]) > 0.2:
            im_bxs[img_name].append([float(b) for b in ctn[1:]])
    threshs = np.array(threshs).astype(float)
    print("mean threashold", threshs.mean())
    print("mean threashold > 0.1", threshs[threshs>0.1].mean())
#     return
    for img_ in im_bxs:
        img = Image.open(imgfile_path.format(img_)).convert('RGB')
        boxes = im_bxs[img_]
        plot_boxes_(img, boxes, 'preds/{}.jpg'.format(img_), class_names)

if __name__ == '__main__':
    save_images(sys.argv[1])
