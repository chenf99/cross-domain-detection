import argparse
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import json

from collections import defaultdict
from lib.label_file import LabelFile
from utils import voc_labels, label_map, bam_labels
from model import SSD300
from datasets import PascalVOCDataset
from PIL import Image
from tqdm import tqdm


def path_to_id(img):
    index = img.find('JPEGImages/')
    img = img[index + len('JPEGImages/'):]
    index = img.find('.')
    return img[:index]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--data_folder', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--checkpoint', help='path of the pretrained model', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_type', choices=('clipart, bam'), required=True)
    args = parser.parse_args()

    n_classes = len(label_map)  # number of different types of objects
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    workers = 4  # number of workers for loading data in the DataLoader
    keep_difficult = True

    cudnn.benchmark = True

    if args.checkpoint == 'pretrained_ssd300.pth.tar':
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = checkpoint['model']
        for m in model.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')
    else:
        model = SSD300(n_classes=n_classes, device=device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])

    model = model.to(device)
    # Switch to eval mode
    model.eval()

    dataset = PascalVOCDataset(args.data_folder, split='train', keep_difficult=keep_difficult)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                             collate_fn=dataset.collate_fn, num_workers=workers, pin_memory=True)

    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_labels = list()

    # only use labels of ground truth, not boxes
    with torch.no_grad():
        for images, boxes, labels, difficulties in tqdm(dataloader, desc='pseudo labeling'):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)
            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200, device=device)
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_labels.extend(labels)

    ids = []
    with open(os.path.join(args.data_folder, 'TRAIN_images.json'), 'r') as j:
        images = json.load(j)
        ids = [path_to_id(img) for img in images]

    new_ids = []

    for i, (pred_b, pred_l, pred_s, gt_l) in enumerate(
            zip(det_boxes, det_labels, det_scores, true_labels)):
        pred_b = pred_b.cpu().numpy()
        pred_l = pred_l.cpu().numpy()
        pred_s = pred_s.cpu().numpy()
        gt_l = gt_l.cpu().numpy()

        labels = ('background',) + voc_labels
        proper_dets = defaultdict(list)  # 这种dict默认提供了key
        name = ids[i]  # img的id
        cnt = 0

        gt_l = set(gt_l)
        for l_ in gt_l:
            cnt += 1
            class_indices = np.where(pred_l == l_)[0]  # 所有预测label正确的坐标
            if len(class_indices) == 0:
                continue
            scores = pred_s[class_indices]
            ind = class_indices[np.argsort(scores)[::-1][0]]  # top1
            assert (l_ == pred_l[ind])
            # Transform to original image dimensions
            img_path = os.path.join(args.root, 'JPEGImages', name + '.jpg')
            img = Image.open(img_path, mode='r')
            original_dims = np.array([img.width, img.height, img.width, img.height])
            pred_b[ind] = pred_b[ind] * original_dims
            # 这个label预测正确的bbox+1
            proper_dets[labels[l_]].append(pred_b[ind])

            # 删除这个预测
            # pred_b = np.concatenate((pred_b[:ind], pred_b[ind + 1:]), 0)
            # pred_l = np.concatenate((pred_l[:ind], pred_l[ind + 1:]), 0)
            # pred_s = np.concatenate((pred_s[:ind], pred_s[ind + 1:]), 0)

        if cnt == 0:
            continue  # 没有ground turth label,直接跳过写入Annotation步骤

        new_ids.append(ids[i] + '\n')
        filename = os.path.join(args.result, 'Annotations', name + '.xml')
        img_path = os.path.join(args.root, 'JPEGImages', name + '.jpg')
        actual_labels = voc_labels if args.data_type == 'clipart' else bam_labels
        labeler = LabelFile(filename, img_path, actual_labels)
        labeler.savePascalVocFormat(proper_dets)

    txt = 'ImageSets/Main/trainval.txt'
    with open(os.path.join(args.result, txt), 'w') as f:
        f.writelines(new_ids)  # 重写这个文件是为了保证每个图片都有对应的Annotation(上述步骤生成的)
    print('Saved to {:s}'.format(args.result))


if __name__ == '__main__':
    main()
