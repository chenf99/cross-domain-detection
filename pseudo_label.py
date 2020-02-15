from collections import defaultdict

import argparse
import chainer
import numpy as np
import os
from chainercv.datasets import voc_bbox_label_names
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

import helper
import opt
from lib.label_file import LabelFile


def main():
    chainer.config.train = False
    chainer.config.cv_resize_backend = "cv2"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--data_type', choices=opt.data_types, required=True)
    parser.add_argument('--det_type', choices=opt.detectors, required=True,
                        default='ssd300')
    parser.add_argument('--result', required=True)
    parser.add_argument('--load', help='load original trained model')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=32)
    args = parser.parse_args()

    model_args = {'n_fg_class': len(voc_bbox_label_names),
                  'pretrained_model': 'voc0712'}
    model = helper.get_detector(args.det_type, model_args)

    if args.load:
        chainer.serializers.load_npz(args.load, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    model.use_preset('evaluate')

    dataset = helper.get_detection_dataset(args.data_type, 'train', args.root)

    iterator = chainer.iterators.SerialIterator(
        dataset, args.batchsize, repeat=False, shuffle=False)

    # model.predict([img]) -> ([pred_bbox], [pred_label], [pred_score])
    imgs, pred_values, gt_values = apply_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)))  # apply model.predict to batches from iterator
    # gt_values from dataset, since function 'predict' only needs img as param
    # delete unused iterator explicitly
    del imgs

    pred_bboxes, pred_labels, pred_scores = pred_values
    _, gt_labels = gt_values

    ids = []

    for i, (pred_b, pred_l, pred_s, gt_l) in enumerate(
            zip(pred_bboxes, pred_labels, pred_scores, gt_labels)):

        labels = dataset.labels
        proper_dets = defaultdict(list)  # 这种dict默认提供了key
        name = dataset.ids[i]  # img的id
        cnt = 0

        gt_l = set(gt_l)
        for l_ in set(gt_l):
            cnt += 1
            class_indices = np.where(pred_l == l_)[0]  # 所有预测label正确的坐标
            if len(class_indices) == 0:
                continue
            scores = pred_s[class_indices]
            ind = class_indices[np.argsort(scores)[::-1][0]]  # top1
            assert (l_ == pred_l[ind])
            proper_dets[labels[l_]].append(pred_b[ind][[1, 0, 3, 2]])
            # 这个label预测正确的bbox+1,并且弄成(xmin, ymin, xmax, ymax)的形式

        if cnt == 0:
            continue  # 没有预测正确的label,直接跳过写入Annotation步骤

        ids.append(dataset.ids[i] + '\n')
        filename = os.path.join(args.result, 'Annotations', name + '.xml')
        img_path = os.path.join(args.root, 'JPEGImages', name + '.jpg')
        labeler = LabelFile(filename, img_path, dataset.actual_labels)
        labeler.savePascalVocFormat(proper_dets)

    txt = 'ImageSets/Main/train.txt'
    with open(os.path.join(args.result, txt), 'w') as f:
        f.writelines(ids)  # 重写这个文件是为了保证每个图片都有对应的Annotation(上述步骤生成的)
    print('Saved to {:s}'.format(args.result))


if __name__ == '__main__':
    main()
