from utils_map import compute_pre_rec, voc_ap, calc_inter_ap
import json

def calculate_ap(gt_classes, opt, dr, gt):
    specific_iou_flagged = False
    if opt.set_class_iou is not None:
        specific_iou_flagged = True
    count_true_positives = {}
    size_dict = {}
    size_dict["small"], size_dict["medium"], size_dict["large"] = [], [], []

    for class_index, class_name in enumerate(gt_classes):
        gt_data = gt[class_name]
        for idx, obj in enumerate(gt_data):
            size_dict[obj["size_name"]].append(obj)

    for idx, scale in enumerate(size_dict):
        count_true_positives[scale] = 0

        for class_index, class_name in enumerate(gt_classes):
            dr_data = dr[class_name]

            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]
                ovmax = -1
                gt_match = -1
                bb = [float(x) for x in detection["bbox"].split()]
                if detection["confidence"] < opt.confidence_threshold:
                    continue
                for idx, obj in enumerate(size_dict[scale]):
                    if obj["file_id"] == file_id:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[0] + bb[2], bbgt[0] + bbgt[2]),
                              min(bb[1] + bb[3], bbgt[1] + bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # ua = compute overlap (IoU) = area of intersection/ area of union
                            ua = ((bb[2] + 1) * (bb[3] + 1) + (bbgt[2] + 1) * (bbgt[3] + 1)) - iw * ih
                            IoU = iw * ih / ua
                            if IoU > ovmax:
                                ovmax = IoU
                                gt_match = obj

                iou_threshold = opt.iou_threshold
                if specific_iou_flagged:
                    specific_iou_classes = opt.set_class_iou[::2]
                    iou_list = opt.set_class_iou[1::2]
                    if class_name in specific_iou_classes:
                        index = specific_iou_classes.index(scale)
                        iou_threshold = float(iou_list[index])
                if ovmax >= iou_threshold:
                    if not bool(gt_match["size_used"]):
                        gt_match["size_used"] = True
                        count_true_positives[scale] +=1
                        obj = gt_match

    return count_true_positives
