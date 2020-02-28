"""the latest version"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
import json
import pandas as pd
import option

opt = option.options

"""path위치 등 txt config를 파일에 저장해 주는 함수"""
def print_configuration(result, conf):
    with open(result, 'a') as result:
        result.write("\n## Show Configuration ##")
        result.write("\nGround Truth Json file path: " + conf.gt_json_path + "\n")
        result.write("Detection Result Json file path: " + conf.dr_json_path + "\n")
        result.write("IoU threshold is " + str(conf.iou_threshold) +
                     ". Detection results under threshold are counted as FP.\n")
        result.write("Confidence Threshold is "+ str(conf.confidence_threshold) +
                     ". Detection results under threshold are not counted.\n")
    return

"""interpolation함수/recall이 증가함에 따라interpolation구간 별 precision의 최대값으로 precision을 정하는 함수"""
def calc_interpolated_prec(desired_rec, latest_pre, rec, prec):
    recall_precision = np.array([rec, prec])
    recall_precision = recall_precision.T

    inter_recall = recall_precision[recall_precision[:, 0] >= desired_rec]
    inter_precision = inter_recall[:, 1]

    if len(inter_precision) > 0:
        inter_precision = max(inter_precision)
        latest_pre = inter_precision
    else:
        inter_precision = latest_pre
    return inter_precision, latest_pre

"""interpolation함수/ 최종적인 recall, precision list를 만든다."""
def calc_inter_ap(opt, rec, prec):
    inter_precisions = []
    latest_pre = 0
    for i in range(opt.n_interpolation):
        recall = float(i)/(opt.n_interpolation - 1)
        inter_precision, latest_pre = calc_interpolated_prec(recall, latest_pre, rec, prec)
        inter_precisions.append(inter_precision)
    return np.array(inter_precisions).mean()


"""throw error and exit"""


def error(msg):
    print(msg)
    sys.exit(0)


"""check if the number is a float between 0.0 and 1.0"""


def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if 0.0 < val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


"""
Calculate the AP given the recall and precision array
    1) Compute version of measured precision/recall curve with precision monotonically decreasing
    2) Compute the AP as the Area Under this curve by numerical integration   ###not interpolated 
"""

"""ap, precision, recall을 리턴하는 함수"""
def voc_ap(rec, prec):

    rec.insert(0, 0.0)  # insert 0.0 at beginning of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]


    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])  # 항상 i+1 의 값보다 i 가 크다 ==> 항상 감소하는 curve됨

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)  # if it was matlab would be i +1
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i-1])*mpre[i])
    return ap, mrec, mpre


"""
convert the lines of a file to a list
"""


# def file_lines_to_list(path):
#     # open txt file lines to a list
#     with open(path) as f:
#         content = f.readlines()
#     # remove whitespace characters like '\n' at the end of each line
#     content = [x.strip() for x in content]
#     return content


"""gt의 데이터를 scale 별로 나누는 함수"""
def area_check(area, size_threshold):
    if area <= (size_threshold) ** 2:
        size_name = "small"
    elif (size_threshold) ** 2 < area <= (3 * size_threshold) ** 2:
        size_name = "medium"
    elif area > (3 * size_threshold) ** 2:
        size_name = "large"
    else:
        ValueError("Check the area")
    return size_name

"""gt의 데이터를 딕셔너리 형으로 정리하는 함수"""
def get_gt_match(gt_path, class_dict):
    gt_counter_per_classes = {}
    counter_images_per_classes = {}

    gt_counter_per_sizes = {}
    counter_images_per_sizes = {}
    size_threshold = opt.size_threshold

    already_seen_classes = []
    already_seen_sizes = []
    gt_class = {}

    for key, value in class_dict.items():
        gt_class[value] = []
    with open(gt_path) as json_file:
        json_data = json.load(json_file)
        json_annotations = json_data["annotations"]
        json_annotations = sorted(json_annotations, key=lambda json_annotations: (json_annotations["category_id"]))
        df = pd.DataFrame(json_annotations)

        for idx, row in df.iterrows():
            image_id = str(row["image_id"])
            left, top, width, height = str(row["bbox"][0]), str(row["bbox"][1]), str(row["bbox"][2]), str(
                row["bbox"][3])
            bbox = left + " " + top + " " + width + " " + height
            area = row["area"]
            class_id = str(row["category_id"])
            class_name = class_dict[class_id]

            size_name = area_check(area, size_threshold)

            if class_name in gt_counter_per_classes:
                gt_counter_per_classes[class_name] += 1
            else:
                # if class did not exits yet
                gt_counter_per_classes[class_name] = 1

            if size_name in gt_counter_per_sizes:
                gt_counter_per_sizes[size_name] += 1
            else:
                gt_counter_per_sizes[size_name] = 1


            if size_name not in already_seen_sizes:
                if size_name in counter_images_per_sizes:
                    counter_images_per_sizes[size_name] += 1
                else:
                    counter_images_per_sizes[size_name] = 1
                already_seen_sizes.append(size_name)

            gt_class[class_name].append({"file_id": image_id, "class_name": class_name, "area": area,
                                         "size_name": size_name, "bbox": bbox, "used": False, "size_used": False})

    return gt_counter_per_classes, gt_counter_per_sizes, counter_images_per_sizes, gt_class

"""class 별로 IoU를 다르게 할 시에만 사용하는 함수다."""
def check_format_class_iou(opt, gt_classes):
    n_args = len(opt.set_class_iou)
    error_msg = \
        '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
    if n_args % 2 != 0:
        error('Error, missing arguments. Flag usage:' + error_msg)

    specific_iou_classes = opt.set_class_iou[::2]  # even elements
    iou_list = opt.set_class_iou[1::2]  # odd elements
    if len(specific_iou_classes) != len(iou_list):
        error('Error, missing arguments. Flag usage:' + error_msg)
    for tmp_class in specific_iou_classes:
        if tmp_class not in gt_classes:
            error('Error, unknown class \"'+tmp_class + '\".Flag usage:' + error_msg)
    for num in iou_list:
        if not is_float_between_0_and_1(num):
            error('Error, IOU must be between 0 and 1. Flag usage:' + error_msg)

"""class_dict를 만드는 함수/ class_dict에는 category_id와 class_name이 매칭되어 있다."""
def make_gt_list(gt_json_path):
    class_dict = dict()
    with open(gt_json_path) as json_file:
        json_data = json.load(json_file)
        json_categories = json_data["categories"]
        category_df = pd.DataFrame(json_categories)

        for idx, row in category_df.iterrows():
            category_id = str(row["id"])
            category_name = row.get("name")
            class_dict[category_id] = category_name
    return class_dict

"""detection result를 딕셔너리 형으로 정리하는 함수다."""
def dr_json(dr_json_path, class_dict):
    det_counter_per_classes = {}
    dr_class = {}
    for key, value in class_dict.items():
        dr_class[value] = []
    with open(dr_json_path) as origin_dr_path:
        json_data = json.load(origin_dr_path)
        json_annotations = json_data["annotations"]
        json_annotations = sorted(json_annotations, key=lambda json_annotations:(json_annotations['category_id']))
        df = pd.DataFrame(json_annotations)
        for idx, row in df.iterrows():
            image_id = str(row['image_id'])
            try:
                class_name = class_dict[str(row["category_id"])]
            except KeyError or ValueError:
                error("No matching class name in gt. This category id  not exist in gt: " + row["category_id"])
            confidence = row['score']
            left, top, width, height = str(row["bbox"][0]), str(row["bbox"][1]), str(row["bbox"][2]), str(row["bbox"][3])
            if class_name in det_counter_per_classes:
                det_counter_per_classes[class_name] += 1
            else:
                det_counter_per_classes[class_name] = 1
            bbox = left + " " + top + " " + width + " " + height
            dr_class[class_name].append({"confidence": confidence, "file_id": image_id, "bbox": bbox})
            #bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse = True)

    for key, value in class_dict.items():
        dr_class[value].sort(key=lambda x: float(x['confidence']), reverse = True)

    return det_counter_per_classes, dr_class


"""precision, recall,tp,fp를 전의 값들을 누적해서 리스트를 만든다."""
def compute_pre_rec(fp, tp, class_name, gt_counter_per_class):
    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val
    # fp와 tp의 리스트를 fp인지 아닌지 반영하는 0 or 1이 아니고, 해당 인덱스까지 fp가 몇번 나왔는지 누적값으로 만든다.
    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx] / gt_counter_per_class[class_name])
    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx] / (fp[idx] + tp[idx]))

    return rec, prec

"""IoU, confidence threshold에 따라 ap를 구하는 부분으로, plot도 포함한다."""
def calculate_ap(results_file_path, plot_result_path, gt_classes, opt, gt_counter_per_class, dr, gt):
    specific_iou_flagged = False
    if opt.set_class_iou is not None:
        specific_iou_flagged = True
    sum_AP = 0.0
    ap_dictionary = {}
    precision_dict = {}
    recall_dict = {}

    with open(results_file_path, 'w') as results_file:
        results_file.write("# AP and precision/recall per class \n")
        if opt.see_pre_rec is False:
            results_file.write(" ==> opt.see_pre_rec is false\n")
        count_true_positives = {}
        texts = ""
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            dr_data = dr[class_name]
            gt_data = gt[class_name]
            tp = []
            fp = []
            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]
                ovmax = -1
                gt_match = -1
                bb = [float(x) for x in detection["bbox"].split()]
                if detection["confidence"] < opt.confidence_threshold:
                    continue
                for idx, obj in enumerate(gt_data):
                    if not obj["file_id"] == file_id:
                        continue
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
                        index = specific_iou_classes.index(class_name)
                        iou_threshold = float(iou_list[index])
                if ovmax >= iou_threshold:
                    if not bool(gt_match["used"]):
                        tp.append(1)
                        fp.append(0)
                        gt_match["used"] = True
                        count_true_positives[class_name] +=1
                        obj = gt_match

                    else:
                        fp.append(1)
                        tp.append(0)
                else:
                    fp.append(1)
                    tp.append(0)
            rec, prec = compute_pre_rec(fp, tp, class_name, gt_counter_per_class)
            if opt.no_interpolation:
                ap, mrec, mpre = voc_ap(rec[:], prec[:])
            else:
                ap, mrec, mpre = voc_ap(rec[:], prec[:])
                ap = calc_inter_ap(opt, rec[:], prec[:])
            sum_AP += ap

            # print(mrec)

            text = "{0:.2f}%".format(
                ap * 100) + " = " + class_name + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)
            rounded_prec = ["%.2f" % elem for elem in prec]
            rounded_rec = ["%.2f" % elem for elem in rec]

            if opt.see_pre_rec:
                results_file.write(
                    text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")

            if opt.draw_plot:
                plt.plot(rec, prec, '.')
                # add a new penultimate point to the list (mrec[-2], 0.0)
                # since the last line segment (and respective area) do not affect the AP value
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mpre[:-1] + [0.0] + [mpre[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')

                # set window title
                fig = plt.gcf()  # gcf - get current figure
                fig.canvas.set_window_title('AP ' + class_name)

                # save the plot
                fig.savefig(os.path.join(plot_result_path, class_name +".png"))
                plt.cla()  # clear axes for next plot

            if not opt.quiet:
                print(text)

            ap_dictionary[class_name] = ap
            precision_dict[class_name] = str(prec)
            recall_dict[class_name] = str(rec)

            # print(precision_dict[class_name])

            texts = texts + class_name + " ap: "+ "{0:.2f}%".format(ap*100)  +"\n"

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / len(gt_classes)
        text = "mAP = {0:.2f}%\n".format(mAP * 100)
        results_file.write(texts)
        results_file.write(text)
        print(text)
    return count_true_positives, mAP, ap_dictionary, precision_dict, recall_dict
