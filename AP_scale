import os
import shutil
import json
import sys
import numpy as np
from m_main import get_configurations
import matplotlib.pyplot as plt
import pandas as pd

# from glob import glob

args = get_configurations()



""" composed of 3 parts = file preset, calculation, results"""

"""
1. file preset part
"""

# path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# make sure that the cwd() is the location of the python script (so that every path makes sense)

gt_json_path = 'C:/Users/Medical-Information/PycharmProjects/project_metric/input/ground_truth/ground_truth.json'
dr_json_path = 'C:/Users/Medical-Information/PycharmProjects/project_metric/input/detection_results/detection_results.json'


# make temp path
TEMP_FILES_PATH = "temp_files"
if not os.path.exists(TEMP_FILES_PATH):
    os.makedirs(TEMP_FILES_PATH)


# make results path
result_path = "result_file"
if os.path.exists(result_path):
    shutil.rmtree(result_path)
    os.makedirs(result_path)
else:
    os.makedirs(result_path)

#make plot path
plot_result_path = "plot_figures"
if args.draw_plot:
    if args.plot_save:
        if os.path.exists(plot_result_path):
            shutil.rmtree(plot_result_path)
            os.makedirs(plot_result_path)
        else:
            os.makedirs(plot_result_path)



'''ignore'''
if args.ignore is None:
    args.ignore = []


"""error msg"""
def error(msg):
    print(msg)
    sys.exit(0)

#
# """check the range of figure"""
# def is_float_between_0_and_1(value):
#     try:
#         val = float(value)
#         if 0.0 < val < 1.0:
#             return True
#         else:
#             return False
#     except ValueError:
#         return False


'''load ground truth files and elements
    Load each of the ground-truth files into a temporary ".json" file
'''
"""get class_dict"""
with open(gt_json_path) as json_file:
    json_data = json.load(json_file)
    json_categories = json_data["categories"]
    category_id = []
    category_name = []
    class_dict = {}
for index in range(len(json_categories)):
    category_id = str(json_categories[index]["id"])
    category_name = str(json_categories[index]["name"])
    class_dict[category_id] = category_name



"""make dataframe"""
def make_df(gt_json_path):
    scale = []
    json_annotations = json_data["annotations"]
    json_annotations = sorted(json_annotations, key=lambda json_annotations: (json_annotations['image_id']))
    df = pd.DataFrame(json_annotations)

    for idx, row in df.iterrows():
        size = float(row["area"])
        category_id = str(row["category_id"])
        if size <= 32**2:
            scale.append("small")
        elif size > 32**2 and size <= 96**2:
            scale.append("medium")
        elif size > 96**2:
            scale.append("large")
        else:
            None
    df['scale']=scale

    return df

df = make_df(gt_json_path)


"""make temp gt"""
def scale_gt(gt_json_path, TEMP_FILES_PATH, df):
    gt_counter_per_scale = {}
    bounding_boxes = []

    for idx, row in df.iterrows():
        file_id = str(row["image_id"])
        category_id = row["category_id"]
        class_name = class_dict[str(category_id)]
        scale = row["scale"]

        # create gt dictionary
        left, top, width, height = str(row["bbox"][0]), str(row["bbox"][1]), str(row["bbox"][2]), str(row["bbox"][3])
        bbox = left + " " + top + " " + width + " " + height
        bounding_boxes.append(
            {"file_name": file_id, "class_name": class_name, "scale": scale, "bbox": bbox, "used": False})

        # count how many gts are in one class(dictionary)

    for item in df["scale"]:
        gt_counter_per_scale.setdefault(item, 0)
        gt_counter_per_scale[item] += 1

    with open(TEMP_FILES_PATH + '/gt_match.json', 'w') as outfile:
        json.dump(bounding_boxes, outfile)

    return gt_counter_per_scale





def dr_json(dr_json_path, TEMP_FILES_PATH, class_dict):
    det_counter_per_classes = {}
    with open(dr_json_path) as origin_dr_path:
        json_data = json.load(origin_dr_path)
        json_annotations = json_data["annotations"]
        json_annotations = sorted(json_annotations, key=lambda json_annotations:(json_annotations['category_id']))
        df = pd.DataFrame(json_annotations)
        for key, value in class_dict.items():
            bounding_boxes = []
            for idx, row in df.iterrows():
                image_id = str(row['image_id'])
                if str(row['category_id']) == str(key):
                    tmp_class_name, confidence = value, row['score']
                    left, top, width, height = str(row["bbox"][0]), str(row["bbox"][1]),\
                                               str(row["bbox"][2]), str(row["bbox"][3])
                    if tmp_class_name in det_counter_per_classes:
                        det_counter_per_classes[tmp_class_name] +=1
                    else:
                        det_counter_per_classes[tmp_class_name] = 1
                    bbox = left + " " + top + " " + width + " " + height
                    bounding_boxes.append({"confidence": confidence, "file_id": image_id, "bbox": bbox})
            bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse = True)
            with open(TEMP_FILES_PATH + '/' + value + '_dr.json', 'w') as outfile:
                json.dump(bounding_boxes, outfile)

    return det_counter_per_classes



"""
2. calculattion part
"""

"""Overall Calculation Frame"""
def voc_ap(rec, prec):

    rec.insert(0, 0.0)  # insert 0.0 at beginning of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]

    # This part makes the precision monotonically decreasing as recall increases (end-->beginning)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # This part creates a list of index, where the recall changes (from 1.0 to 0.9 / 0.9 to 0.8 )
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    # The Average Precision (AP) is the average point in precision
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i]) #intergrating version, no need ap_sum, just ap

    return ap, mrec, mpre


"""
interpolation preset
"""
def calc_interpolated_prec(desired_rec, latest_pre, rec, prec):
    recall_precision = np.array([rec, prec])
    recall_precision = recall_precision.T

    inter_recall = recall_precision[recall_precision[:, 0] >= desired_rec]
    inter_precision = inter_recall[:, 1]

    if len(inter_precision) > 0:
        inter_precision = max(inter_precision)
        latest_pre = inter_precision
    else:
        inter_precision = 0
    return inter_precision, latest_pre


"""
interpolation
"""
def calc_inter_ap(args, rec, prec):
    inter_precisions = []
    latest_pre = 0
    for i in range(args.N_inter):
        recall = float(i) / (args.N_inter - 1)
        inter_precision, latest_pre = calc_interpolated_prec(recall, latest_pre, rec, prec)
        inter_precisions.append(inter_precision)
    return np.array(inter_precisions).mean()



"""get ap and plot"""
def calculate_ap(TEMP_FILE_PATH, result_path, gt_classes, gt_scale, args, gt_counter_per_scale):
    sum_AP = 0.0
    ap_dictionary = {}
    scale = ["small", "medium", "large"]

    # open file to store the results
    with open(result_path + "/results.txt", 'w') as results_file:
        results_file.write("----- AP and precision/recall per class ----- \n")
        count_true_positives = {}
        count_false_positives = {}

        # for scale_index, scale_name in enumerate(gt_scale):
        for scale_index, scale_name in enumerate(gt_scale):
            count_true_positives[scale_name] = 0
            tp = []
            fp = []

            gt_file = TEMP_FILES_PATH + '/gt_match.json'

            for class_index, class_name in enumerate(gt_classes):
                dr_file = TEMP_FILE_PATH + "/" + class_name + "_dr.json"
                dr_data = json.load(open(dr_file))
                '''load detection results of that class'''
                for idx, detection in enumerate(dr_data):
                    file_id = detection["file_id"]  # assign detection results to gt object if any
                    ground_truth_data = json.load(open(gt_file))
                    IoUmax = -1
                    gt_match = -1
                    # load detected object bounding-box
                    bb = [float(x) for x in detection["bbox"].split()]

                    for obj in ground_truth_data:
                        if obj["scale"] == scale_name:
                            # if obj["class_name"] == class_name:
                            bbgt = [float(x) for x in obj["bbox"].split()]
                            # 순서: left top right bottom
                            # bi = detection과 gt 중 교집합 box의 좌표
                            bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]),
                                  min(bb[2] + bb[0], bbgt[2] + bbgt[0]), min(bb[3] + bb[1], bbgt[3]) + bbgt[1]]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                # ua = compute overlap (IoU) = area of intersection/ area of union
                                ua = ((bb[2] + 1) * (bb[3] + 1) + (bbgt[2] + 1) * (bbgt[3] + 1)) - iw * ih
                                IoU = iw * ih / ua
                                if IoU > IoUmax:
                                    IoUmax = IoU
                                    gt_match = obj
                    IoU_th = args.IoU_th
                    if IoUmax >= IoU_th:
                        if not bool(gt_match["used"]):
                            # true positive
                            tp.append(1)
                            fp.append(0)
                            gt_match["used"] = True
                            count_true_positives[scale_name] += 1
                            # update json file
                            with open(gt_file, 'w') as f:
                                f.write(json.dumps(ground_truth_data))
                        else:
                            # false positive (multiple detection)
                            fp.append(1)
                            tp.append(0)
                            # count_false_positives[scale_name] += 1
                    else:
                        fp.append(1)
                        tp.append(0)
                        # count_false_positives[scale_name] += 1

            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val

            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val

            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx] / gt_counter_per_scale[scale_name])

            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx] / (fp[idx] + tp[idx]))

            # rec, prec = compute_pre_rec(fp, tp, class_name, gt_counter_per_class)

            if args.no_interpolation:
                ap, mrec, mpre = voc_ap(rec[:], prec[:])
            else:
                ap, mrec, mpre = voc_ap(rec[:], prec[:])
                ap = 0.0
                ap = calc_inter_ap(args, rec[:], prec[:])
            # ap, mrec, mpre = voc_ap(rec[:], prec[:])


            sum_AP += ap
            text = str(scale_name) + " AP " + " = " + "{0:.2f}%".format(ap * 100) # scale_name + " AP = {0:.2f}%".format(ap*100)
            rounded_prec = ['%.2f' % elem for elem in prec]
            rounded_rec = ['%.2f' % elem for elem in rec]
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")

            """Draw plot"""
            if args.draw_plot:
                plt.plot(rec, prec, '-o')
                # add a new penultimate point to the list (mrec[-2], 0.0)
                # since the last line segment (and respective area) do not affect the AP value
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mpre[:-1] + [0.0] + [mpre[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')

                # set window title
                fig = plt.gcf()  # gcf - get current figure
                fig.canvas.set_window_title('AP ' + scale_name)

                # set plot title
                plt.title('scale: ' + text)
                # plt.suptitle('This is a somewhat long figure title', fontsize=16)

                # set axis titles
                plt.xlabel('Recall')
                plt.ylabel('Precision')

                # optional - set axes
                axes = plt.gca()  # gca - get current axes
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
                # Alternative option -> wait for button to be pressed
                # while not plt.waitforbuttonpress(): pass # wait for key display
                # Alternative option -> normal display
                # plt.show()

                # save the plot
                fig.savefig(os.path.join(plot_result_path, scale_name + ".png"))
                # plt.cla()  # clear axes for next plot


            if not args.quiet:
                print(text)


        # results_file.write("\n-----mAP of all classes-----\n")
        mAP = sum_AP / 3
        text = "mAP = {0:.2f}%".format(mAP*100)
        # results_file.write(text + "\n")
        print(text)

    return count_true_positives


"""3. making results part"""

gt_counter_per_scale= scale_gt(gt_json_path, TEMP_FILES_PATH, df)
det_counter_per_classes = dr_json(dr_json_path, TEMP_FILES_PATH,class_dict)


gt_classes = sorted(list(class_dict.values()))
gt_scale = sorted(list(gt_counter_per_scale.keys()))
n_classes = len(gt_classes)
dr_classes = list(det_counter_per_classes.keys())


count_true_positives = calculate_ap(TEMP_FILES_PATH, result_path, gt_classes, gt_scale, args,
                                    gt_counter_per_scale)



'''Write num of gt object per classes to results.txt'''
with open(result_path + "/results.txt", 'a') as results_file:
    results_file.write("\n----- Number of gt objects per scale-----\n")
    for scale_name in sorted(gt_counter_per_scale):
        results_file.write(scale_name + ":" + str(gt_counter_per_scale[scale_name])+"\n")




'''Finish counting tp'''
for class_name in dr_classes:
    if class_name not in gt_classes:
        for scale_name in range(len(gt_scale)):
            count_true_positives[scale_name] = 0


# with open(result_path + "/results.txt", 'a') as results_file:
#     results_file.write("\n----- Number of detected objects per scale-----\n")
#     for scale_name in range(len(gt_scale)):
#         text = scale_name + ": "
#         text += " (tp:" + str(count_true_positives[scale_name]) + ""
#         text += ", fp:" + str(count_false_positives[scale_name]) + ")\n"
#         results_file.write(text)


shutil.rmtree(TEMP_FILES_PATH)
