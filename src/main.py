import option
import os, time, shutil
import utils_map, size_ap_v2
import json
# from collections import OrderedDict

start = time.time()
opt = option.options

gt_json_path = opt.gt_json_path
dr_json_path = opt.dr_json_path

# if there are no classes to ignore then replace None by empty list
if opt.ignore is None:
    opt.ignore = []

# make sure that the cwd() is the location of the python script (so that every path makes sense)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

results_files_path = "results"
if not os.path.exists(results_files_path):
    os.makedirs(results_files_path)
else:
    shutil.rmtree(results_files_path)
result_file_path = results_files_path + "/results" + ".txt"

plot_result_path = "plot_figures"
if opt.draw_plot:
    if opt.plot_save:
        if os.path.exists(plot_result_path):
            shutil.rmtree(plot_result_path)
            os.makedirs(plot_result_path)
        else:
            os.makedirs(plot_result_path)


final_result_path = "final_result"
if not os.path.exists(results_files_path):
    os.makedirs(results_files_path)
else:
    shutil.rmtree(results_files_path)
# result_file_path = final_result_path + "/results" + ".json"



class_dict = utils_map.make_gt_list(gt_json_path)
gt_counter_per_class, gt_counter_per_size, counter_images_per_size, gt \
    = utils_map.get_gt_match(gt_json_path, class_dict)


gt_classes = list(class_dict.values())
# sort classes alphabetically

gt_classes = sorted(gt_classes)
n_classes = len(gt_classes)

if opt.set_class_iou is not None:
    utils_map.check_format_class_iou(opt, gt_classes)
det_counter_per_classes, dr = utils_map.dr_json(dr_json_path, class_dict)
dr_classes = list(det_counter_per_classes.keys())
dr_sizes = ["small", "medium", "large"]


count_true_positives, mAP, ap_dictionary, precision_dict, recall_dict = \
    utils_map.calculate_ap(result_file_path, plot_result_path, gt_classes, opt, gt_counter_per_class, dr, gt)

size_count_true_positives = \
    size_ap_v2.calculate_ap(gt_classes, opt, dr, gt)

with open(result_file_path, 'a') as results_file:

    '''ap for classes'''
    results_file.write("\n# Number of gt objects per class\n")
    for class_name in sorted(gt_counter_per_class):
        results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

    for class_name in dr_classes:
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0

    results_file.write("\n# Number of detected objects per class\n")
    for class_name in sorted(gt_classes):
        try: n_det = det_counter_per_classes[class_name]
        except: n_det = 0  # If there is no gt class in dt, n_dt = 0
        text = class_name + ": " + str(n_det)
        text += " (tp:" + str(count_true_positives[class_name]) + ""
        text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
        results_file.write(text)

    '''ground truth & detection number for sizes'''
    results_file.write("\n# Number of gt objects per size\n")
    for class_name in gt_counter_per_size:
        results_file.write(class_name + ": " + str(gt_counter_per_size[class_name]) + "\n")

    results_file.write("\n# Number of detected objects per size\n")
    for class_name in dr_sizes:
        text = class_name + ": " + str(size_count_true_positives[class_name]) + "\n"
        results_file.write(text)

utils_map.print_configuration(result_file_path, opt)


finish = time.time()
print("time: ", finish - start)

# print(ap_dictionary)

"""---------------------------------------------------"""

data = {}
data["mAP"] = mAP
data["IoU"] = opt.iou_threshold
data["gt_path"] = gt_json_path
data["dr_path"] = dr_json_path
data["time"] = finish - start
data["average precision"] = ap_dictionary
positive_dict = {}
size_dict = {}

with open(final_result_path, 'w') as final_result_file:
    for class_name in sorted(gt_counter_per_class):
        try: n_det = det_counter_per_classes[class_name]
        except: n_det = 0
        if class_name not in det_counter_per_classes.keys():
            det_counter_per_classes[class_name] = 0
        positive_dict[class_name] = {"tp" : str(count_true_positives[class_name]),
                                     "fp" : str(n_det - count_true_positives[class_name]),
                                     "gt_count" : gt_counter_per_class[class_name],
                                     "prediction_count": det_counter_per_classes[class_name],
                                     "precision": precision_dict[class_name],
                                     "recall": recall_dict[class_name]}

    size_dict["ground_truth"] = {"small": gt_counter_per_size["small"],
                                 "medium": gt_counter_per_size["medium"],
                                 "large": gt_counter_per_size["large"]}
    size_dict["prediction"] = {"small": size_count_true_positives["small"],
                               "medium": size_count_true_positives["medium"],
                               "large": size_count_true_positives["large"]}

    data["count"] = positive_dict
    data["by scale"] = size_dict
    json.dump(data, final_result_file)


print(gt_counter_per_class)


# for class_name in sorted(gt_counter_per_class):
#     print(precision_dict[class_name])
