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
                    bi = [max(bb[0], bbgt[0]), min(bb[1], bbgt[1]), min(bb[0] + bb[2], bbgt[0] + bbgt[2]),
                          max(bb[1] + bb[3], bbgt[1] + bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # ua = compute overlap (IoU) = area of intersection/ area of union
                        ua = ((bb[2]+1) * (bb[3]+1) + (bbgt[2]+1) * (bbgt[3]+1)) - iw * ih
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
