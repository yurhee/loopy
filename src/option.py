import easydict

gt_json_path = 'C:/Users/Medical-Information/PycharmProjects/project_metric/input/ground_truth/ground_truth.json'
dr_json_path = 'C:/Users/Medical-Information/PycharmProjects/project_metric/input/detection_results/detection_results.json'

options = easydict.EasyDict({
        "gt_json_path": gt_json_path,
        "dr_json_path": dr_json_path,

        "iou_threshold": 0.5,
        "size_threshold": 32,
        "confidence_threshold": 0.3,

        "n_interpolation": 11,
        "no_interpolation": False,  # just use pascal voc 2007

        "ignore": None,
        "set_class_iou": None,  # 특정 클래스에 대해 특정 iou 설정, ex: person 0.7
        "quiet": True,

        "see_pre_rec": False,

        "draw_plot": True,
        "plot_save": True
    })


'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''
