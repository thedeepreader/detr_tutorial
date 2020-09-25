import re
import os
import cv2
import json
import itertools
import numpy as np
from glob import glob
import scipy.io as sio
from pycocotools import mask as cocomask
from PIL import Image


MAX_N = 10

categories = [
    {
        "supercategory": "none",
        "name": "face",
        "id": 0
    }
]

phases = ["train", "val"]
for phase in phases:
    root_path = "WIDER_{}/images/".format(phase)
    gt_path = os.path.join("wider_face_split/wider_face_{}.mat".format(phase))
    json_file = "{}.json".format(phase)

    gt = sio.loadmat(gt_path)
    event_list = gt.get("event_list")
    file_list = gt.get("file_list")
    face_bbox_list = gt.get("face_bbx_list")

    res_file = {
        "categories": categories,
        "images": [],
        "annotations": []
    }

    annot_count = 0
    image_id = 0
    processed = 0
    for event_idx, path in enumerate(event_list):
        base_path = path[0][0]
        for file_idx, img_name in enumerate(file_list[event_idx][0]):
            file_path = img_name[0][0]
            face_bbox = face_bbox_list[event_idx][0][file_idx][0]
            num_boxes = face_bbox.shape[0]

            if num_boxes > MAX_N:
                continue

            img_path = os.path.join(root_path, base_path, file_path + ".jpg")
            filename = os.path.join(base_path, file_path + ".jpg")

            img = Image.open(img_path)
            img_w, img_h = img.size
            img_elem = {"file_name": filename,
                        "height": img_h,
                        "width": img_w,
                        "id": image_id}

            res_file["images"].append(img_elem)

            for i in range(num_boxes):
                xmin = int(face_bbox[i][0])
                ymin = int(face_bbox[i][1])
                xmax = int(face_bbox[i][2]) + xmin
                ymax = int(face_bbox[i][3]) + ymin
                w = xmax - xmin
                h = ymax - ymin
                area = w * h
                poly = [[xmin, ymin],
                        [xmax, ymin],
                        [xmax, ymax],
                        [xmin, ymax]]

                annot_elem = {
                    "id": annot_count,
                    "bbox": [
                        float(xmin),
                        float(ymin),
                        float(w),
                        float(h)
                    ],
                    "segmentation": list([poly]),
                    "image_id": image_id,
                    "ignore": 0,
                    "category_id": 0,
                    "iscrowd": 0,
                    "area": float(area)
                }

                res_file["annotations"].append(annot_elem)
                annot_count += 1

            image_id += 1

            processed += 1

    with open(json_file, "w") as f:
        json_str = json.dumps(res_file)
        f.write(json_str)

    print("Processed {} {} images...".format(processed, phase))
print("Done.")



