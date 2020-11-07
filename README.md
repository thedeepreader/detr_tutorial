# Training DETR on Custom Dataset
Tutorial Video [link](https://www.youtube.com/watch?v=RkhXoj_Vvr4&lc=UgwHlStd7pa4KMszFQx4AaABAg&ab_channel=DeepReader)  
Code based on [End-to-end Object Detection with Transformer](https://github.com/facebookresearch/detr)

## Step 1.
Download wider face dataset [link](http://shuoyang1213.me/WIDERFACE/) and unzip the files in dataset folder

## Step 2.
Move to the dataset folder, and convert downloaded wider face dataset into COCO format
```
$ python face_to_coco.py
```

## Step 3.
Move to detr folder and run main.py
```
$ python -m torch.distributed.launch --nproc_per_node=[num_of_gpus] --use_env main.py --data_path ../dataset/
```

## Step 4.
Run test.py
```
$ python test.py --data_path ../dataset/WIDER_test/images/ --resume [path_to_checkpoint.pth]
```
