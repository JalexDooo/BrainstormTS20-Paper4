# The Segmentation of Gliomas in BraTS 2020 Challenge

This repository is the work of "_The multimodal brain tumor MRI images segmentation based on ConvRes-Trans-UNet_" based on **pytorch** implementation. 
<!-- You could click the link to access the [paper](https://arxiv.org/pdf/1904.03355.pdf). The multimodal brain tumor dataset (BraTS 2018) could be acquired from [here](https://www.med.upenn.edu/sbia/brats2018.html). -->

## Dilated multi-fiber network


<!-- <div  align="center">  
 <img src="https://github.com/China-LiuXiaopeng/BraTS-DMFNet/blob/master/fig/Architecture.jpg"
     align=center/>
</div> -->

 <!-- <center>Architecture of 3D DMFNet</center> -->


## Requirements
- python 3.6
- pytorch 1.0 or later CUDA version
- nibabel
- SimpleITK
- matplotlib
- tqdm
- pandas


cd /home/sunjindong/TransformerForSegmentation && python main.py train --model='ResUp_Blank' --description='_again_nolr_' --lr=0.01 --use_gpu_num=4 --batch_size=8 --max_epoch=75 --use_random=True

### Training

Multiply gpus training with batch_size=8 is recommended. The total training time take less than 30 hours in gtxforce 2080Ti. Training like this:

```
python main.py train --model='ResUp_Blank' --description='_' --lr=0.01 --use_gpu_num=4 --batch_size=8 --max_epoch=75 --use_random=True
```

### Test

You could obtain the resutls as paper reported by running the following code:

```
cd /home/sunjindong/TransformerForSegmentation && python main.py test --model='ResUp_Blank' --description='_' --use_gpu_num=1 --batch_size=1 --load_model='ResUp_Blank_199_.pth' --is_train=False --use_random=False --predict_path='ResUp_Blank_again_nolr_199_val/'
```
Then make a submission to the online evaluation server.

## Citation

If you use our code or model in your work or find it is helpful, please cite the paper:
```
***Unknown***
```

## Acknowledge
Balabala

