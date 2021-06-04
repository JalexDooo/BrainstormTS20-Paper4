# Redesigned code version 3.0

# cd /home/sunjindong/BrainstormBraTS2020 && python main.py train --model='BaseLineModel' --use_gpu_num=4
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py train --model='BaseLineModel' --use_gpu_num=4 --description='+r0'


###################### paper experiments
# Baseline r0
# task 5442
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py train --model='BaseLineModel' --use_gpu_num=4 --lr_decay=1.0 --description='_r0_'

# task 5324 训练出错 model='_r0' line 9
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py train --model='BaseLineModel' --use_gpu_num=4 --lr_decay=1.0 --description='_+r0'

# task 5585
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py train --model='BaseLineModel' --use_gpu_num=4 --lr_decay=0.1 --description='_+r0_dec'

# task 
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py train --model='BaseLineModel' --use_gpu_num=4 --lr_decay=0.1 --description='_++r0_dec'


## test
# cd /home/sunjindong/BrainstormBraTS2020 && python main.py test --model='BaseLineModel' --use_gpu_num=1 --description='_+r0_dec' --load_model='BaseLineModel.pth' --is_train=False --use_random=False --predict_path='test/'


############### 修改
# /home/sunjindong/BrainstormBraTS2020Random2D

# BaseLine
cd /home/sunjindong/BrainstormBraTS2020Random2D && python main.py train --model='BaseLineModel' --use_gpu_num=2 --batch_size=8 --lr_decay=1.0 --description='_' --use_random=True
cd /home/sunjindong/BrainstormBraTS2020Random2D && python main.py train --model='BaseLineModel' --use_gpu_num=2 --batch_size=8 --lr=0.01 --lr_decay=0.9 --max_epoch=100 --description='_' --use_random=True

# Resup +随即强度增强
cd /home/sunjindong/BrainstormBraTS2020Random2D && python main.py train --model='ResUp' --use_gpu_num=4 --batch_size=8 --lr_decay=0.1 --max_epoch=100 --description='_' --use_random=True

# Resup -随即强度增强
cd /home/sunjindong/BrainstormBraTS2020Random2D && python main.py train --model='ResUp' --use_gpu_num=2 --batch_size=8 --lr_decay=0.95 --max_epoch=100 --description='_norandbias_' --use_random=True

# Resup -随即强度增强 nolr
cd /home/sunjindong/BrainstormBraTS2020Random2D && python main.py train --model='ResUp' --use_gpu_num=4 --batch_size=16 --lr=0.001 --lr_decay=1.0 --max_epoch=100 --description='_norandbias_nolrdecay_' --use_random=True



### test
cd /home/sunjindong/BrainstormBraTS2020Random2D && python main.py test --model='ResUp' --use_gpu_num=1 --batch_size=1 --description='_' --load_model='ResUp_199_.pth' --is_train=False --use_random=False --predict_path='__199_test/'

cd /home/sunjindong/BrainstormBraTS2020Random2D && python main.py test --model='ResUp' --use_gpu_num=1 --batch_size=1 --description='_norandbias_' --load_model='ResUp_99_.pth' --is_train=False --use_random=False --predict_path='_Resup_99_val/'

cd /home/sunjindong/BrainstormBraTS2020Random2D && python main.py test --model='ResUp' --use_gpu_num=1 --batch_size=1 --description='_norandbias_nolrdecay_' --load_model='ResUp_99_.pth' --is_train=False --use_random=False --predict_path='_resup_99_norandbiasnolrdecay_val/'




# aneu

# /home/aneu/BrainstormBraTS2020Random2D/BrainstormBraTS2020Random2D
# train
cd /home/aneu/BrainstormBraTS2020Random2D/BrainstormBraTS2020Random2D && python main.py train --model='ResUp_Blank' --use_gpu_num=2 --batch_size=8 --lr=0.001 --lr_decay=0.1 --max_epoch=100 --description='_' --use_random=True --train_path='/home/aneu/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

# test
cd /home/aneu/BrainstormBraTS2020Random2D/BrainstormBraTS2020Random2D && python main.py test --model='ResUp_Blank' --use_gpu_num=1 --batch_size=1 --description='_' --load_model='ResUp_Blank_99_.pth' --is_train=False --use_random=False --predict_path='_resup_blank_val/' --val_path='/home/aneu/dataset/MICCAI_BraTS2020_ValidationData'


# Transformer for brats20
# train
cd /home/sunjindong/TransformerForSegmentation && python main.py trans_train --model='TransUNet' --use_gpu_num=4

# test
cd /home/aneu/TransformerForSegmentation && python main.py trans_test --model='TransUNet' --use_gpu_num=1 --load_model='TransUNet_89_.pth' --is_train=False --predict_path='_transunet_89_val/' --val_path='/home/aneu/dataset/MICCAI_BraTS2020_ValidationData'


# TransResNet for brats20
cd /home/sunjindong/TransformerForSegmentation && python main.py trans_train --model='TransResNet' --use_gpu_num=4

# test
cd /home/aneu/TransformerForSegmentation && python main.py trans_test --model='TransResNet' --use_gpu_num=1 --load_model='TransResNet_49_.pth' --is_train=False --predict_path='_transresnet_49_val/' --val_path='/home/aneu/dataset/MICCAI_BraTS2020_ValidationData'

# TransResNet_3d
cd /home/sunjindong/TransformerForSegmentation && python main.py trans_train_3d --model='TransResNet_3d' --use_gpu_num=4 --batch_size=4 --max_epoch=500

cd /home/aneu/TransformerForSegmentation && python main.py trans_train_3d --model='TransResNet_3d' --use_gpu_num=4 --batch_size=4 --max_epoch=500 --train_path='/home/aneu/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

# test
cd /home/aneu/TransformerForSegmentation && python main.py trans_test_3d --model='TransResNet_3d' --use_gpu_num=1 --batch_size=1 --load_model='TransResNet_3d_84_.pth' --is_train=False --predict_path='_transresnet_pe_3d_84_val/' --val_path='/home/aneu/dataset/MICCAI_BraTS2020_ValidationData'

# 84: 3d, pe-embedding,trans-4


# train
cd /home/aneu/TransformerForSegmentation && python main.py train --model='ResNet_3d' --use_gpu_num=4 --batch_size=8 --max_epoch=130 --train_path='/home/aneu/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
cd /home/sunjindong/TransformerForSegmentation && python main.py train --model='ResNet_3d' --use_gpu_num=4 --batch_size=16 --max_epoch=130 --use_random=True
cd /home/sunjindong/TransformerForSegmentation && python main.py train --model='ResNet_3d' --description='_groupnorm_' --use_gpu_num=4 --batch_size=8 --max_epoch=130 --use_random=True
cd /home/sunjindong/TransformerForSegmentation && python main.py train --model='ResUp_Blank' --description='_again_nolr_' --lr=0.01 --use_gpu_num=4 --batch_size=8 --max_epoch=75 --use_random=True
cd /home/sunjindong/TransformerForSegmentation && python main.py train --model='ResUp_Blank' --description='_again_nolr_' --lr=0.0001 --use_gpu_num=4 --batch_size=8 --max_epoch=10 --use_random=True --load_model='ResUp_Blank_6_.pth'
cd /home/sunjindong/TransformerForSegmentation && python main.py train --model='BaseLineModel' --description='_nolr_' --lr=0.0001 --use_gpu_num=4 --batch_size=8 --max_epoch=50 --use_random=True --load_model='BaseLineModel_29_.pth'
cd /home/sunjindong/TransformerForSegmentation && python main.py train --model='ResUp_Blank' --description='_lr_schedular_' --lr=0.001 --lr_decay=0.1 --use_gpu_num=2 --batch_size=8 --max_epoch=100 --use_random=True
cd /home/sunjindong/TransformerForSegmentation && python main.py train --model='BaseLineModel' --description='_lr_schedular_' --lr=0.001 --lr_decay=0.1 --use_gpu_num=8 --batch_size=32 --max_epoch=200 --use_random=True

cd /home/aneu/TransformerForSegmentation && python main.py trans_train_3d --model='TransResNet_3d' --use_gpu_num=4 --batch_size=4 --max_epoch=800 --train_path='/home/aneu/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

# use_random = True
# task aneu 9337 原始main.py + 原始data.py + GroupNorm的ResNet。
# task aneu 9369 ... + BatchNorm的ResNet
# task sunjindong 9378  ... + DoubleConvLayer(Decoder) + BatchSize16


# test
cd /home/sunjindong/TransformerForSegmentation && python main.py test --model='ResNet_3d' --use_gpu_num=1 --batch_size=1 --load_model='ResNet_3d_83_.pth' --is_train=False --use_random=False --predict_path='ResNet_3d_83_val/'

cd /home/aneu/TransformerForSegmentation && python main.py test --model='ResUp_Blank' --description='_' --use_gpu_num=1 --batch_size=1 --load_model='ResUp_Blank_35_.pth' --is_train=False --use_random=False --predict_path='ResNet_3d_35_val/' --val_path='/home/aneu/dataset/MICCAI_BraTS2020_ValidationData'

cd /home/sunjindong/TransformerForSegmentation && python main.py test --model='BaseLineModel' --description='_nolr_' --use_gpu_num=1 --batch_size=1 --load_model='BaseLineModel_49_.pth' --is_train=False --use_random=False --predict_path='BaseLineModel_nolr_val/'

cd /home/sunjindong/TransformerForSegmentation && python main.py test --model='ResUp_Blank' --description='_again_nolr_' --use_gpu_num=1 --batch_size=1 --load_model='ResUp_Blank_9_.pth' --is_train=False --use_random=False --predict_path='ResUp_Blank_again_nolr_9_val/'

cd /home/sunjindong/TransformerForSegmentation && python main.py test --model='ResUp_Blank' --description='_lr_schedular_' --use_gpu_num=1 --batch_size=1 --load_model='ResUp_Blank_99_.pth' --is_train=False --use_random=False --predict_path='ResUp_Blank_lr_schedular_99_val/'
cd /home/sunjindong/TransformerForSegmentation && python main.py test --model='BaseLineModel' --description='_lr_schedular_' --use_gpu_num=1 --batch_size=1 --load_model='BaseLineModel_199_.pth' --is_train=False --use_random=False --predict_path='BaseLineModel_lr_schedular_199_val/'


cd /sunjindong/TransformerForSegmentation && python -u main.py sa_lut_train --model='SA_LuT_Nets' --description='_' --lr=0.1 --lr_decay=0.1 --use_gpu_num=4 --batch_size=8 --max_epoch=200 --use_random=True --random_width=128
cd /sunjindong/TransformerForSegmentation && python -u main.py sa_lut_train --model='BaseLineModel' --description='_' --lr=0.1 --lr_decay=0.1 --use_gpu_num=4 --batch_size=8 --max_epoch=200 --use_random=True --random_width=128
