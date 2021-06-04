import torch as t
import warnings


class BraTS2020Config(object):
    model = 'BaseLineModel'

    is_train = True
    predict_path = './predict'
    predict_figure = './figure'
    description = ''

    train_path = '/sunjindong/dataset/MICCAI_BraTS2020_TrainingData'
    val_path = '/sunjindong/dataset/MICCAI_BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData' 
    random_width = 16
    image_box = [128, 128, 128] #[192, 192, 144]
    trans_box = [192, 192, 144]

    use_gpu = True
    use_gpu_num = 4
    use_random = False
    load_model = None

    loss_function = 'CrossEntropyLoss'
    batch_size = 8
    max_epoch = 100
    lr = 0.01
    lr_decay = 0.4

    def _parse(self, kwargs):
        """
        update config
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)

    # print('user config:')
    # for k, v in self.__class__.__dict__.items():
    # 	if not k.startswith('_'):
    # 		print(k, getattr(self, k))


class ModelConfig(object):
    hidden_size = 256 # 768 # 576 = 16 x 36
    transformer__mlp_dim = 1024 # 3072 # 1024
    transformer__num_heads = 16 # 12 # 16
    transformer__num_layers = 2 # 12 # 16
    transformer__dropout_rate = 0.1

    resnet__grid = (16, 16)
    resnet__num_layers = (3, 4, 9)
    resnet__width_factor = 1

    batch_size = 1
    use_gpu_num = 4

    decoder_channels = (256, 128, 64, 16)
    n_skip = 3
    skip_channels = [512, 256, 64, 16]

    n_classes = 4

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: TransConfig has not attribute %s" % k)
            setattr(self, k, v)


config = BraTS2020Config()
modelconfig = ModelConfig()
