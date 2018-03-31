import argparse

from tensorpack.network_models import anytime_network
from tensorpack.network_models.anytime_network import AnytimeMultiScaleDenseNet

import ann_app_utils 

"""
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = ann_app_utils.parser_add_app_arguments(parser)
    anytime_network.parser_add_msdensenet_arguments(parser)
    args = parser.parse_args()

    model_cls = AnytimeMultiScaleDenseNet
    
    ann_app_utils.cifar_svhn_train_or_test(args, model_cls)
