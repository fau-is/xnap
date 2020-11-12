import os
import argparse
import xnap.utils as utils


def load():
    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument('--task', default="nap")
    parser.add_argument('--data_set', default="helpdesk.csv")
    parser.add_argument('--data_dir', default="./data/")
    parser.add_argument('--model_dir', default="nap/models/")
    parser.add_argument('--result_dir', default="./results/")

    # parameters for explanation
    parser.add_argument('--explain', default=False, type=utils.str2bool)
    parser.add_argument('--rand_lower_bound', default=5, type=int)
    parser.add_argument('--rand_upper_bound', default=5, type=int)

    # parameters for deep neural network
    parser.add_argument('--dnn_num_epochs', default=100, type=int)
    parser.add_argument('--dnn_architecture', default=0, type=int)
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--dim', default=0, type=int)

    # parameters for validation
    parser.add_argument('--num_folds', default=10, type=int)
    parser.add_argument('--cross_validation', default=True, type=utils.str2bool)
    parser.add_argument('--split_rate_test', default=0.5, type=float)
    parser.add_argument('--batch_size_train', default=128, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)

    # parameters for gpu processing
    parser.add_argument('--gpu_ratio', default=0.2, type=float)
    parser.add_argument('--cpu_num', default=1, type=int)
    parser.add_argument('--gpu_device', default="0", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    return args
