import argparse
from general_run import train_general,valid_general
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='UNet', type=str, help='model')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--model_path', type=str, help='if test, import the model')
    args = parser.parse_args()
    args.model_name = 'FCNet'
    print(args.model_name)

    if args.test==False:
        train_general(args)
    else:
        valid_general(args)
