import numpy as np
import os.path
from skimage.io import imread, imsave
from utils import flow_utils

def main():
	flow_filename = '../flownet2_validation_examples/NBA2K19_2019.01.31_23.50.52_frame124678.pred.flo'
	save_dir = '../'
	flow_utils.visulize_flow_file(flow_filename, save_dir)

if __name__ == "__main__":
	main()