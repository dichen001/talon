"""
This file is to run the build_extraction_dataset()
So that we can build our feature data metrix (saved in 'dataset_filename' below)
from the manually marked emails (should be saved in 'folder')

Also, the marked file should be in the format as the files in folder.

FYI, refer to /talon/signature/learning/dataset.py
"""
import os
import talon
from pathlib import Path
Path('C:\Program Files').parent
import dataset as d
import evaluate as e
# from test import ROOT_DIR
DIR = os.path.abspath(os.path.dirname(__file__))
#ROOT_DIR = os.path.join(DIR,'../../..')
ROOT_DIR = '/'.join(DIR.split('/')[:-3])
base_dir = ROOT_DIR + '/tests/fixtures/signature'
train_folder = base_dir + '/emails/train'
test_folder = base_dir + '/emails/test2'
# change relatively to where you store your marked training emails.
dataset_filename = base_dir + '/tmp/extraction.data'
performance_filename = base_dir + '/tmp/performance2'

#folder = '/Users/Jack/Dropbox/Github/talon/tests/fixtures/signature/emails/P'
#dataset_filename = '/Users/Jack/Dropbox/Github/talon/tests/fixtures/signature/tmp/extraction.data'
d.build_extraction_dataset(train_folder, dataset_filename, 1)

talon.init()
e.predict(test_folder,performance_filename)