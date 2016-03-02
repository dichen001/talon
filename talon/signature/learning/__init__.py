"""
This file is to run the build_extraction_dataset()
So that we can build our feature data metrix (saved in 'dataset_filename' below)
from the manually marked emails (should be saved in 'folder')

Also, the marked file should be in the format as the files in folder.

FYI, refer to /talon/signature/learning/dataset.py
"""
import dataset as d
import os
from test import ROOT_DIR

base_dir = ROOT_DIR + '/tests/fixtures/signature'
folder = base_dir + '/emails/P'
# change relatively to where you store your marked training emails.
dataset_filename = base_dir + '/tmp/extraction.data'

#folder = '/Users/Jack/Dropbox/Github/talon/tests/fixtures/signature/emails/P'
#dataset_filename = '/Users/Jack/Dropbox/Github/talon/tests/fixtures/signature/tmp/extraction.data'
d.build_extraction_dataset(folder, dataset_filename, 1)