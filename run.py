"""
This file is to run the build_extraction_dataset()
So that we can build our feature data metrix (saved in 'dataset_filename' below)
from the manually marked emails (should be saved in 'folder')
Also, the marked file should be in the format as the files in folder.
FYI, refer to /talon/signature/learning/dataset.py
"""
import os
import pickle
#import talon
from pathlib import Path
Path('C:\Program Files').parent
import talon.signature.learning.dataset as d
import talon.signature.learning.evaluate as e
#import analysis as a
# from test import ROOT_DIR
DIR = os.path.abspath(os.path.dirname(__file__))
#ROOT_DIR = os.path.join(DIR,'../../..')
base_dir = DIR + '/tests/fixtures/signature'
train_folder = base_dir + '/emails/train'
test_folder = base_dir + '/emails/test'
# change relatively to where you store your marked training emails.
dataset_filename = base_dir + '/tmp/extraction.data'
performance_filename = base_dir + '/tmp/performance'
result_filename = base_dir + '/tmp/merged_ml'
to_folder = base_dir + '/tmp/classified_emails/'

email_folder = base_dir + '/emails/body'
process_folder = base_dir + '/emails/process/'

csv_file = base_dir + '/emails/test.csv'

turk_folder = base_dir + '/emails/MTreviewd_Emails/'
processed_folder = base_dir + '/emails/MTreviewd_Emails_Processed/'
csv_results = processed_folder + 'results_summary.csv'

dest_folder = base_dir + '/emails/big_turk_processed'



# generage csv templetes for MT assignment
# e.get_content_sender(turk_folder,dest_folder,csv_file)

emails = e.get_all_emails(turk_folder)
e.preprocess(emails, turk_folder, processed_folder, csv_results)
# e.run_test(base_dir)

#d.build_extraction_dataset(train_folder, dataset_filename, 1)
# talon.init()
# e.predict(test_folder,performance_filename)
# a.statistics(result_filename,test_folder,to_folder,'brute')
