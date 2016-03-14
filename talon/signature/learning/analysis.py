# -*- coding: utf-8 -*-

"""
To analysis the prediction results.
"""

import os
import numpy as np
from shutil import copyfile
from evaluate import RESULT_DELIMITER, BODY_SUFFIX


def build_diff_filename(msg_filename):
    """By the message filename gives expected sender's filename."""
    return msg_filename[:-len(BODY_SUFFIX)] + '_diff'

def collect_emails(emails,src,dest,method,classes):
    for e in emails:
                diff = build_diff_filename(e)+ '_'+ method
                if os.path.isfile(src + '/' + diff):
                    copyfile(src+'/'+diff, dest+'/'+classes+'/'+diff)
                    copyfile(src+'/'+e, dest+'/'+classes+'/'+e)
                elif os.path.isfile(src + '2/' + diff):
                    copyfile(src+'2/'+diff, dest+'/'+classes+'/'+diff)
                    copyfile(src+'2/'+e, dest+'/'+classes+'/'+e)
                else:
                    print(diff)

def statistics(filename,src,dest,method):
    dest = dest+method
    if os.path.isfile(filename+'.data'):
        with open(filename+'.data','r') as f:
            content = f.read().splitlines()
            content = map(lambda x: x.split(RESULT_DELIMITER),content)
            matrix = np.array(content)
            ratio = matrix[:,0]
            email = matrix[:,1]

            index_0 = [i for i in range(len(ratio)) if ratio[i] == '0']
            ratio_0 = ratio[index_0]
            email_0 = email[index_0]

            index_P = [i for i in range(len(ratio)) if ratio[i] != '0' and ratio[i] != '-1']
            ratio_P = ratio[index_P]
            email_P = email[index_P]

            index_N1 = [i for i in range(len(ratio)) if ratio[i] == '-1']
            ratio_N1 = ratio[index_N1]
            email_N1 = email[index_N1]

            index_N2 = [i for i in range(len(ratio)) if ratio[i] == '-2']
            ratio_N2 = ratio[index_N2]
            email_N2 = email[index_N2]

            precision = float(len(email_P))/(len(email_P) + len(email_N2))
            recall = float(len(email_P))/(len(email_P) + len(email_N1))
            f_score = 2*(precision*recall)/(precision+recall)

            with open(filename+'_statics','w') as stat:
                stat.write('**** Summery ****\n0: '+str(len(email_0))+'\n-1: '+str(len(email_N1))+'\n-2: '+str(len(email_N2))+'\n>0: '+str(len(email_P)))
                stat.write('\nPrecision: '+str(precision)+'\tRecall: '+str(recall)+'\tF1-Score: '+str(f_score))
                stat.write('\n**** Details ****\n0: \n'+str(email_0)+'\n\n-1: \n'+str(email_N1)+'\n\n-2: \n'+str(email_N2)+'\n\n>0: \n'+str(email_P))

            collect_emails(email_N1,src,dest,method,'N1')
            collect_emails(email_N2,src,dest,method,'N2')
            collect_emails(email_P,src,dest,method,'P')
            collect_emails(email_0,src,dest,method,'0')


            print ratio_N2

