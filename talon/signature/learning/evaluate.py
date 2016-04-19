# -*- coding: utf-8 -*-

"""
To evaluate the prediction results.
"""

import os
import csv
import email
import timeit
import difflib
import regex as re
import numpy as np
from shutil import copyfile
from random import randint,random,shuffle

import talon

from talon import quotations

from talon.signature.learning.featurespace import build_pattern, features
from talon.signature.bruteforce import extract_signature
from talon.signature.learning.classifier import train, init
from talon.signature import extraction
from talon.utils import get_delimiter
from talon.signature.constants import SIGNATURE_MAX_LINES



RESULT_DELIMITER = ' -%%%%- '

SENDER_SUFFIX = '_sender'
BODY_SUFFIX = '_body'
RESULT_SUFFIX = '_results'
PREDICT_SUFFIX = '_predicts'
DIFF_SUFFIX = '_diff'
SIG_SUFFIX = '_sig'
DETAILS_SUFFIX = '_details'
ORIGIN_SUFFIX = '_origin'

SIGNATURE_ANNOTATION = '#sig#'
REPLY_ANNOTATION = '#reply#'
NAME_ANNOTATION = '#name#'
TITLE_ANNOTATION = '#title#'
COMPANY_ANNOTATION = '#comp#'
ADDR_ANNOTATION = '#addr#'
NUM_ANNOTATION = '#num#'
WORK_ANNOTATION = '#work#'
FAX_ANNOTATION = '#fax#'
EMAIL_ANNOTATION = '#email#'
URL_ANNOTATION = '#url#'
SLOGAN_ANNOTATION = '#slogan#'
QUOTE_ANNOTATION = '#quote#'

ANNOTATIONS = [SIGNATURE_ANNOTATION, REPLY_ANNOTATION]



def is_sender_filename(filename):
    """Checks if the file could contain message sender's name."""
    return filename.endswith(SENDER_SUFFIX)


def build_filename(msg_filename, SUFFIX):
    """By the message filename gives expected sender's filename."""
    return msg_filename[:-len(BODY_SUFFIX)] + SUFFIX

def build_sender_filename(msg_filename):
    """By the message filename gives expected sender's filename."""
    return msg_filename[:-len(BODY_SUFFIX)] + SENDER_SUFFIX

def build_result_filename(msg_filename):
    """By the message filename gives expected sender's filename."""
    return msg_filename[:-len(BODY_SUFFIX)] + RESULT_SUFFIX

def build_predict_filename(msg_filename):
    """By the message filename gives expected sender's filename."""
    return msg_filename[:-len(BODY_SUFFIX)] + PREDICT_SUFFIX

def build_diff_filename(msg_filename):
    """By the message filename gives expected sender's filename."""
    return msg_filename[:-len(BODY_SUFFIX)] + DIFF_SUFFIX

def parse_msg_sender(filename, sender_known=True):
    """Given a filename returns the sender and the message.
    Here the message is assumed to be a whole MIME message or just
    message body.
    >>> sender, msg = parse_msg_sender('msg.eml')
    >>> sender, msg = parse_msg_sender('msg_body')
    If you don't want to consider the sender's name in your classification
    algorithm:
    # >>> parse_msg_sender(filename, False)
    """
    sender, msg = None, None
    if os.path.isfile(filename) and not is_sender_filename(filename):
        with open(filename) as f:
            msg = f.read()
            sender = u''
            if sender_known:
                sender_filename = build_sender_filename(filename)
                if os.path.exists(sender_filename):
                    with open(sender_filename) as sender_file:
                        sender = sender_file.read().strip()
                else:
                    # if sender isn't found then the next line fails
                    # and it is ok
                    lines = msg.splitlines()
                    for line in lines:
                        match = re.match('From:(.*)', line)
                        if match:
                            sender = match.group(1)
                            break
    return (sender, msg)


def predict(repetition, base_dir, emails, performance_filename,
                             sender_known=True):
    """evaluation of prediction.
    """
    r1= performance_filename+ repetition+'_brute'
    if os.path.exists(r1):
        os.remove(r1)
    r2= performance_filename+repetition+'_ml'
    if os.path.exists(r2):
        os.remove(r2)
    with open(r1, 'a') as total_results_b:
        with open(r2, 'a') as total_results_ml:
            # for filename in os.listdir(folder):
            for email in emails:

                filename = base_dir + '/emails/total/' + email
                sender, msg = parse_msg_sender(filename, sender_known)
                if not sender or not msg:
                    print 'Empty Sender: ' + filename
                    continue
                unmarked_msg = remove_marks(msg)
                text, brute_sig = extract_signature(unmarked_msg)
                if brute_sig is None:
                    brute_sig = ''

                text, ml_sig = talon.signature.extract(unmarked_msg, sender=sender)
                if ml_sig is None:
                    ml_sig = ''

                result_filename = build_result_filename(filename)
                with open(result_filename, 'r') as rf:
                    true = rf.read()
                    true = '\n'.join(true.splitlines()[::-1])

                predict_filename = build_predict_filename(filename) #filename[:-len('_body')] + PREDICT_SUFFIX
                with open(predict_filename+'_brute', 'w') as pf:
                    pf.write(brute_sig)
                with open(predict_filename+'_ml', 'w') as pf:
                    pf.write(ml_sig)

                diff_brute = difflib.SequenceMatcher(None,true,brute_sig)
                diff_ml = difflib.SequenceMatcher(None,true,ml_sig)
                '''
                0: not marked, not predicted
                >0: marked, predicted
                -2: not marked, predicted
                -1: marked, not predicted
                '''
                if true == '':
                    if diff_brute.ratio() > 0:
                        label_brute = 0
                    else:
                        label_brute = -2
                    if diff_ml.ratio() > 0:
                        label_ml = 0
                    else:
                        label_ml = -2
                else:
                    if diff_brute.ratio() > 0:
                        label_brute = diff_brute.ratio()
                    else:
                        label_brute = -1
                    if diff_ml.ratio() > 0:
                        label_ml = diff_ml.ratio()
                    else:
                        label_ml = -1

                diff_results1 = ''
                diff_results2 = ''
                if diff_brute.ratio() != 1.0:
                    diff_results1 = email+':\n##True:\n'+true+'\n##Brute:\n'+brute_sig
                if diff_ml.ratio() != 1.0:
                    diff_results2 = email+':\n##True:\n'+true+'\n##ML:\n'+ml_sig

                diff_filename = build_diff_filename(filename)
                with open(diff_filename+'_brute', 'w') as df1:
                    df1.write(diff_results1)
                with open(diff_filename+'_ml', 'w') as df2:
                    df2.write(diff_results2)

                total_results_b.write(str(label_brute) + ' -%%%%- ' + email+'\n')
                total_results_ml.write(str(label_ml) + ' -%%%%- ' + email+'\n')
    return r1,r2

# detailed version. to extract more features from Signature Block.
def predict_v2(repetition, base_dir, emails, performance_filename,
                             sender_known=True):
    """evaluation of prediction.
    """
    dir = base_dir + '/emails/process/'
    r1= performance_filename+ repetition+'_brute'
    if os.path.exists(r1):
        os.remove(r1)
    r2= performance_filename+repetition+'_ml'
    if os.path.exists(r2):
        os.remove(r2)
    # with open(r1, 'a') as total_results_b:
    with open(r2, 'a') as total_results_ml:
        # for filename in os.listdir(folder):
        for email in emails:

            filename = dir + email
            sender_f = build_filename(filename,SENDER_SUFFIX)
            origin_f = build_filename(filename, ORIGIN_SUFFIX)
            details_f = build_filename(filename, DETAILS_SUFFIX)
            signature_f = build_filename(filename, SIG_SUFFIX)

            sender = open(sender_f,'r').read()
            msg = open(origin_f,'r').read()
            details = open(details_f,'r').read()
            signature = open(signature_f,'r').read()

            if not sender or not msg:
                print 'Empty Sender: ' + filename
                continue

            # text, brute_sig = extract_signature(msg)
            # if brute_sig is None:
            #     brute_sig = ''

            text, ml_sig = talon.signature.extract(msg, sender=sender)
            if ml_sig is None:
                ml_sig = ''
            else:
                ml_details = predict_details(msg,sender)

            predict_filename = build_predict_filename(filename) #filename[:-len('_body')] + PREDICT_SUFFIX
            # with open(predict_filename+'_brute', 'w') as pf:
            #     pf.write(brute_sig)
            with open(predict_filename+'_ml', 'w') as pf:
                pf.write(ml_sig)

            # diff_brute = difflib.SequenceMatcher(None,signature,brute_sig)
            diff_ml = difflib.SequenceMatcher(None,signature,ml_sig)
            '''
            0: not marked, not predicted
            >0: marked, predicted
            -2: not marked, predicted
            -1: marked, not predicted
            '''
            if signature == '':
                # if diff_brute.ratio() > 0:
                #     label_brute = 0
                # else:
                #     label_brute = -2
                if diff_ml.ratio() > 0:
                    label_ml = 0
                else:
                    label_ml = -2
            else:
                # if diff_brute.ratio() > 0:
                #     label_brute = diff_brute.ratio()
                # else:
                #     label_brute = -1
                if diff_ml.ratio() > 0:
                    label_ml = diff_ml.ratio()
                else:
                    label_ml = -1

            # diff_sig1 = ''
            diff_sig2 = ''
            # if diff_brute.ratio() != 1.0:
            #     diff_sig1 = email+':\n##True:\n'+signature+'\n##Brute:\n'+brute_sig
            if diff_ml.ratio() != 1.0:
                diff_sig2 = email+':\n##True:\n'+signature+'\n##ML:\n'+ml_sig

            diff_filename = build_diff_filename(filename)
            # with open(diff_filename+'_brute', 'w') as df1:
            #     df1.write(diff_sig1)
            with open(diff_filename+'_ml', 'w') as df2:
                df2.write(diff_sig2)

            # total_results_b.write(str(label_brute) + ' -%%%%- ' + email+'\n')
            total_results_ml.write(str(label_ml) + ' -%%%%- ' + email+'\n')
    # return r1,r2
    return r2

def predict_details(sig,sender):
    dict = {}


def remove_marks(msg):
    lines = msg.splitlines()
    for i in xrange(1, min(11,len(lines)) + 1):
        line = lines[-i]
        if line[:len(SIGNATURE_ANNOTATION)] == \
                SIGNATURE_ANNOTATION:
            lines[-i] = line[len(SIGNATURE_ANNOTATION):]

        elif line[:len(REPLY_ANNOTATION)] == REPLY_ANNOTATION:
            lines[-i] = line[len(REPLY_ANNOTATION):]
    demarked_msg = '\n'.join(lines)
    return  demarked_msg

# to remove header details and the forward message from the emails, only return the reply body.
def process(msg,filename,sender):
    if not filename.endswith('.'):
        return None
    content = email.message_from_string(msg)
    body = []
    if content.is_multipart():
        for payload in content.get_payload():
            body.append(payload.get_payload())
    else:
        body.append(content.get_payload())
    if body is None: # discard mail without body
        print filename + ': body is None!'
    reply = quotations.extract_from(body[0], 'text/plain')
    #print filename + ":\n" + reply
    if re.search('\d+\.$',filename) is not None:
        reply_filename = re.sub('\d+\.$', change_name_body, filename)
        f = open(reply_filename, 'w')
        f.write(reply.strip())
        f.close()
        sender_filename = re.sub('\d+\.$', change_name_sender, filename)
        f = open(sender_filename, 'w')
        f.write(sender.strip())
        f.close()
        print filename
    return reply

def change_name_body(matchobj):
    if matchobj.group(0)[-1]=='.':
        return matchobj.group(0)[:-1] + '_body'

def change_name_sender(matchobj):
    if matchobj.group(0)[-1]=='.':
        return matchobj.group(0)[:-1] + '_sender'

def change_name_diff(matchobj):
    if matchobj.group(0)[-1]=='.':
        return matchobj.group(0)[:-1] + '_diff'

def get_all_emails(path):
    f = []
    for dirpath, dirnames, filenames in os.walk(path):
        for i in range(len(filenames)):
            filename = filenames[i]
            if str(filename).endswith('_body'):
                f.append(filename)
    return f


"Split data into training and testing"
def split_data(data,repetition,r):
    total = len(data)
    bin = total/repetition
    start = r*bin
    end = (r+1)*bin
    test = data[start:end]
    training = list(set(data)-set(test))
    return training,test

def preprocess(emails, folder, dest_folder, csv_results):
    """
    used to preprocess the marked file(xx_body), to generate the original one(xx_origin),
    the signature part(_sig), and the details infromation(xx_detail).
    :param folder:
    :return:
    """
    with open(csv_results,'w') as csvfile:
        fieldnames = ['filename', 'sender', 'content', 'has_sig', 'sig', 'name', 'title', 'company', 'address', 'number', 'work_number', 'fax', 'email', 'url', 'slogan', 'quote']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for email in emails:
            filename = folder + email
            sender, msg = parse_msg_sender(filename, sender_known=True)
            if not sender or not msg:
                print 'Empty: ' + filename
                continue
            delim = get_delimiter(msg)
            lines = msg.split(delim)

            sig = []
            dict = {}
            label = -1
            for i in xrange(1, min(SIGNATURE_MAX_LINES,len(lines)) + 1):
                line = lines[-i]
                if line[:len(SIGNATURE_ANNOTATION)] == SIGNATURE_ANNOTATION:
                    label = 1
                    line = line[len(SIGNATURE_ANNOTATION):]
                    dict,line = find_details(dict,line)
                    sig.append(line)
                    lines[-i] = line
            origin = build_filename(filename, ORIGIN_SUFFIX)
            details = build_filename(filename, DETAILS_SUFFIX)
            signature = build_filename(filename, SIG_SUFFIX)

            writer.writerow({'filename': email,
                             'sender': sender,
                             'content': msg,
                             'has_sig': label,
                             'sig': delim.join(sig[::-1]),
                             'name': dict.get('name'),
                             'title': dict.get('title'),
                             'company':dict.get('company'),
                             'address': dict.get('address'),
                             'number': dict.get('num'),
                             'work_number': dict.get('work_num'),
                             'fax': dict.get('fax'),
                             'email': dict.get('email'),
                             'url': dict.get('url'),
                             'slogan': dict.get('slogan'),
                             'quote': dict.get('quote')})

            # if label == 1:
            #     origin_f = open(origin, 'w')
            #     o = delim.join(lines)
            #     origin_f.write(o)
            #     # for i in lines:
            #     #     origin_f.writelines(i)
            #     origin_f.close()
            #
            #     signature_f = open(signature, 'w')
            #     s = delim.join(sig[::-1])
            #     signature_f.write(s)
            #     # for i in sig:
            #     #     signature_f.writelines(i)
            #     signature_f.close()
            #
            #     dict_f = open(details+".csv",'w')
            #     w = csv.writer(dict_f)
            #     for key, val in dict.items():
            #         w.writerow([key, val])
            #     dict_f.close()

def find_details(dict,line):
    if line[:len(NAME_ANNOTATION)] == NAME_ANNOTATION:
        line = line[len(NAME_ANNOTATION):]
        if dict.get('name'):
            dict['name'] = dict['name'] + ' |&*&| ' + line
        else:
            dict['name'] = line
    elif line[:len(TITLE_ANNOTATION)] == TITLE_ANNOTATION:
        line = line[len(TITLE_ANNOTATION):]
        if dict.get('title'):
            dict['title'] = dict['title'] + ' |&*&| ' + line
        else:
            dict['title'] = line
    elif line[:len(COMPANY_ANNOTATION)] == COMPANY_ANNOTATION:
        line = line[len(COMPANY_ANNOTATION):]
        if dict.get('company'):
            dict['company'] = dict['company'] + ' |&*&| ' + line
        else:
            dict['company'] = line
    elif line[:len(ADDR_ANNOTATION)] == ADDR_ANNOTATION:
        line = line[len(ADDR_ANNOTATION):]
        if dict.get('address'):
            dict['address'] = dict['address'] + ' |&*&| ' + line
        else:
            dict['address'] = line
    elif line[:len(NUM_ANNOTATION)] == NUM_ANNOTATION:
        line = line[len(NUM_ANNOTATION):]
        if dict.get('num'):
            dict['num'] = dict['num'] + ' |&*&| ' + line
        else:
            dict['num'] = line
    elif line[:len(WORK_ANNOTATION)] == WORK_ANNOTATION:
        line = line[len(WORK_ANNOTATION):]
        if dict.get('work_num'):
            dict['work_num'] = dict['work_num'] + ' |&*&| ' + line
        else:
            dict['work_num'] = line
    elif line[:len(FAX_ANNOTATION)] == FAX_ANNOTATION:
        line = line[len(FAX_ANNOTATION):]
        if dict.get('fax'):
            dict['fax'] = dict['fax'] + ' |&*&| ' + line
        else:
            dict['fax'] = line
    elif line[:len(EMAIL_ANNOTATION)] == EMAIL_ANNOTATION:
        line = line[len(EMAIL_ANNOTATION):]
        if dict.get('email'):
            dict['email'] = dict['email'] + ' |&*&| ' + line
        else:
            dict['email'] = line
    elif line[:len(URL_ANNOTATION)] == URL_ANNOTATION:
        line = line[len(URL_ANNOTATION):]
        if dict.get('url'):
            dict['url'] = dict['url'] + ' |&*&| ' + line
        else:
            dict['url'] = line
    elif line[:len(SLOGAN_ANNOTATION)] == SLOGAN_ANNOTATION:
        line = line[len(SLOGAN_ANNOTATION):]
        if dict.get('slogan'):
            dict['slogan'] = dict['slogan'] + ' |&*&| ' + line
        else:
            dict['slogan'] = line
    elif line[:len(QUOTE_ANNOTATION)] == QUOTE_ANNOTATION:
        line = line[len(QUOTE_ANNOTATION):]
        if dict.get('quote'):
            dict['quote'] = dict['quote'] + ' |&*&| ' + line
        else:
            dict['slogan'] = line
    return dict,line

def build_extraction_dataset(repetition, source_folder, emails, dataset_filename, sender_known=True):
    """Builds signature extraction dataset using emails in the `folder`
    .
    The emails in the `folder` should be annotated i.e. signature lines
    should be marked with `#sig#`.
    """
    global EXTRACTOR_DATA
    dataset_filename = dataset_filename+repetition
    if os.path.exists(dataset_filename):
        os.remove(dataset_filename)
    with open(dataset_filename, 'a') as dataset:
        for email in emails:
            filename = source_folder + email
            sender, msg = parse_msg_sender(filename, sender_known)
            if not sender or not msg:
                #print 'Empty: ' + filename
                continue

            ### Use 2 lines below to save the marked signature part into '*_result' file.
            ##
            result_filename = build_result_filename(filename)
            if os.path.exists(result_filename):
                os.remove(result_filename)
            with open(result_filename, 'a') as result:
            ## indent below after comment is taken off
                lines = msg.splitlines()
                for i in xrange(1, min(SIGNATURE_MAX_LINES,len(lines)) + 1):
                    line = lines[-i]
                    label = -1
                    if line[:len(SIGNATURE_ANNOTATION)] == \
                            SIGNATURE_ANNOTATION:
                        label = 1
                        line = line[len(SIGNATURE_ANNOTATION):]
                        # ##
                        # result.write(line + '\n')
                        # ##
                    elif line[:len(REPLY_ANNOTATION)] == REPLY_ANNOTATION:
                        line = line[len(REPLY_ANNOTATION):]
                    X = build_pattern(line, features(sender))
                    X.append(label)
                    labeled_pattern = ','.join([str(e) for e in X])
                    dataset.write(labeled_pattern + '\n')
    return dataset_filename

def collect_emails(emails,src,dest,method,classes):
    for e in emails:
                diff = build_diff_filename(e)+ '_'+ method
                if os.path.isfile(src + diff):
                    copyfile(src+ diff, dest+'/'+classes+'/'+diff)
                    copyfile(src + e, dest+'/'+classes+'/'+e)
                elif os.path.isfile(src + '2/' + diff):
                    copyfile(src+'2/'+diff, dest+'/'+classes+'/'+diff)
                    copyfile(src+'2/'+e, dest+'/'+classes+'/'+e)
                else:
                    print 'Can not find diff file: ' + diff

def statistics(repetition,filename,src,dest,method):
    dest = dest+method
    if os.path.isfile(filename):
        with open(filename,'r') as f:
            content = f.read().splitlines()
            content = map(lambda x: x.split(RESULT_DELIMITER),content)
            matrix = np.array(content)
            ratio = matrix[:,0]
            email = matrix[:,1]

            index_0 = [i for i in range(len(ratio)) if ratio[i] == '0']
            ratio_0 = ratio[index_0]
            email_0 = email[index_0]

            index_1 = [i for i in range(len(ratio)) if float(ratio[i]) >= 0.5]
            ratio_1 = ratio[index_1]
            email_1 = email[index_1]


            index_P = [i for i in range(len(ratio)) if ratio[i] != '0' and ratio[i] != '-1']
            ratio_P = ratio[index_P]
            email_P = email[index_P]

            index_N1 = [i for i in range(len(ratio)) if ratio[i] == '-1']
            ratio_N1 = ratio[index_N1]
            email_N1 = email[index_N1]

            index_N2 = [i for i in range(len(ratio)) if ratio[i] == '-2']
            ratio_N2 = ratio[index_N2]
            email_N2 = email[index_N2]

            if len(email_P) == 0:
                precision = 0
                recall = 0
                f_score = 0
            else:
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

            #print ratio_N2
            return precision, recall, f_score

def run_test(base_dir):
    start = timeit.default_timer()
    emails_folder = base_dir + '/emails/body'
    source_folder = base_dir + '/emails/total/'
    stat_folder = base_dir + '/tmp/statistics/'
    dataset_filename = base_dir + '/tmp/trained_model/extraction_'
    performance_filename = base_dir + '/tmp/predictions/performance_'
    classifier_dir = base_dir + '/tmp/classifiers/'
    global EXTRACTOR

    iteration = 10
    repetition = 5
    emails = get_all_emails(emails_folder)

    brute_p = []
    brute_r = []
    brute_f = []
    ml_p = []
    ml_r = []
    ml_f = []

    for i in range(iteration):
        shuffle(emails)
        tmp_dataset_filename = dataset_filename + str(i) + '_'
        tmp_performance_filename = performance_filename + str(i) + '_'
        for r in range(repetition):
            training,testing = split_data(emails,repetition,r)
            #print len(test),len(train)
            extraction_filename = build_extraction_dataset(str(r), source_folder, training, tmp_dataset_filename)

            classifier_name = classifier_dir + str(i) + '_' + str(r)
            extraction.EXTRACTOR = train(init(), extraction_filename, classifier_name)
            #talon.init()

            brute_result, ml_result = predict(str(r), base_dir, testing, tmp_performance_filename)
            brute_precision, brute_recall, brute_f_score = statistics(str(r),brute_result,source_folder,stat_folder,'brute')
            ml_precision, ml_recall, ml_f_score = statistics(str(r),ml_result,source_folder,stat_folder,'ml')
            print 'i:\t p:\t r:\t f:\t'
            print str(i*iteration+r) + '\t' + str(round(ml_precision,4)) + '\t' + str(round(ml_recall,4)) + '\t' + str(round(ml_f_score,4))
            brute_p.append(brute_precision)
            brute_r.append(brute_recall)
            brute_f.append(brute_f_score)
            ml_p.append(ml_precision)
            ml_r.append(ml_recall)
            ml_f.append(ml_f_score)
    brute_p_iqr = np.subtract(*np.percentile(brute_p, [75, 25]))
    brute_r_iqr = np.subtract(*np.percentile(brute_r, [75, 25]))
    brute_f_iqr = np.subtract(*np.percentile(brute_f, [75, 25]))
    ml_p_iqr = np.subtract(*np.percentile(ml_p, [75, 25]))
    ml_r_iqr = np.subtract(*np.percentile(ml_r, [75, 25]))
    ml_f_iqr = np.subtract(*np.percentile(ml_f, [75, 25]))
    brute_p = map(lambda x: round(x,3),brute_p)
    brute_r = map(lambda x: round(x,3),brute_r)
    brute_f = map(lambda x: round(x,3),brute_f)
    ml_p = map(lambda x: round(x,3),ml_p)
    ml_r = map(lambda x: round(x,3),ml_r)
    ml_f = map(lambda x: round(x,3),ml_f)


    print brute_f
    print brute_f_iqr
    print ml_f
    print ml_f_iqr
    with open(base_dir + '/tmp/FINAL_RESULTS','w') as fr:
        fr.write('Brute-Force:\n')
        fr.write('p:\n'+str(brute_p)+'\n')
        fr.write('r:\n'+str(brute_r)+'\n')
        fr.write('f:\n'+str(brute_f)+'\n')
        fr.write('p_median = '+ str(round(np.median(brute_p),4))+'\n')
        fr.write('r_median = '+ str(round(np.median(brute_r),4))+'\n')
        fr.write('f_median = '+ str(round(np.median(brute_f),4))+'\n')
        fr.write('p_iqr =  '+str(round(brute_p_iqr,4))+'\n')
        fr.write('r_iqr: '+str(round(brute_r_iqr,4))+'\n')
        fr.write('f_iqr: '+str(round(brute_f_iqr,4))+'\n')
        fr.write('\n******************\n')
        fr.write('Machine Learning:\n')
        fr.write('p:\n'+str(ml_p)+'\n')
        fr.write('r:\n'+str(ml_r)+'\n')
        fr.write('f:\n'+str(ml_f)+'\n')
        fr.write('p_median = '+ str(round(np.median(ml_p),4))+'\n')
        fr.write('r_median = '+ str(round(np.median(ml_r),4))+'\n')
        fr.write('f_median = '+ str(round(np.median(ml_f),4))+'\n')
        fr.write('p_iqr: '+str(round(ml_p_iqr,4))+'\n')
        fr.write('r_iqr: '+str(round(ml_r_iqr,4))+'\n')
        fr.write('f_iqr: '+str(round(ml_f_iqr,4))+'\n')
        fr.write('\n******************\n')
        runtime = timeit.default_timer() - start
        fr.write('Iteration: ' +str(iteration) + ' Bin Repetition: ' + str(repetition)+' Total Runtime:'+str(runtime))
    print runtime


## this function is used to create csv files containing 'filename' 'sender' and 'content'
## so that the csv file can be upload to MT to generate template email for MT workers to process.
def get_content_sender(folder,dest_folder,csvfile):
    with open(csvfile,'w') as csvfile:
        fieldnames = ['filename', 'sender', 'content']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for filename in os.listdir(folder):
            if filename.startswith('.') or not filename.endswith('.'):
                continue
            #print filename
            email_name = filename
            filename = os.path.join(folder, filename)
            sender, msg = parse_msg_sender(filename, sender_known=1)
            if not sender or not msg:
                #print 'Empty: ' + filename
                continue
            # use 2 lines below to pre-process emails to get the body and sender file for later Email Extraction.
            dest_file = os.path.join(dest_folder, email_name)
            msg = process(msg,dest_file,sender)
            if len(email_name) > 25:
                continue
            writer.writerow({'filename': email_name, 'sender': sender, 'content': msg})
