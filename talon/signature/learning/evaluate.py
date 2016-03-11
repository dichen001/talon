# -*- coding: utf-8 -*-

"""The module's functions build datasets to train/assess classifiers.

For signature detection the input should be a folder with two directories
that contain emails with and without signatures.

For signature extraction the input should be a folder with annotated emails.
To indicate that a line is a signature line use #sig# at the start of the line.

A sender of an email could be specified in the same file as
the message body e.g. when .eml format is used or in a separate file.

In the letter case it is assumed that a body filename ends with the `_body`
suffix and the corresponding sender file has the same name except for the
suffix which should be `_sender`.
"""

import os
import email
import difflib
import regex as re

from talon.signature.bruteforce import extract_signature
import talon
from talon import signature


from talon import quotations
from talon.signature.constants import SIGNATURE_MAX_LINES
from talon.signature.learning.featurespace import build_pattern, features


SENDER_SUFFIX = '_sender'
BODY_SUFFIX = '_body'
RESULT_SUFFIX = '_results'
PREDICT_SUFFIX = '_predicts'
DIFF_SUFFIX = '_diff'

SIGNATURE_ANNOTATION = '#sig#'
REPLY_ANNOTATION = '#reply#'

ANNOTATIONS = [SIGNATURE_ANNOTATION, REPLY_ANNOTATION]


def is_sender_filename(filename):
    """Checks if the file could contain message sender's name."""
    return filename.endswith(SENDER_SUFFIX)


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


def predict(folder, performance_filename,
                             sender_known=True):
    """evaluation of prediction.
    """
    r1= performance_filename+'_brute.data'
    if os.path.exists(r1):
        os.remove(r1)
    r2= performance_filename+'_ml.data'
    if os.path.exists(r2):
        os.remove(r2)
    with open(r1, 'a') as total_results_b:
        with open(r2, 'a') as total_results_ml:
            for filename in os.listdir(folder):
                if re.search('_body$',filename) is None:
                    print 'not body: ' + filename
                    continue
                else:
                     tmp_f = filename
                #print filename
                filename = os.path.join(folder, filename)
                sender, msg = parse_msg_sender(filename, sender_known)
                if not sender or not msg:
                    #print 'Empty: ' + filename
                    continue
                unmarked_msg = remove_marks(msg)
                text, brute_sig = extract_signature(unmarked_msg)
                if brute_sig is None:
                    brute_sig = ''

                text, ml_sig = signature.extract(unmarked_msg, sender=sender)
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
                label_brute = -1                # -1: predict different from mark

                if diff_brute.ratio() > 0:
                    diff_results1 = ''
                    if true == '':
                        label_brute = 0         # 0: no sig. no predict.
                    else:
                        label_brute = round(diff_brute.ratio(),3)         # 1: has sig. right predict.
                if diff_brute.ratio() != 1.0:
                    diff_results1 = tmp_f+':\n##True:\n'+true+'\n##Brute:\n'+brute_sig
                diff_ml = difflib.SequenceMatcher(None,true,ml_sig)
                label_ml = -1
                if diff_ml.ratio() > 0 :
                    diff_results2 = ''
                    if true =='':
                        label_ml = 0
                    else:
                        label_ml = round(diff_ml.ratio(),3)
                if diff_ml.ratio() != 1.0:
                    diff_results2 = tmp_f+':\n##True:\n'+true+'\n##ML:\n'+ml_sig

                diff_filename = build_diff_filename(filename)
                with open(diff_filename+'_brute', 'w') as df1:
                    df1.write(diff_results1)
                with open(diff_filename+'_ml', 'w') as df2:
                    df2.write(diff_results2)

                total_results_b.write(str(label_brute) + ' -%%%%- ' + tmp_f+'\n')
                total_results_ml.write(str(label_ml) + ' -%%%%- ' + tmp_f+'\n')

def remove_marks(msg):
    lines = msg.splitlines()
    for i in xrange(1, min(SIGNATURE_MAX_LINES,
                           len(lines)) + 1):
        line = lines[-i]
        label = -1
        if line[:len(SIGNATURE_ANNOTATION)] == \
                SIGNATURE_ANNOTATION:

            lines[-i] = line[len(SIGNATURE_ANNOTATION):]
            # ##
            # result.write(line + '\n')
            # ##
        elif line[:len(REPLY_ANNOTATION)] == REPLY_ANNOTATION:
            lines[-i] = line[len(REPLY_ANNOTATION):]
    demarked_msg = '\n'.join(lines)
    return  demarked_msg

# to remove header details and the forward message from the emails, only return the reply body.
def process(msg,filename,sender):
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
        reply_filename = re.sub('\d+\.$', change_name_reply, filename)
        f = open(reply_filename, 'w')
        f.write(reply.strip())
        f.close()
        sender_filename = re.sub('\d+\.$', change_name_sender, filename)
        f = open(sender_filename, 'w')
        f.write(sender.strip())
        f.close()
        print filename
    return reply

def change_name_reply(matchobj):
    if matchobj.group(0)[-1]=='.':
        return matchobj.group(0)[:-1] + '_body'

def change_name_sender(matchobj):
    if matchobj.group(0)[-1]=='.':
        return matchobj.group(0)[:-1] + '_sender'

def change_name_diff(matchobj):
    if matchobj.group(0)[-1]=='.':
        return matchobj.group(0)[:-1] + '_diff'