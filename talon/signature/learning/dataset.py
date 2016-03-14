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
import regex as re

from talon import quotations
from talon.signature.constants import SIGNATURE_MAX_LINES
from talon.signature.learning.featurespace import build_pattern, features


SENDER_SUFFIX = '_sender'
BODY_SUFFIX = '_body'
RESULT_SUFFIX = '_results'
PREDICT_SUFFIX = '_predicts'

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
    name = '/'.join(msg_filename.split('/')[:-1])
    return name[:-len(BODY_SUFFIX)] + PREDICT_SUFFIX

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


def build_detection_class(folder, dataset_filename,
                          label, sender_known=True):
    """Builds signature detection class.

    Signature detection dataset includes patterns for two classes:
    * class for positive patterns (goes with label 1)
    * class for negative patterns (goes with label -1)

    The patterns are build of emails from `folder` and appended to
    dataset file.

    >>> build_signature_detection_class('emails/P', 'train.data', 1)
    """
    with open(dataset_filename, 'a') as dataset:
        for filename in os.listdir(folder):
            filename = os.path.join(folder, filename)
            sender, msg = parse_msg_sender(filename, sender_known)
            if sender is None or msg is None:
                continue
            msg = re.sub('|'.join(ANNOTATIONS), '', msg)
            X = build_pattern(msg, features(sender))
            X.append(label)
            labeled_pattern = ','.join([str(e) for e in X])
            dataset.write(labeled_pattern + '\n')


def build_detection_dataset(folder, dataset_filename,
                            sender_known=True):
    """Builds signature detection dataset using emails from folder.

    folder should have the following structure:
    x-- folder
    |    x-- P
    |    |    | -- positive sample email 1
    |    |    | -- positive sample email 2
    |    |    | -- ...
    |    x-- N
    |    |    | -- negative sample email 1
    |    |    | -- negative sample email 2
    |    |    | -- ...

    If the dataset file already exist it is rewritten.
    """
    if os.path.exists(dataset_filename):
        os.remove(dataset_filename)
    build_detection_class(os.path.join(folder, u'P'),
                          dataset_filename, 1)
    build_detection_class(os.path.join(folder, u'N'),
                          dataset_filename, -1)


def build_extraction_dataset(folder, dataset_filename,
                             sender_known=True):
    """Builds signature extraction dataset using emails in the `folder`.

    The emails in the `folder` should be annotated i.e. signature lines
    should be marked with `#sig#`.
    """
    if os.path.exists(dataset_filename):
        os.remove(dataset_filename)
    with open(dataset_filename, 'a') as dataset:
        for filename in os.listdir(folder):
            #print filename
            filename = os.path.join(folder, filename)
            sender, msg = parse_msg_sender(filename, sender_known)
            if not sender or not msg:
                #print 'Empty: ' + filename
                continue
            ## use 2 lines below to pre-process emails to get the body and sender file for later Email Extraction.
            # msg = process(msg,filename,sender)
            # continue

            # ### Use 2 lines below to save the marked signature part into '*_result' file.
            # ##
            # result_filename = build_result_filename(filename)
            # if os.path.exists(result_filename):
            #     os.remove(result_filename)
            # with open(result_filename, 'a') as result:
            # ## indent below after comment is taken off
            lines = msg.splitlines()
            for i in xrange(1, min(SIGNATURE_MAX_LINES,
                                   len(lines)) + 1):
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