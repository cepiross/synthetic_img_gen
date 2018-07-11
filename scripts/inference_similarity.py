#!/usr/bin/env python2
"""
Compute cosine similarity from class-wise average precision vector

Author: jpc5731@cse.psu.edu (Jinhang Choi)

Usage: inference_similarity.py \
                <base_model_inference_results> \
                <target_data_inference_results_on_base_model>
"""
import os
import h5py
import math
import argparse
import numpy as np


def compute_ap(CLASS_ID, GROUND_TRUTH, relevant):
    """
    Compute class-wise top-1 average precision
    """
    ap = np.zeros_like(CLASS_ID).astype(float)
    ap_pos = np.zeros_like(CLASS_ID).astype(int)
    relcnt = np.zeros_like(CLASS_ID).astype(int)
    precision = np.zeros_like(relevant).astype(float)

    progress = int(len(CLASS_ID)*0.2)
    print 'In processing class',
    for idx, classid in np.ndenumerate(CLASS_ID):
        if idx[0] % progress == 0:
            print '.',
        pos = np.where(GROUND_TRUTH == classid)[0]
        if pos.size > 0:
            ap_pos[idx] = pos[-1]
            relcnt[idx] = np.count_nonzero(relevant[pos])

            if relcnt[idx] > 0:
                cnt = 0
                for i, rel_pos in np.ndenumerate(pos):
                    if relevant[rel_pos] == True:
                        cnt = cnt + 1
                        precision[rel_pos] = float(cnt)/(float(i[0])+1.)
                    else:
                        precision[rel_pos] = 0

                ap[idx] = np.sum(precision[pos])/float(relcnt[idx])
            else:
                ap[idx] = 0
        else:
            ap[idx] = 0
    print ''
    return ap

def classID_matching(arg1, arg2):
    GT1 = arg1[0]
    GT2 = arg2[0]

    # Caveat: if the model is not consistent with the target dataset
    # in terms of classe identities, you have to change the target class IDs
    # to fit on your model's IDs so that prediction is correct.
    # e.g.
    #  idx     p1 (model)     p2 (target)
    # -----------------------------------
    #   0  chips_barbaras  chips_barbaras
    #   1       chips_box       chips_box
    #   2   chips_doritos       chips_can
    #   3    chips_kettle   chips_doritos
    #   4      chips_lays    chips_kettle
    #   5         pop_box      chips_lays
    #   6        pop_flat      pop_bottle
    #   7                         pop_box
    #   8                        pop_flat
    # -----------------------------------
    # For comparison of p2 to p1 model,
    #   ground truth of p2 should be shrinken to p1, i.e.
    #   (0, 1, 2, 3, 4, 5, 6, 7, 8) -> (0, 1, -1, 2, 3, 4, -1, 5, 6)
    GT2[GT2 == 2] = -1
    pos = GT2 > 2
    GT2[pos] = GT2[pos] - 1
    GT2[GT2 == 5] = -1
    pos = GT2 > 5
    GT2[pos] = GT2[pos] - 1
    return

def print_similarity(result_db_first,
                 result_db_second):
    """
    Print inference comparison
    """
    if result_db_first is None or result_db_second is None:
        return
    db1 = h5py.File(result_db_first, 'r')
    db2 = h5py.File(result_db_second, 'r')

    if db1 is not None and db2 is not None:
        filename1 = os.path.splitext(os.path.basename(result_db_first))[0]
        filename2 = os.path.splitext(os.path.basename(result_db_second))[0]

        GROUND_TRUTH1 = [int(item.split('_')[1]) for item in db1['input_ids']]
        CLASS_ID1 = np.unique(GROUND_TRUTH1)
        res = db1['outputs'][db1['outputs'].keys()[0]]
        PRED_ID1 = np.array([np.argmax(item) for item in res])

        GROUND_TRUTH2 = np.array([int(item.split('_')[1]) for item in db2['input_ids']])
        CLASS_ID2 = np.unique(GROUND_TRUTH2)
        res = db2['outputs'][db2['outputs'].keys()[0]]
        PRED_ID2 = np.array([np.argmax(item) for item in res])
        ######### index regression #########
        classID_matching([GROUND_TRUTH1], [GROUND_TRUTH2])
        ####################################
        relevant1 = np.equal(GROUND_TRUTH1, PRED_ID1)
        relevant2 = np.equal(GROUND_TRUTH2, PRED_ID2)
        ap1 = compute_ap(CLASS_ID1, GROUND_TRUTH1, relevant1)
        ap2 = compute_ap(CLASS_ID1, GROUND_TRUTH2, relevant2)
        print 'cosine similarity: %f' % (np.dot(ap1, ap2)/(np.linalg.norm(ap1)*np.linalg.norm(ap2)))
        print 'average precision of %s: ' % filename1, ap1
        print 'average precision of %s: ' % filename2, ap2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'result_db1',
        help='Inference cases (validation set) from the current database on base model'
    )
    parser.add_argument(
        'result_db2',
        help='Inference cases (validation set) from new candidate database to be attached against base model'
    )

    args = parser.parse_args()

    print_similarity(
        args.result_db1,
        args.result_db2
    )