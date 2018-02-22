#!/usr/bin/env python2
import h5py
import math
import argparse
import pylab as pl
import numpy as np

def print_figure(result_db,
                 label_txt):
    """
    Print inference results
    """
    if result_db is None or label_txt is None:
        return
    db = h5py.File(result_db, 'r')

    if db is not None:
        labels = np.loadtxt(label_txt, dtype='object')
        NUM_COLS = 6
        NUM_IMGS = len(db['input_ids'])
        NUM_ROWS = NUM_IMGS // NUM_COLS + (NUM_IMGS % NUM_COLS > 0)
        NUM_TOPK_CLASSES = 3
        fig = pl.figure(figsize=(16,4))
        fig.set_canvas(pl.gcf().canvas)
        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                idx = row*NUM_COLS + col
                if idx == NUM_IMGS:
                    break
                pl.subplot(NUM_ROWS*2, NUM_COLS, row * 2 * NUM_COLS + col + 1)
                pl.xticks([])
                pl.yticks([])
                pl.imshow(db['input_data'][idx], interpolation='nearest')
        
        res = db['outputs'][db['outputs'].keys()[0]]
        for elem_id, elem_data in enumerate(res):
            row = elem_id // NUM_COLS
            col = elem_id % NUM_COLS
            img_labels = sorted(zip(elem_data, labels), key=lambda x: x[0])[-NUM_TOPK_CLASSES:]
            ax = pl.subplot(NUM_ROWS*2, NUM_COLS, (row*2 + 1)*NUM_COLS + col + 1, aspect='equal')
            ax.yaxis.set_label_position("right")
            ax.yaxis.set_label_coords(1.25, 0.5)

            height = 10
            margin = 1
            ylocs = np.array(range(NUM_TOPK_CLASSES)) * (height + margin)+ margin
            width = max(ylocs)
            top_class = img_labels[-1][1]
            pl.barh(ylocs, [l[0]*width for l in img_labels], height=height, \
                    color=['r' if l[1] == top_class else 'b' for l in img_labels]) #color=['r' if l[1] == labels[true_label] else 'b' for l in img_labels])
            pl.yticks(ylocs + (height+margin)/2.0, [l[1].replace('_','\n') for l in img_labels], fontsize=16)
            pl.xticks([0, width/2.0, width], ['0%', '50%', '100%'])
            pl.ylim(0, ylocs[-1] + height + margin)
        pl.tight_layout()
        pl.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'result_data',
        help='An input database containing inference results'
    )
    parser.add_argument(
        'label_txt',
        help='Labels to print'
    )

    args = parser.parse_args()

    print_figure(
        args.result_data,
        args.label_txt
        )