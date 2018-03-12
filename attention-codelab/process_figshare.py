from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import os

DEFAULT_DATA_DIR = 'data/'

def process_figshare(input_data_dir = DEFAULT_DATA_DIR, output_data_dir = DEFAULT_DATA_DIR):
    
    already_exist = True
    for split in ['train', 'test', 'dev']:
        if not os.path.isfile(os.path.join(output_data_dir, 'wiki_%s.csv' % split)):
            already_exist = False
            
    if already_exist:
        print('Processed files already exist.')
        return

    print('Processing files...', end = '')
    toxicity_annotated_comments = pd.read_csv(os.path.join(input_data_dir, 'toxicity_annotated_comments.tsv'), sep = '\t')
    toxicity_annotations = pd.read_csv(os.path.join(input_data_dir, 'toxicity_annotations.tsv'), sep = '\t')

    annotations_gped = toxicity_annotations.groupby('rev_id', as_index=False).agg({'toxicity': 'mean'})
    all_data = pd.merge(annotations_gped, toxicity_annotated_comments, on = 'rev_id')

    all_data['comment'] = all_data['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    all_data['comment'] = all_data['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

    all_data['is_toxic'] = all_data['toxicity'] > 0.5

    # split into train, valid, test
    wiki_splits = {}
    for split in ['train', 'test', 'dev']:
        wiki_splits[split] = all_data.query('split == @split')

    for split in wiki_splits:
        wiki_splits[split].to_csv(os.path.join(output_data_dir, 'wiki_%s.csv' % split), index=False)
    print('Done!')

# TODO(nthain): Add input and output dirs as flags.
if __name__ == "__main__":
    process_figshare()