from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import os

DEFAULT_DATA_DIR = 'data/'

def prepare_figshare(input_data_dir = DEFAULT_DATA_DIR, output_data_dir = DEFAULT_DATA_DIR):
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
        
if __name__ == "__main__":
    prepare_figshare()