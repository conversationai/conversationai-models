"""Tests for dawid_skene."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import pandas as pd
import tempfile
import unittest

import dawid_skene

class DawidSkeneTest(unittest.TestCase):

  # The contents of Maximum Likelihood Estimation of Observer Error-Rates
  # Using the EM Algorithm Table 1.
  def setUp(self):
    self.table_1 = pd.DataFrame.from_dict({
        'patient':
            range(1, 46),
        11: [
            1, 3, 1, 2, 2, 2, 1, 3, 2, 2, 4, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2,
            2, 1, 1, 2, 1, 1, 1, 1, 3, 1, 2, 2, 4, 2, 2, 3, 1, 1, 1, 2, 1, 2
        ],
        12: [
            1, 3, 1, 2, 2, 2, 2, 3, 2, 3, 4, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2,
            2, 1, 1, 3, 1, 1, 1, 1, 3, 1, 2, 2, 3, 2, 3, 3, 1, 1, 2, 3, 2, 2
        ],
        13: [
            1, 3, 2, 2, 2, 2, 2, 3, 2, 2, 4, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2,
            1, 1, 1, 2, 1, 1, 2, 1, 3, 1, 2, 2, 3, 1, 2, 3, 1, 1, 1, 2, 1, 2
        ],
        2: [
            1, 4, 2, 3, 3, 3, 2, 3, 2, 2, 4, 3, 1, 3, 1, 2, 1, 1, 2, 1, 2, 2, 3,
            2, 1, 1, 2, 1, 1, 1, 1, 3, 1, 2, 3, 4, 2, 3, 3, 1, 1, 2, 2, 1, 2
        ],
        3: [
            1, 3, 1, 1, 2, 3, 1, 4, 2, 2, 4, 3, 1, 2, 1, 1, 1, 1, 2, 3, 2, 2, 2,
            2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 3, 2, 2, 4, 1, 1, 1, 2, 1, 2
        ],
        4: [
            1, 3, 2, 2, 2, 2, 1, 3, 2, 2, 4, 4, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
            2, 1, 1, 2, 1, 1, 2, 1, 3, 1, 2, 3, 4, 3, 3, 3, 1, 1, 1, 2, 1, 2
        ],
        5: [
            1, 4, 2, 1, 2, 2, 1, 3, 3, 3, 4, 3, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2,
            2, 1, 1, 2, 1, 1, 1, 1, 3, 1, 2, 2, 3, 2, 3, 2, 1, 1, 1, 2, 1, 2
        ]
    })

  def test_paper_example(self):
    with tempfile.TemporaryDirectory() as tempdirname:
      f = tempfile.NamedTemporaryFile(delete=False)
      f.file.close()
      data = self.table_1.set_index('patient').stack().rename_axis(['patient', 'observer']).to_frame('label').reset_index()
      data['observer'] = data['observer'].map({11:1, 12:1, 13:1, 2:2, 3:3, 4:4, 5:5})
      data.to_csv(f.name, header=True)

      Flags = collections.namedtuple('Flags', 'n_examples label unit_id_col worker_id_col comment_text_path data_path pseudo_count tolerance max_iter job_dir')
      Flags.data_path = f.name
      Flags.label = 'label'
      Flags.worker_id_col = 'observer'
      Flags.unit_id_col = 'patient'
      Flags.n_examples = 350
      Flags.pseudo_count = 1.0
      Flags.comment_text_path = None
      Flags.max_iter = 25
      Flags.tolerance = 1
      Flags.job_dir = tempdirname
      dawid_skene.main(Flags)
      os.unlink(f.name)
      predictions = pd.read_csv(os.path.join(tempdirname, 'predictions_label_315.csv'))
      print(predictions)
      error_rates = pd.read_csv(os.path.join(tempdirname, 'error_rates_label_315.csv'))
      print(error_rates)


if __name__ == '__main__':
  unittest.main()
