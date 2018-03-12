"""

A class to help visualize attention weights.

------------------------------------------------------------------------

Copyright 2018, Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
import numpy as np

pd.set_option('max_columns', 100)
tokenizer = tf.contrib.learn.preprocessing.tokenizer
WORDS_FEATURE = 'words'
MAX_DOCUMENT_LENGTH = 60

class wordVal(object):
    """A helper class that represents a word and value simultaneously."""
    def __init__(self, word, val):
        self.word = word
        self.val = val
  
    def __str__(self):
        return self.word

class attentionDisplay(object):
    """A class to visualize attention weights produced by a classifer on a given string."""
 
    def __init__(self, vocab_processor, classifier, words_feature = 'words'):
        """
        Args:
          * vocab_processor: a trained vocabulary processor from tf.contrib.learn.preprocessing.VocabularyProcessor 
          * classifier: the classifier of class Estimator produced in Attention_Model_Codelab.ipynb
          * words_feature (string): if provided, the key for the comments in the feed dictionary expected by the classifier
        """
        
        self.vocab_processor = vocab_processor
        self.classifier = classifier
        self.words_feature = words_feature
    
    def _rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % rgb

    def _color_wordvals(self, s):
        r = 255-int(s.val*255)
        color = self._rgb_to_hex((255, r, r))
        return 'background-color: %s' % color

    def _predict_sentence(self, input_string):
        x_test = self.vocab_processor.transform([input_string])
        x_test = np.array(list(x_test))
        
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={self.words_feature: x_test},
            num_epochs=1,
            shuffle=False)
        
        predictions = self.classifier.predict(input_fn=test_input_fn)
        y_predicted = []
        alphas_predicted = []
        for p in predictions:
            y_predicted.append(p['class'])
            alphas_predicted.append(p['attention'])
        return y_predicted, alphas_predicted

    def _resize_and_tokenize(self, input_string):
        tokenized_sentence = list(tokenizer([input_string]))[0]
        tokenized_sentence = tokenized_sentence + [''] * (MAX_DOCUMENT_LENGTH - len(tokenized_sentence))
        tokenized_sentence = tokenized_sentence[:MAX_DOCUMENT_LENGTH]
        return tokenized_sentence

    def display_prediction_attention(self, input_string):
        """Visualizes the attention weights of the initialized classifier on the given string."""
        pred, attn = self._predict_sentence(input_string)
        if pred[0]:
            print('Toxic')
        else:
            print('Not toxic')
        tokenized_string = self._resize_and_tokenize(input_string)
        wordvals = [wordVal(w,v) for w, v in zip(tokenized_string, attn[0])]
        word_df = pd.DataFrame(wordvals).transpose()
        return word_df.style.applymap(self._color_wordvals)

