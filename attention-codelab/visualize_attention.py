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
    def __init__(self, word, val):
        self.word = word
        self.val = val
  
    def __str__(self):
        return self.word

class attentionDisplay(object):
    def __init__(self, vocab_processor, classifier, words_feature = 'words'):
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
        pred, attn = self._predict_sentence(input_string)
        if pred[0]:
            print('Toxic')
        else:
            print('Not toxic')
        tokenized_string = self._resize_and_tokenize(input_string)
        wordvals = [wordVal(w,v) for w, v in zip(tokenized_string, attn[0])]
        word_df = pd.DataFrame(wordvals).transpose()
        return word_df.style.applymap(self._color_wordvals)

