# =================================================================================================
# Written by Muhammad Mahir Latif
# word2vec Tensorflow execution, based on implementation in files "wordvector", "windowmodel" and
# "docload" by Patrick Coady.
# https://github.com/pat-coady/word2vec
#
#==================================================================================================
#import libraries
from src.wordvector import WordVector
from src.windowmodel import WindowModel
import src.docload as docload
import numpy as np
import sklearn.utils
import time
import math


#================================================================================================

# give path to a previously saved corpus transformation.
# leave blank if processing new text for the first time.
load_path = ''

#Files for text corpus. Can give multiple inputs
files = ['data/ASoundOfThunder.txt']
        #,'data/sign_of_the_four.txt'
        #,'data/adventures_of_sherlock_holmes.txt']
        #,'data/preprocessed-input-large']


# the following code runs the below functions in order
# load_books() -> get the word_counter (count of occurrence each word) and the word_list (details in docload).
# build_dict() -> take the counter and list from prev function and build a dictionary,
#                 sorted from most common to least common. If unique words exceed vocab_size by "n" ,
#                 ignore the least common "n" words.
# doc2num() -> returns "word_array", a numerical transformation for all words, where each word is represented
#              by a number which is that word's position in the dictionary. Words not found in dictionary
#              will be mapped to 1 larger than biggest value in dictionary. The array order will match
#              the order in the document (i.e. word_array[0] is the first word in the document,
#              word_array[1] is the 2nd word, ...)

if load_path == '':
    word_array, dictionary, num_lines, num_words, num_unique_words = docload.build_word_array(
    files, vocab_size=80000)

    # Save to pickle file for faster loading next time.
    save_path = 'data/DataPickle'
    docload.save_word_array(save_path, word_array, dictionary, num_lines, num_words, num_unique_words)

else:
    word_array, dictionary, num_lines, num_words, num_unique_words = docload.load_word_array(load_path)

print('Documents loaded and processed: {} lines, {} total words, {} unique words.'
      .format(num_lines, num_words, num_unique_words))

#================================================================================================
#training set
print('Building training set ...')

#The below function takes the word_array defined in the data loading process.
#The window size is fixed at "2". More details in "windowmodel.py"
#y is all the "middle" words. x are the corresponding 4 surrounding words.
#middle words that are edge cases (less than 4 surrounding are ignored.
x, y = WindowModel.build_training_set(word_array)

# shuffle and split 10% validation data
x_shuf, y_shuf = sklearn.utils.shuffle(x, y, random_state=0)
split = round(x_shuf.shape[0]*0.9)
x_val, y_val = (x_shuf[split:, :], y_shuf[split:, :])
x_train, y_train = (x[:split, :], y[:split, :])

print('Training set built.')

#set parameters and initialize model
graph_params = {'batch_size': 32,
                'vocab_size': np.max(x)+1,
                'embed_size': 64,
                'hid_size': 64,
                'neg_samples': 64,
                'learn_rate': 0.01,
                'momentum': 0.9,
                'embed_noise': 0.1,
                'hid_noise': 0.3,
                'optimizer': 'Momentum'}
model = WindowModel(graph_params)


print('Model built. Vocab size = {}. Document length = {} words.'
      .format(np.max(x)+1, len(word_array)))



#Train Model
print('Training ...')
starttime = time.time()
results = model.train(x_train, y_train, x_val, y_val, epochs=100, verbose=True)
endtime = time.time()

print("Model Trained in ", math.floor((endtime-starttime)/60)," minutes and ", (endtime-starttime)%60 , " seconds.")


#WordVector class of our trained embedding matrix
word_vector_embed = WordVector(results['embed_weights'], dictionary)


#================================================================================================
#input a word

word = "shoot"
print('Word Embedding of :', "'" + word + "'")
print(word_vector_embed.get_vector_by_name(word=word), '\n')
print('Embedding layer: 8 closest words to:', "'" + word + "'")
print(word_vector_embed.n_closest(word=word, num_closest=8, metric='cosine'), '\n')


#TSNE plot
embed_2d,word_list = word_vector_embed.project_2d(20,500)

