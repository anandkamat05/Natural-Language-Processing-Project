'''
Created on Dec 14, 2017

@author: Anand
'''
from __future__ import print_function
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import sent_tokenize
from pythonrouge.pythonrouge import Pythonrouge
from pprint import pprint

#Load training data
positive_data = []
negative_data = []

#Loading Positive data
with open('rt-polarity.pos', 'rb') as pos:
        for sent in pos:
            positive_data.append(sent)
        del(pos)
        
#Loading Negative data
with open('rt-polarity.neg', 'rb') as neg:
        for sent in neg:
            negative_data.append(sent)
        del(neg)

#Loading Commentary Data
with open('./processed_text_commentaries/doc10.txt', 'rb') as commentary:
        for c_text in commentary:
            comm1 = c_text
        del(commentary)

comm1 = str.lower(comm1)

#Sentence Tokenizer creating list of sentences from commentary data
comm1_list = sent_tokenize(comm1, language = 'english')

#List of Sentences where Chelsea is mentioned
filtered_list = []
for sentence in comm1_list:
    if 'huddersfield' in sentence:
        filtered_list.append(sentence)

#PreProcessing
pos_label = []
neg_label = []

for w in positive_data:
    pos_label.append('pos')
    neg_label.append('neg')
    
target = pos_label + neg_label

training_set = np.append(positive_data[:4500], negative_data[:4500])
training_labels = np.append(pos_label[:4500], neg_label[:4500])

test_set = np.append(positive_data[4501:], negative_data[4501:])
test_labels = np.append(pos_label[4501:], neg_label[4501:])

#Using the Count Vectorizer
cv = CountVectorizer(analyzer='word', binary=False, decode_error='ignore'
        , input='content',lowercase=True, max_df=1.0, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words = None,
        strip_accents='ascii', tokenizer=None, vocabulary=None)

training_vector = cv.fit_transform(training_set) 
test_vector = cv.transform(test_set)
pred_vector = cv.transform(filtered_list)

# #Using a Multi Layer Perceptron to train on Data
# mlp_classifier = MLPClassifier(hidden_layer_sizes=(10, ), activation= 'relu', solver= 'adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# mlp_classifier.fit(training_vector, training_labels)
# #Scoring on the test set
# mlp_score = mlp_classifier.score(test_vector, test_labels)
# #Predicting commentator data
# mlp_prediction = mlp_classifier.predict(pred_vector)
# print(mlp_prediction)
# print(mlp_score)


# #Using a Naive Bayes Classifier to train on Data
# nb_classifier = MultinomialNB(alpha = 1.0, fit_prior=True, class_prior=None)
# nb_classifier.fit(training_vector, training_labels)
# #Scoring on the test set
# nb_score = nb_classifier.score(test_vector, test_labels)
# #Predicting commentator data
# nb_prediction = nb_classifier.predict(pred_vector)
# print(nb_prediction)


# #Using a Random Forest Classifier to train on Data
# rf_classifier = RandomForestClassifier(max_depth=2, random_state=0)
# rf_classifier.fit(training_vector, training_labels)
# #Scoring on the test set
# rf_score = rf_classifier.score(test_vector, test_labels)
# #Predicting commentator data
# rf_prediction = rf_classifier.predict(pred_vector)
# print(rf_prediction)
# print(rf_score)



####################################### Evaluation ###################################################################################

if __name__ == '__main__':
    summary = './sample/summary/'
    reference = './sample/reference/'
    ROUGE_dir = './pythonrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl'
    data_dir = './pythonrouge/RELEASE-1.5.5/data/'
    print('evaluate sumamry & reference in these dirs')
    print('summary:\t{}\nreference:\t{}'.format(summary, reference))
    rouge = Pythonrouge(summary_file_exist=True,
                        peer_path=summary, model_path=reference,
                        n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                        recall_only=True,
                        stemming=True, stopwords=True,
                        word_level=True, length_limit=True, length=50,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge.calc_score()
    print('ROUGE-N(1-2) & SU4 F-measure only')
    pprint(score)
    print('Evaluate ROUGE based on sentecnce lists')
    """
    ROUGE evaluates all system summaries and its corresponding reference
    a summary or summaries at onece.
    Summary should be double list, in each list has each summary.
    Reference summaries should be triple list because some of reference
    has multiple gold summaries.
    """
    summary = [["Great location, very good selection of food for\
                 breakfast buffet.",
                "Stunning food, amazing service.",
                "The food is excellent and the service great."],
               ["The keyboard, more than 90% standard size, is just\
                 large enough .",
                "Surprisingly readable screen for the size .",
                "Smaller size videos   play even smoother ."]]
    reference = [
                 [["Food was excellent with a wide range of choices and\
                   good services.", "It was a bit expensive though."],
                  ["Food can be a little bit overpriced, but is good for\
                  hotel."],
                  ["The food in the hotel was a little over priced but\
                  excellent in taste and choice.",
                  "There were also many choices to eat in the near\
                  vicinity of the hotel."]],
                 [["The size is great and allows for excellent\
                   portability.",
                   "Makes it exceptionally easy to tote around, and the\
                   keyboard is fairly big considering the size of this\
                   netbook."],
                  ["Size is small and manageable.",
                   "Perfect size and weight.",
                   "Great size for travel."],
                  ["The keyboard is a decent size, a bit smaller then\
                  average but good.",
                  "The laptop itself is small but big enough do do\
                  things on it."],
                  ["In spite of being small it is still comfortable.",
                  "The screen and keyboard are well sized for use"]]
                  ]
    rouge = Pythonrouge(summary_file_exist=False,
                        summary=summary, reference=reference,
                        n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                        recall_only=True, stemming=True, stopwords=True,
                        word_level=True, length_limit=True, length=50,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge.calc_score()
    print('ROUGE-N(1-2) & SU4 recall only')
    pprint(score)

# r = Rouge155()
# r.system_dir = 'C:/Users/anand/OneDrive/University McGill/NLP/Project/generated_summaries'
# r.model_dir = 'C:/Users/anand/OneDrive/University McGill/NLP/Project/annotated_summaries'
# r.system_filename_pattern = 'Summaries(\d+).txt'
# r.model_filename_pattern = 'Summary[A-Z].#ID#.txt'
# 
# output = r.convert_and_evaluate()
#print(output)
#output_dict = r.output_to_dict(output)