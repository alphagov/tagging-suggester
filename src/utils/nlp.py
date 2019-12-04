from nltk.stem.porter import PorterStemmer
import nltk
import numpy as np
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def dictionary_of_translated_tokenized_text(text):
    translated_tokenized_text = {}
    for word in text.split(" "):
        translated_tokenized_text["".join(tokenize(word))] = word
    return translated_tokenized_text

def prediction_explanation(vectorizer, transformed_text, translated_tokenized_text_to_predict):
    """
    Returns the top 5 words that most explain a prediction
    """
    indices_of_top_words = np.argsort(transformed_text.toarray()[0])[::-1][0:5]
    feature_names = vectorizer.get_feature_names()
    prediction_words = []
    for index in indices_of_top_words:
        if transformed_text[0,index] > 0:
            tokenized_feature_name = feature_names[index]
            words_and_scores = {}
            for tokenized_word, word in translated_tokenized_text_to_predict.items():
                if tokenized_feature_name in tokenized_word:
                    score = normalized_damerau_levenshtein_distance(tokenized_word, tokenized_feature_name)
                    words_and_scores[word] = score
            if any(words_and_scores):
                best_word = sorted(words_and_scores.items(), key=lambda kv: kv[1])[0][0]
                prediction_words.append(best_word)
    # Return a unique but still ordered list of words
    seen = set()
    seen_add = seen.add
    return [x for x in prediction_words if not (x in seen or seen_add(x))]
