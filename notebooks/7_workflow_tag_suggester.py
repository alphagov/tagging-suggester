# * This script is designed to suggest topics for a new piece of content
# his is done by using TF-IDF and logistic regression
# to learn which terms are seen in which taxons. Each taxon's child content are included in it's corpus when doing
# so as a child taxon may have lots of specific terms not seen in it's parent taxon. It also uses depth first to
# traverse the tree so it's important not to lose this information when deciding which taxon is most relevant
# compared to it's siblings

import os
import pandas as pd
import csv
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tree import Tree
import sys
import pickle
import numpy as np

def fetch_combined_text_for_new_item():
    return "blah"

def suggest_taxons(tree, combined_text_for_new_item):
    top_level_taxons = find_top_level_taxons(tree, combined_text_for_new_item)
    taxons = []
    for taxon in top_level_taxons:
        taxons.append(find_lower_level_taxons(taxon, combined_text_for_new_item))
    return taxons

def find_top_level_taxons(tree, combined_text_for_new_item):
    node_scores = {}
    model_path = os.path.join(DATADIR, "top_level_model.pkl")
    model = pickle.load(open(model_path, "rb"))
    model.predict([combined_text_for_new_item])


def find_lower_level_taxons(taxon, combined_text_for_new_item):
    while any(taxon.children):
        print("Looking at children of " + taxon.title)
        taxon_content_id, probability = get_score_for(taxon, taxon.children, combined_text_for_new_item)
        if not taxon_content_id:
            # There are no scores, so none were relevant, we can break
            break
        else:
            best_taxon = tree.find(taxon_content_id)
            print("Best taxon is: " + best_taxon.title)
            print("Which has probability score of: " + str(probability) + "%")
            taxon = best_taxon
    return taxon

if __name__== "__main__":
    DATADIR = os.getenv("DATADIR")
    if DATADIR is None:
        print("You must set a DATADIR environment variable, see the readme in alphagov/govuk-taxonomy-supervised-learning repo for more details")
        sys.exit()
    content_path = os.path.join(DATADIR, 'embedded_clean_content.pkl')
    CONTENT = pd.read_pickle(content_path)
    tree = Tree(DATADIR)
    combined_text_for_new_item = fetch_combined_text_for_new_item()
    suggest_taxons(tree, combined_text_for_new_item)

def get_embedded_sentences_for_taxon(content, taxon):
    return get_content_for_taxon(content, taxon)['combined_text_embedding'].to_list()

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

def get_embedded_titles_for_taxon(content, taxon):
    return get_content_for_taxon(content, taxon)['title_embedding'].to_list()

def get_content_for_taxon(taxon):
    content_taxon_mapping_path = os.path.join(DATADIR, 'content_to_taxon_map.csv')
    content_taxon_mapping = pd.read_csv(content_taxon_mapping_path, low_memory=False)
    content_ids_for_taxon = list(content_taxon_mapping[content_taxon_mapping['taxon_id'] == taxon.content_id]['content_id'])
    return CONTENT[CONTENT['content_id'].isin(content_ids_for_taxon)];

def get_score_for(current_node, child_taxons, content_item_content):
    content_ids = [taxon.content_id for taxon in child_taxons]
    content_ids.sort()
    key = ",".join(content_ids)
    model_path = os.path.join(DATADIR, f"tag_suggestion_#{key}_model.pkl")
    vectorizer_path = os.path.join(DATADIR, f"tag_suggestion_#{key}_vectorizer.pkl")
    print(model_path)
    print(vectorizer_path)
    if os.path.exists(model_path) and os.path.isfile(model_path):
        texts = []
        y = []
        for child_taxon in child_taxons:
            if child_taxon.content_id == current_node.content_id:
                # Ignore taxon if its the same as the current one
                print("Ignoring " + child_taxon.title + " as its the same as " + current_node.title)
                continue
            length_of_content = []
            print("looking at all children of " + child_taxon.title)
            for taxon in child_taxon.recursive_children():
                print("Those children include: " + taxon.title)
                for i, content_item in get_content_for_taxon(taxon).iterrows():
                    texts.append(content_item['combined_text'])
                    y.append(child_taxon.content_id)
                    length_of_content.append(len(tokenize(content_item['combined_text'])))
        length_of_content.sort()
        if len(list(set(y))) <= 1 or len(length_of_content) == 0:
            print("One or fewer classes, returning early")
            return (False, False);
        max_features = 500
        vectorizer = TfidfVectorizer(tokenizer=tokenize, analyzer='word', stop_words='english', max_features=max_features )
        X = vectorizer.fit_transform(texts).toarray()
        pickle.dump(vectorizer, open(vectorizer_path, "wb" ) )
        model = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=200).fit(X, y)
        pickle.dump(model, open(model_path, "wb"))
    else:
        model = pickle.load(open(model_path, "rb"))
        vectorizer = pickle.load(open(vectorizer_path, "rb"))
    prediction = model.predict(vectorizer.transform([content_item_content]))[0]
    probability = np.max(model.predict_proba(vectorizer.transform([content_item_content]))[0])[0]
    return (prediction, probability);


# # global vectorizers
# models = {}
# vectorizers = {}
# content_to_retag = []
# content_for_human_verification_to_untag = []
# content_to_untag = []
# debugging_info = []
# for index, row in problem_content.iterrows():
#     content_to_retag_base_path = row["base_path"]
#     current_taxon = tree.find(row["taxon_id"])
#     if not current_taxon.all_siblings_and_children():
#         # No children or siblings in the same branch for current taxon, just return it
#         return current_taxon
#     content_item_content = content[content['base_path'] == content_to_retag_base_path].iloc[0,:]['combined_text']
#     # Get the score of the current taxon so we can see if it's children do any better
#     scores_for_current_taxon = -1
#     if scores_for_current_taxon:
#         best_score = scores_for_current_taxon
#         node = apex_node
#         while any(node.children):
#             print("Looking at children of " + node.title)
#             taxon_content_id, probability = get_score_for(node, node.children, content_item_content, [], content)
#             if not taxon_content_id:
#                 # There are no scores, so none were relevant, we can break
#                 break
#             else:
#                 best_taxon = tree.find(taxon_content_id)
#                 print("Best taxon is: " + best_taxon.title)
#                 print("Which has probability score of: " + str(probability) + "%")
#                 best_score = probability
#                 node = best_taxon
#         if node is not apex_node and node is not current_taxon:
#             content_to_retag.append([content_to_retag_base_path, current_taxon.title_and_parent_title(), current_taxon.base_path, node.content_id, node.title, node.title_and_parent_title(), node.base_path, best_score, []])
#
# with open("content_to_retag_" + apex_node.title + ".csv", 'w') as csvfile:
#     filewriter = csv.writer(csvfile)
#     filewriter.writerow(['content_to_retag_base_path', 'current_taxon_title', 'current_taxon_base_path', 'suggestion_content_id', 'suggestion_title', 'suggestion_title_and_level_1', 'suggestion_base_path', 'suggestion_cosine_score', 'other_suggestions'])
#     for row in content_to_retag:
#         filewriter.writerow(row)
#
# with open("depth_first_content_for_human_verification_to_untag_" + apex_node.title + ".csv", 'w') as csvfile:
#     filewriter = csv.writer(csvfile)
#     filewriter.writerow(['content_to_retag_base_path', "current_taxon", "current_taxon_base_path", "more_info"])
#     for row in content_for_human_verification_to_untag:
#         filewriter.writerow(row)
#
# with open("depth_first_content_to_untag_" + apex_node.title + ".csv", 'w') as csvfile:
#     filewriter = csv.writer(csvfile)
#     filewriter.writerow(['content_to_retag_base_path', "current_taxon", "current_taxon_base_path", "more_info"])
#     for row in content_to_untag:
#         filewriter.writerow(row)