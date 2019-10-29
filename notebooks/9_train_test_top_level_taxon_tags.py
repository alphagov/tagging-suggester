# This is an experimental script to make models for top level tag suggestions
# it uses output from 8_get_representative_content to be able to cross validate
# it's suggestions.
# Some parts of this script take a very long time to run and are very computationally
# intensive, hence the liberal use of pickle. I recommend running this in a console,
# generating pickle files, clearing large objects and loading from pickle where necessary
# in order to make it faster/a crash less likely to set you back by some time.

import os
import pandas as pd
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from tree import Tree
import pickle
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

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

# This gets all well placed content in a branch of the tree
# (ie all children of a top level)
# returns this in a list
def get_recursive_content_for_taxon_ignoring_test_data(taxon, content, test_data):
    all_test_data_content_ids = test_data['content_id'].to_list()
    all_child_taxon_ids = [taxon.content_id for taxon in branch_node.recursive_children()]
    content_taxon_mapping_path = os.path.join(DATADIR, 'content_to_taxon_map.csv')
    content_taxon_mapping = pd.read_csv(content_taxon_mapping_path, low_memory=False)
    content_ids_for_taxon = list(content_taxon_mapping[content_taxon_mapping['taxon_id'].isin(all_child_taxon_ids)]['content_id'])
    content_in_top_level_taxon = content[content['content_id'].isin(content_ids_for_taxon)].copy()
    print(f"Num pieces content in: {taxon.title}: {str(content_in_top_level_taxon.shape)}")
    content_except_test_data = content_in_top_level_taxon[~content_in_top_level_taxon['content_id'].isin(all_test_data_content_ids)]
    print(f"Num pieces content in: {taxon.title} without test_data = : {str(content_except_test_data.shape)}")
    return content_except_test_data['combined_text'].to_list();

DATADIR = os.getenv("DATADIR")
if DATADIR is None:
    print("You must set a DATADIR environment variable, see the readme in alphagov/govuk-taxonomy-supervised-learning repo for more details")
    sys.exit()
tree = Tree(DATADIR)
clean_content_path = os.path.join(DATADIR, 'embedded_clean_content.pkl')
content = pd.read_pickle(clean_content_path)
well_placed_content = {}
test_datas = {}
model = None
vectorizer = None
texts = []
y = []
for branch_node in tree.apex_nodes():
    most_representative_content_path = os.path.join(DATADIR, f"well_placed_content_#{branch_node.title}.csv")
    most_representative_content = pd.read_csv(most_representative_content_path, low_memory=False)
    test_data = most_representative_content.sample(frac=0.7)
    test_datas[branch_node.title] = test_data
    texts_for_branch_node = get_recursive_content_for_taxon_ignoring_test_data(branch_node, content, test_data)
    texts += texts_for_branch_node
    y += [branch_node.title] * len(texts_for_branch_node)

vectorizer = TfidfVectorizer(tokenizer=tokenize, analyzer='word', stop_words='english', max_features=1000 )
X = vectorizer.fit_transform(texts).toarray()

pickle.dump(X, open("x.pkl", "wb"))
pickle.dump(y, open("y.pkl", "wb"))
pickle.dump(vectorizer, open("TfidfVectorizer.pkl", "wb"))
pickle.dump(test_datas, open("test_datas.pkl", "wb"))

model = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=200).fit(X, y)
pickle.dump(test_datas, open("LogisticRegression.pkl", "wb"))
model = pickle.load(open("LogisticRegression.pkl", "rb"))
vectorizer = pickle.load(open("TfidfVectorizer.pkl", "rb"))

X = pickle.load(open("x.pkl", "rb"))
model = MLPClassifier(solver='lbfgs', alpha=1e-6,
                    hidden_layer_sizes=(len(tree.apex_nodes())), random_state=1)
model.fit(X, y)
pickle.dump(model, open("MLPClassifier.pkl" "wb"))

taxons_for_content_ids = {}
for actual_taxon_name, test_data in test_datas.items():
    for index, row in test_data.iterrows():
        content_id = row['content_id']
        if content_id not in taxons_for_content_ids:
            taxons_for_content_ids[content_id] = []
        if actual_taxon_name not in taxons_for_content_ids[content_id]:
            taxons_for_content_ids[content_id].append(actual_taxon_name)

f_ones = {}
for threshold in [0.2, 0.3, 0.4, 0.5]:
    all_predictions = []
    all_actuals = []
    for taxon_name, test_data in test_datas.items():
        for index, row in test_data.iterrows():
            # Multiple suggestions
            indicies = np.argwhere(model.predict_proba(vectorizer.transform([row['combined_text']]))[0] > threshold)
            all_predictions.append([item for sublist in model.classes_[indicies].tolist() for item in sublist])
            # Single suggestion
            # indicies = np.array([np.argmax(model.predict_proba(vectorizer.transform([row['combined_text']]))[0] > threshold)])
            # all_predictions.append(model.classes_[indicies].tolist())
            # Shared
            all_actuals.append(taxons_for_content_ids[row['content_id']])
    key = str(threshold)
    m = MultiLabelBinarizer().fit(all_actuals)
    f_ones[key] = f1_score(m.transform(all_actuals), m.transform(all_predictions), average='macro')
print(f_ones)