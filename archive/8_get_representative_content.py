# Gets 1000 most 'representative' pieces of content for each top level taxon
# (and it's children


import os
import pandas as pd
from sklearn.metrics import pairwise_distances_chunked
from nltk.stem.porter import PorterStemmer
import nltk
from tree import Tree

def get_well_placed_content_in_taxon(content, taxon, similarity_threshold = 0.5):
    print('Looking for misplaced content in taxon: ', taxon.title)
    taxon_embeddings = get_embedded_sentences_for_taxon(content, taxon)
    distances_between_all_content_item_pairs = pairwise_distances(
        taxon_embeddings,
        metric = 'cosine',
        n_jobs = -1
    )
    content_for_taxon = get_content_for_taxon(content, taxon).copy()
    content_for_taxon['mean_cosine_score'] = distances_between_all_content_item_pairs.mean(axis=1)
    well_placed_content = content_for_taxon.loc[content_for_taxon['mean_cosine_score'] < similarity_threshold].copy()
    well_placed_content["taxon_id"] = taxon.content_id
    well_placed_content["taxon_title"] = taxon.unique_title()
    return well_placed_content;

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

def get_content_for_taxon(content, taxon):
    content_taxon_mapping_path = os.path.join(DATADIR, 'content_to_taxon_map.csv')
    content_taxon_mapping = pd.read_csv(content_taxon_mapping_path, low_memory=False)
    content_ids_for_taxon = list(content_taxon_mapping[content_taxon_mapping['taxon_id'] == taxon.content_id]['content_id'])
    return content[content['content_id'].isin(content_ids_for_taxon)];

# Finds all content that might be incorrectly tagged
# Currently hard coded to look in money branch but could look anywhere
def find_well_placed_items(apex_node, content, tree):
    well_placed_content_path = os.path.join(DATADIR, f"well_placed_content_#{apex_node.title}.csv")
    if os.path.exists(well_placed_content_path) and os.path.isfile(well_placed_content_path):
        print(f"file exists: #{well_placed_content_path}")
        return well_placed_content_path
    taxons_to_search = [apex_node] + apex_node.recursive_children()
    all_content_for_apex = pd.DataFrame()
    for taxon in taxons_to_search:
        print('Looking for content in taxon: ', taxon.title)
        all_content_for_apex = all_content_for_apex.append(get_content_for_taxon(content, taxon))
    distances_between_all_content_item_pairs = pairwise_distances_chunked(
        all_content_for_apex['combined_text_embedding'].to_list(),
        metric = 'cosine',
        n_jobs = -1
    )
    all_content_for_apex['mean_cosine_score'] = list(enumerate(distances_between_all_content_item_pairs))[0][1][0]
    well_placed_items = all_content_for_apex.sort_values('mean_cosine_score',ascending = True).head(1000)
    well_placed_content_path = os.path.join(DATADIR, f"well_placed_content_#{apex_node.title}.csv")
    print("Found " + str(len(well_placed_items)) + " misplaced items. Saving csv to " + well_placed_content_path)
    well_placed_items.to_csv(well_placed_content_path)
    return well_placed_content_path;



DATADIR = os.getenv("DATADIR")
if DATADIR is None:
    print("You must set a DATADIR environment variable, see the readme in alphagov/govuk-taxonomy-supervised-learning repo for more details")
    sys.exit()
tree = Tree(DATADIR)
clean_content_path = os.path.join(DATADIR, 'embedded_clean_content.pkl')
content = pd.read_pickle(clean_content_path)
well_placed_content = {}
for branch_node in tree.apex_nodes():
    well_placed_content[branch_node.title] = find_well_placed_items(branch_node, content, tree)

