import os
import errno
import pandas as pd
from sklearn.metrics import pairwise_distances_chunked
from utils import directories as dirs

class RepresentativeContent:
    def __init__(self, content, tree):
        self.content = content
        self.tree = tree

    def generate(self):
        """
        Generates csv files of the top 1000 pieces of content in all children of
        apex nodes that is considered similar enough to the rest of the content
        in that apex node
        :return: Dictionary of locations of those csv files
        """
        well_placed_content = {}
        for apex_node in self.tree.apex_nodes():
            well_placed_content[apex_node.title] = self.find_well_placed_items_in_apex_node(apex_node)
        return well_placed_content

    def find_well_placed_items_in_apex_node(self, apex_node):
        """
        Finds all content that it considers to be representative of a branch node
        and all it's children
        in that apex node
        :param apex_node: Instance of Node class for an apex node
        :return: Location of csv file for representative content for that apex node
        """
        representative_content_path = RepresentativeContent.path_for_representative_content(apex_node)
        if os.path.exists(representative_content_path) and os.path.isfile(representative_content_path):
            print(f"File exists for apex node: {apex_node.unique_title()}, loading from file: #{representative_content_path}")
            return
        print(f"File does not exist for apex node: {apex_node.unique_title()}, generating...")
        taxons_to_search = [apex_node] + apex_node.recursive_children()
        all_content_for_apex = pd.DataFrame()
        for taxon in taxons_to_search:
            print('    Looking for content in taxon: ', taxon.title)
            all_content_for_apex = all_content_for_apex.append(self.content.content_rows_for_taxon(taxon))
        distances_between_all_content_item_pairs = pairwise_distances_chunked(
            all_content_for_apex['combined_text_embedding'].to_list(),
            metric = 'cosine',
            n_jobs = -1
        )
        all_content_for_apex['mean_cosine_score'] = list(enumerate(distances_between_all_content_item_pairs))[0][1][0]
        representative_content = all_content_for_apex.sort_values('mean_cosine_score',ascending = True).head(1000)
        print(f"Found {str(len(representative_content))} representative items. Saving csv to: {representative_content_path}")
        representative_content.to_csv(representative_content_path)
        return representative_content_path;

    def path_for_representative_content(taxon):
        return os.path.join(dirs.processed_data_dir(), "representative_content", f"representative_content_#{taxon.unique_title()}.csv")

    def representative_content_for_taxon(taxon):
        """
        Representative content for a taxon.
        Will throw an error if the file it needs does not exist
        :param taxon: Instance of Node class for a taxon you want content for
        :return: List, content for that taxon
        """
        # TODO: What to do about test/train content?
        representative_content_path = RepresentativeContent.path_for_representative_content(taxon)
        if os.path.exists(representative_content_path) and os.path.isfile(representative_content_path):
            most_representative_content = pd.read_csv(representative_content_path, low_memory=False)
            return most_representative_content['combined_text'].to_list()
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f"Representative content file for {taxon.unique_title()} has not been generated, should be at: {representative_content_path}")