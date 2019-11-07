import os
import errno
import pandas as pd
import utils.directories as dirs

class Content:
    def __init__(self):
        content_path = os.path.join(dirs.raw_data_dir(), "embedded_clean_content.pkl")
        if not os.path.exists(content_path) or not os.path.isfile(content_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f"Content file doesn't exist, should be at: {content_path}")
        self.content = pd.read_pickle(content_path)
        content_taxon_mapping_path = os.path.join(dirs.raw_data_dir(), 'content_to_taxon_map.csv')
        if not os.path.exists(content_taxon_mapping_path) or not os.path.isfile(content_taxon_mapping_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f"Content taxon mapping file doesn't exist, should be at: {content_taxon_mapping_path}")
        self.content_taxon_mapping = pd.read_csv(content_taxon_mapping_path, low_memory=False)

    def content_for_taxon(self, taxon):
        """
        Returns 'combined_text' content directly within a taxon
        :param taxon: an instance of Node you want the content for
        :return: List of content for that taxon
        """
        content_ids_for_taxon = list(self.content_taxon_mapping[self.content_taxon_mapping['taxon_id'] == taxon.content_id]['content_id'])
        return self.content[self.content['content_id'].isin(content_ids_for_taxon)]['combined_text'].to_list();

    def content_rows_for_taxon(self, taxon):
        """
        Returns content all rows directly within a taxon
        :param taxon: an instance of Node you want the content for
        :return: Dataframe of content for that taxon
        """
        content_ids_for_taxon = list(self.content_taxon_mapping[self.content_taxon_mapping['taxon_id'] == taxon.content_id]['content_id'])
        return self.content[self.content['content_id'].isin(content_ids_for_taxon)];
