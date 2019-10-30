from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import utils.directories as dirs
import utils.nlp as nlp
import os

class BranchPredictor:
    def train(self, content, tree):
        """
        Trains the predictor and generates all necessary models, saving them to file
        :param content: instance of Content class
        :param tree: instance of Tree class
        :return: None
        """
        self.content = content
        self.tree = tree
        for apex_node in self.tree.apex_nodes():
            for node in apex_node.children:
                self.train_models_for_node_and_children(node)

    def train_models_for_node_and_children(self, node):
        """
        Trains the models for an apex node and all it's children, saving them to file
        :param node: an apex node instance of Node
        :return: None
        """
        self.train_model_for_node(node)
        for child_node in node.children:
            self.train_models_for_node_and_children(child_node)

    def train_model_for_node(self, node):
        """
        Trains the models for an node, saving it to file
        :param node: an instance of Node
        :return: None
        """
        # recursive_children includes self so we only need to search
        # if there is more than one
        if os.path.isfile(self.model_path(node)) == False:
            if len(node.recursive_children()) > 1:
                texts = []
                y = []
                for child_node in node.recursive_children():
                    content_for_taxon = self.content.content_for_taxon(child_node)
                    if len(content_for_taxon) > 0:
                        texts += content_for_taxon
                        y += [child_node.unique_title()] * len(content_for_taxon)
                # Check there is more than one class to train on
                if len(list(set(y))) > 1:
                    print(f"Generating BranchPredictor model for {node.title}")
                    vectorizer = TfidfVectorizer(tokenizer=nlp.tokenize, analyzer='word', stop_words='english', max_features=500 )
                    X = vectorizer.fit_transform(texts).toarray()
                    dirs.save_pickle_file(vectorizer, self.vectorizer_path(node))
                    model = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=200)
                    model.fit(X, y)
                    dirs.save_pickle_file(model, self.model_path(node))

    def model_path(self, node):
        return os.path.join(dirs.processed_data_dir(), "models", "branch_predictor", f"branch_predictor_model_#{node.unique_title()}.pkl")

    def vectorizer_path(self, node):
        return os.path.join(dirs.processed_data_dir(), "models", "branch_predictor", f"branch_predictor_vectorizer_#{node.unique_title()}.pkl")