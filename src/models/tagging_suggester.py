from utils.content import *
from utils.tree import *
from data.representative_content import *
from models.apex_node_predictor import *
from models.branch_predictor import *
import datetime

class TaggingSuggester:
    def train(self):
        """
        Processes data and trains models necessary to make tagging suggestions.
        Saves all files to disk
        """
        print("START")
        start = datetime.datetime.now()
        print("Loading data")
        content = Content()
        tree = Tree()
        print("Processing data")
        RepresentativeContent(content, tree).generate()
        print("Training models")
        ApexNodePredictor().train(tree)
        BranchPredictor().train(content, tree)
        end = datetime.datetime.now()
        print(f"DONE! Took: {end - start}")

    def predict(self, text):
        print("PREDICTING")
        start = datetime.datetime.now()
        tree = Tree()
        apex_nodes = ApexNodePredictor().predict(tree, text)
        suggestions = []
        for apex_node in apex_nodes:
            suggestions.append(BranchPredictor().predict(apex_node, text))
        end = datetime.datetime.now()
        print(f"DONE! Took {end - start}")
        for suggestion in suggestions:
            print(suggestion)
        return suggestions
