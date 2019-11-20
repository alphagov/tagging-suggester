from src.utils.content import *
from src.utils.tree import *
from src.data.representative_content import *
from src.models.apex_node_predictor import *
from src.models.branch_predictor import BranchPredictor
import datetime

class TaggingSuggester:
    def __init__(self):
        self.tree = Tree()

    def train(self):
        """
        Processes data and trains models necessary to make tagging suggestions.
        Saves all files to disk
        """
        print("START")
        start = datetime.datetime.now()
        print("Loading data")
        content = Content()
        print("Processing data")
        RepresentativeContent(content, self.tree).generate()
        print("Training models")
        ApexNodePredictor().train(self.tree)
        BranchPredictor().train(content, self.tree)
        end = datetime.datetime.now()
        print(f"DONE! Took: {end - start}")

    def predict(self, text):
        print("PREDICTING")
        start = datetime.datetime.now()
        apex_nodes = ApexNodePredictor().predict(self.tree, text)
        suggestions = []
        for apex_node in apex_nodes:
            suggestions.append(BranchPredictor().predict(self.tree, apex_node, text))
        end = datetime.datetime.now()
        print(f"DONE! Took {end - start}")
        for suggestion in suggestions:
            print(suggestion)
        return suggestions
