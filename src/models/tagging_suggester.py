from src.utils.content import *
from src.utils.tree import *
import src.utils.misc as misc
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

    def predict(self, text, request_record):
        print("PREDICTING")
        start = datetime.datetime.now()
        translated_tokenized_text_to_predict = nlp.dictionary_of_translated_tokenized_text(text)
        apex_nodes, request_record = ApexNodePredictor().predict(self.tree, text, request_record)
        suggestions = []
        for apex_node in apex_nodes:
            suggestion, request_record = BranchPredictor().predict(self.tree, apex_node, text, translated_tokenized_text_to_predict, request_record)
            suggestions.append(suggestion)
        suggestions = misc.flatten(suggestions)
        request_record.predictions = str(suggestions)
        end = datetime.datetime.now()
        duration = end - start
        print(f"DONE! Took {duration}")
        request_record.prediction_duration = str(duration)
        return [suggestions, request_record]
