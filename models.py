from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date, Text

Base = declarative_base()

class Request(Base):
    __tablename__ = 'requests'
    id = Column(Integer, primary_key=True)
    created_at = Column(Date)
    prediction_duration = Column(String)
    edition_id = Column(Integer)
    apex_node_predictor_probabilities = Column(Text)
    branch_predictor_probabilities = Column(Text)
    predictions = Column(Text)
    api_version = Column(String)

    def __repr__(self):
        return "<Request(created_at='{}', prediction_duration='{}', edition_id='{}', apex_node_predictor_probabilities={}, branch_predictor_probabilities={}, predictions={}, api_version={})>" \
            .format(self.created_at, self.prediction_duration, self.edition_id, self.apex_node_predictor_probabilities, self.branch_predictor_probabilities, self.predictions, self.api_version)
