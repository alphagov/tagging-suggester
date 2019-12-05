from flask import Flask, request, jsonify, g
from src.models.tagging_suggester import *
import nltk
from datetime import datetime
import src.utils.app_config as app_config
from models import Request
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

app = Flask(__name__)

tagging_suggester = TaggingSuggester()

@app.before_request
def before_request():
    engine = create_engine(app_config.database_url(), strategy='threadlocal')
    Session = sessionmaker(bind=engine)
    g.db_session = Session()

@app.route('/create', methods=['POST'])
def tagging_suggestion():
    json = request.get_json()
    request_record = Request(
        created_at = datetime.now().isoformat(),
        edition_id = json['edition_id'],
        branch_predictor_probabilities = "",
        api_version = "1.0",
    )
    result, request_record = tagging_suggester.predict(json['text'], request_record)
    g.db_session.add(request_record)
    g.db_session.commit()
    return jsonify({ "suggestions": result })

@app.after_request
def after_request(response):
    if g.db_session is not None:
        g.db_session.close()
    return response

if __name__=="__main__":
    nltk.download('punkt')
    app.run(debug=True)

