import os

def database_url():
    database_url = os.getenv("TAGGING-SUGGESTER-DATABASE-URL")
    if database_url is None:
        database_url = "postgres://taggingsuggester@localhost:5432/tagging_suggester_development"
    return database_url
