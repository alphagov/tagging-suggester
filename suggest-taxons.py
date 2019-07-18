import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import os
import pandas as pd

from tqdm import tnrange, tqdm_notebook
from sklearn.metrics import pairwise_distances_chunked
import urllib.request
import json


# This script will suggest taxons for a new piece of content

# To try it out, you'll need 'clean_content.csv' in your DATADIR,
# an appropriate environment variable to the DATADIR
# Edit the variable below to contain the text for your new piece of content

text = "I broke free on a Saturday morning I put the pedal to the floor Headed north on Mills Avenue And listened to the engine roar My broken house behind me And good things ahead A girl named Cathy Wants a little of my time Six cylinders underneath the hood Crashing and kicking Aha! Listen to the engine whine I am going to make it through this year If it kills me I am going to make it though this year If it kills me I played video games in a drunken haze I was seventeen years young Hurt my knuckles punching the machines The taste of Scotch rich on my tongue And then Cathy showed up And we hung out Trading swigs from a bottle All bitter and clean Locking eyes Holding hands Twin high maintenance machines I am going to make it through this year If it kills me I am going to make it though this year If it kills me I drove home in the California dusk I could feel the alcohol inside of me hum Pictured the look on my stepfather's face Ready for the bad things to come I down shifted As I pulled into the driveway The motor screaming out Stuck in second gear The scene ends badly As you might imagine In a cavalcade of anger and fear There will be feasting and dancing In Jerusalem next year"

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
embed = hub.Module(module_url)

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    new_content = session.run(embed([text]))

DATADIR = os.getenv("DATADIR")
labelled = pd.read_csv(os.path.join(DATADIR, 'labelled.csv.gz'), compression='gzip', low_memory=False)
embedded_sentences = np.load(os.path.join(DATADIR, 'embedded_clean_contentdata.npy'))

content = pd.read_csv(
    os.path.join(DATADIR, 'clean_content.csv'),
    low_memory=False)

def get_top_20_links(D_chunk, start):
    """return only the top 20 (including self) related link indices and distance metric values
    according to distance metric"""
    top_k_indices = np.argpartition(D_chunk, range(20))[:, :20]
    return top_k_indices, D_chunk[:, top_k_indices]

content_generator = pairwise_distances_chunked(
    X=new_content,
    Y=embedded_sentences,
    reduce_func=get_top_20_links,
    working_memory=0,
    metric='cosine',
    n_jobs=-1)

close_content_urls = []
urls = pd.DataFrame(columns=['close_content_urls', 'cosine_sims'])
for i, (indices, values) in enumerate(tqdm_notebook(content_generator)):
    close_content_urls = [content.iat[i, 0] for i in indices[0]]
    i_urls = pd.DataFrame({
        'close_content_urls': close_content_urls,
        'cosine_sims': values.reshape(20)
    })
    urls = urls.append(i_urls, ignore_index=True)
    break
similar_item_urls = list(urls['close_content_urls'])
taxons = []
for url in similar_item_urls:
    try:
        page = urllib.request.urlopen('https://www.gov.uk/api/content' + url)
        content_item = json.loads(page.read())
        taxons += list(map(lambda taxon: taxon['base_path'], content_item.get('links', {}).get('taxons', [])))
    except:
        print("Couldnt process: " + url)

taxons = list(set(taxons))
print("Suggested taxons (in order of relevancy)")
for taxon in taxons:
    print(taxon)