# CrisisTweetMap
[Using Natural Language Processing to categorize and map tweets in real-time during the covid-19 crisis.](https://devpost.com/software/crisistweetmap-txahf2)  

![Crisis Tweet Map](https://raw.githubusercontent.com/amr-amr/CrisisTweetMap/master/doc/output.gif) 

Tweets are classified among the following categories:
```
affected_people
other_useful_information
disease_transmission
disease_signs_or_symptoms
prevention
treatment
not_related_or_irrelevant
deaths_reports
```

# To run the app
1. Download [Trained AllenNLP Model](https://drive.google.com/file/d/1NVJknCSK_Gk6-1xORJ35TgOMIsOBdEtS/view?usp=sharing)
to `/tweet_classifier/saved_models/bert_classification/`

2. Setup up  `tweepy_auth.json` with Twitter API keys

3. Download and run ElasticSearch geonames gazetteer container
    ```
    docker pull elasticsearch:5.5.2
    wget https://s3.amazonaws.com/ahalterman-geo/geonames_index.tar.gz --output-file=wget_log.txt
    tar -xzf geonames_index.tar.gz
    docker run -d -p 127.0.0.1:9200:9200 -v $(pwd)/geonames_index/:/usr/share/elasticsearch/data elasticsearch:5.5.2
    ```

4. Install requirements  
    `pip install -r requirements.txt`

5. Run live twitter scraper/classifier  
    `python stream_twitter.py`

6. Run live dashboard  
    `python app.py`
    
    
# Shoutouts
These following repositories made my life much easier with working examples of the different components I needed for this project.
- [smacawi](https://github.com/smacawi/tweet-classifier) for tweet scraping / classification
- [CrisisNLP](https://crisisnlp.qcri.org/) for the training data
- [mordecai](https://github.com/openeventdata/mordecai/tree/master/mordecai) for elasticsearch gazetteer
- [dash uber app](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-uber-rides-demo) for dash frontend

# TODO:
- Implement pubsub to fix ProtocolError handling (caused by pipeline lagging behind tweet stream) and OperationalError handling (caused by database locking);
- Fix dash app to play nice with multiple sessions;
- Fix spacy NER for location extraction or find faster alternative;
- Fix geolocation for smarter decision instead of greedy choice;
- Add tweet feed for chronological tweet visualization;
- Separate user/topic/message geolocation (e.g. user from Toronto mentionning China while in England);
- Add clickData event for mapbox plot;
- Add custom scraping search queries for users + historic search API integration to populate past x tweets;
- ... 
