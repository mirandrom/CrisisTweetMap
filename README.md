# CrisisTweetMap
Using Natural Language Processing to categorize and map tweets in real-time during crises.


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
- [mordecai](https://github.com/openeventdata/mordecai/tree/master/mordecai) for elasticsearch gazetteer
- [dash uber app](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-uber-rides-demo) for dash frontend