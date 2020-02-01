from pathlib import Path
import json
import dataset

import tweepy
from tweepy import OAuthHandler


class TweetScraper(object):
    # adapted from https://github.com/smacawi/twitter-scraper
    def __init__(self, auth_dict: dict, db: str, table: str):
        self.api = self._load_api(auth_dict)
        self.db_table = self._create_db(db, table)
        self.stream_listener = self._create_stream_listener()
        self.tweets_scraped = 0

    @classmethod
    def from_json(cls, json_path: str, kwargs):
        """Loads auth key/value pairs from json file.

        Parameters
        ----------
        json_path : str
            Path to auth json containing 'consumer_key', 'consumer_secret', 'access_token', 'access_token_secret'

        Returns
        -------
        TwitterScraper
        """
        with Path(json_path).open() as j:
            auth_dict = json.load(j)
        return cls(auth_dict, **kwargs)

    def _load_api(self, auth_dict: dict):
        """Validate auth keys/tokens and return tweepy api object.

        Parameters
        ----------
        auth_dict : dict
            A dictionary with 'consumer_key', 'consumer_secret', 'access_token', 'access_token_secret'.

        Returns
        -------
        tweepy.API
            API object if authorization is successful.

        """
        required_keys = ['consumer_key', 'consumer_secret', 'access_token', 'access_token_secret']
        assert all(k in auth_dict for k in required_keys), print(f"Required keys: {required_keys}")

        auth = OAuthHandler(auth_dict['consumer_key'], auth_dict['consumer_secret'])
        auth.set_access_token(auth_dict['access_token'], auth_dict['access_token_secret'])
        return tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    def _create_db(self, db, table):
        # TODO: ensure no issues with existing db/table
        db = dataset.connect(f"sqlite:///{db}.db")
        return db.create_table(table, primary_id=False)

    def _create_stream_listener(self):
        """Creates stream listener object from nested class so that stream listener can access `_parse_tweet` method.
        """
        return TwitterScraper.StreamListener(self)

    class StreamListener(tweepy.StreamListener):
        def __init__(self, twitter_scraper):
            super().__init__()
            self.twitter_scraper = twitter_scraper

        def on_status(self, tweet):
            self.twitter_scraper.db_table.insert(self.twitter_scraper.parse_tweet(tweet))
            if self.twitter_scraper.tweets_scraped % 100 == 0:
                print(self.twitter_scraper.tweets_scraped, tweet.text)
            self.twitter_scraper.tweets_scraped += 1

        def on_error(self, status_code):
            # TODO: complete error handling
            if status_code == 420:
                return False

    def parse_tweet(self, tweet: tweepy.Status):
        """Parses relevant information from tweet object.

        For reference, with regards to tweet object:
        https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object.html

        Parameters
        ----------
        tweet : tweepy.Status
            Tweet object, response from Twitter API.

        Returns
        -------
        parsed_tweet : dict
            Dictionary with relevant information from tweet object.
        """
        tweet = tweet._json
        if 'extended_tweet' in tweet:
            text = tweet['extended_tweet']['full_text']
        else:
            text = tweet['text']

        # retweet information
        if 'retweeted_status' in tweet:
            rt = tweet['retweeted_status']
            rt_extended = 'extended_tweet' in rt
            rt_text = rt['text'] if not rt_extended else rt['extended_tweet']['full_text']
            rt_user_id = rt['user']['id']
            rt_user = rt['user']['screen_name']
            rt_id = rt['id']
        else:
            rt_text = ""
            rt_user_id = ""
            rt_user = ""
            rt_id = ""

        # quote information
        if 'quoted_status' in tweet:
            qt = tweet['quoted_status']
            qt_extended = 'extended_tweet' in qt
            qt_text = qt['text'] if not qt_extended else qt['extended_tweet']['full_text']
            qt_user_id = qt['user']['id']
            qt_user = qt['user']['screen_name']
            qt_id = qt['id']
        else:
            qt_text = ""
            qt_user_id = ""
            qt_user = ""
            qt_id = ""

        # reply information
        if tweet['in_reply_to_status_id'] is not None:
            reply_to_tweet_id = tweet['in_reply_to_status_id']
            reply_to_user = tweet['in_reply_to_screen_name']
            reply_to_user_id = tweet['in_reply_to_user_id_str']
            try:
                reply_to_tweet_text = self.api.get_status(int(reply_to_tweet_id)).text
            except Exception as e:
                print(e)
                reply_to_tweet_text = ""
        else:
            reply_to_tweet_id = ""
            reply_to_user = ""
            reply_to_user_id = ""
            reply_to_tweet_text = ""

        # network information
        if qt_id:
            source_id = qt_id
            source_user = qt_user
            source_user_id = qt_user_id
            source_text = qt_text
            edge_type = "quote"
        elif rt_id:
            source_id = rt_id
            source_user = rt_user
            source_user_id = rt_user_id
            source_text = rt_text
            edge_type = "retweet"
        elif reply_to_tweet_id:
            source_id = reply_to_tweet_id
            source_user = reply_to_user
            source_user_id = reply_to_user_id
            source_text = reply_to_tweet_text
            edge_type = "reply"
        else:
            source_id = ""
            source_user = ""
            source_user_id = ""
            source_text = ""
            edge_type = ""

        parsed_tweet = dict(
            # tweet information
            id=tweet['id_str'],
            text=text,
            hashtags=",".join([ht['text'] for ht in tweet['entities']['hashtags']]),
            user_mentions_id=",".join([um['id_str'] for um in tweet['entities']['user_mentions']]),
            user_mentions=",".join([um['screen_name'] for um in tweet['entities']['user_mentions']]),
            # user information
            user_id=tweet['user']['id_str'],
            user=tweet['user']['screen_name'],
            user_name=tweet['user']['name'],
            user_followers_count=tweet['user']['followers_count'],
            user_friends_count=tweet['user']['friends_count'],
            user_favourites_count=tweet['user']['favourites_count'],
            user_statuses_count=tweet['user']['statuses_count'],
            user_listed_count=tweet['user']['listed_count'],
            user_location=tweet['user']['location'],
            user_verified=tweet['user']['verified'],
            user_created_at=tweet['user']['created_at'],
            # reply information
            reply_to_tweet_id=reply_to_tweet_id,
            reply_to_user=reply_to_user,
            reply_to_user_id=reply_to_user_id,
            reply_to_tweet_text=reply_to_tweet_text,
            # retweet information
            rt_text=rt_text,
            rt_id=rt_id,
            rt_user=rt_user,
            rt_user_id=rt_user_id,
            # quote information
            qt_text=qt_text,
            qt_user_id=qt_user_id,
            qt_user=qt_user,
            qt_id=qt_id,
            # time/location information
            created_at=tweet['created_at'],
            coordinates=str(tweet['coordinates']),
            place=str(tweet['place']),
            # network information
            source_id=source_id,
            source_user=source_user,
            source_user_id=source_user_id,
            source_text=source_text,
            edge_type=edge_type,
        )
        return parsed_tweet

    def search_tweets_historic(self, search_kwargs):
        """Searches historic tweets using Twitter API search endpoint.

        For reference, with regard to search endpoint:
        https://developer.twitter.com/en/docs/tweets/search/guides/standard-operators

        Parameters
        ----------
        kwargs :
            Any of
                'q', lang', 'locale', 'since_id', 'geocode',
                'max_id', 'since', 'until', 'result_type',
                'count', 'include_entities', 'from',
                'to', 'source'
        Returns
        -------
        tweets : dict[list]
            List of parsed tweets as dictionaries
        """
        tweet_count = 0
        try:
            print(search_kwargs)
            for tweet in tweepy.Cursor(self.api.search, **search_kwargs).items():
                parsed_tweet = self.parse_tweet(tweet)
                self.db_table.insert(parsed_tweet)
                tweet_count += 1
                if tweet_count % 100 == 0:
                    print(tweet_count, parsed_tweet["created_at"], parsed_tweet["text"])
        except tweepy.TweepError as e:
            print("Tweepy error : " + str(e))

    def search_tweets_live(self, search_kwargs):
        """ Uses Twitter API search endpoint for near-live scraping

        One of the advantages of this over the streaming endpoint is greater
        filtering/searching capabilities.

        TODO: implement
        """
        raise NotImplemented

    def stream_tweets(self, tw_filter):
        stream = tweepy.Stream(self.api.auth, self.stream_listener)
        stream.filter(**tw_filter)