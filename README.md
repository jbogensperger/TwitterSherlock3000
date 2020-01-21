# TwitterSherlock3000
Twitter sentiment analysis application for custom queries (including a Pytorch BERT model)

SET UP:

Please download the model from the most recent repository release and place it in "models/" so it is reachable for the program e.g. "models/pytorch_BERT-20191219_06-02-07"

Provide a "twitter_credentials.json" directly in the main folder or use the "credential_writer.py" to create your "twitter_credentials.json" with your personal credential for the twitter API.

After installing all components from the "requirements.txt" you can start the webapp via running the "TwitterAnalyzerWebApp.py" file.


USAGE GUIDE:

Place your query in the query page and press "Analyze Tweets". 

The programm will check all matching tweets (in standard mode without paying only 100..) for a positive or negative sentiment.

The final statistics present:
    present Hashtags
    Amount of positive tweets containing this hashtag
    Amount of likes summed up for all positive tweets containing this hashtag
    Amount of negative tweets containing this hashtag
    Amount of likes summed up for all negative tweets containing this hashtag

