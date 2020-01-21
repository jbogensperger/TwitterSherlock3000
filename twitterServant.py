import json
import os
import re
import torch
from pandas import DataFrame
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification, WEIGHTS_NAME
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np

import pandas as pd
from twython import Twython


class AuthError(Exception):
    pass

class ModelError(Exception):
    pass

modelfilePath='models/pytorch_BERT-20191219_06-02-07/'
output_model_file = os.path.join(modelfilePath, WEIGHTS_NAME)
MAX_LEN = 35 #NEEDS TO BE THE SAME AS IN THE MODEL!! --> don't change






def preProcessDataAndExtractHashtags(tweetList):
    tweetList = tweetList.str.replace(r'[.]+', '.', regex=True)
    tweetList = tweetList.str.replace(r'[?]+', '?', regex=True)
    tweetList = tweetList.str.replace(r'[!]+', '!', regex=True)
    #  all single numbers, but leave things like 8am
    tweetList = tweetList.str.replace(r' [123456789]+ ', ' ')

    # twitter handles
    tweetList = tweetList.str.replace(r'@[^\s]+', ' ')
    # Remove textual smileys apart so there are no single "D" left over from ":D"
    tweetList = tweetList.str.replace(':D', ' ')
    tweetList = tweetList.str.replace(':P', ' ')
    tweetList = tweetList.str.replace(':p', ' ')
    tweetList = tweetList.str.replace(':d', ' ')

    # links
    tweetList = tweetList.str.replace(r'http\S+', ' ', case=False)
    tweetList = tweetList.str.replace(r'www.\S+', ' ', case=False)
    # And the links where they forgot a space to separate them..
    tweetList = tweetList.str.replace(r'\S+http\S+', ' ', case=False)
    tweetList = tweetList.str.replace(r'\S+http\S+', ' ', case=False)

    #Extract Hashtags
    hashTagList = []
    for tweet in tweetList:
        hashTags = re.findall(r"#(\w+)", tweet)
        hashTagList.append(hashTags)

    # all hashtags but not the words after it and the rests of special chars
    tweetList = tweetList.str.replace('[@#\"\(\)=:;]', ' ')

    # remove special chars
    tweetList = tweetList.str.replace(r'[-%&;§=*~#]+', '', regex=True)
    tweetList = tweetList.str.strip()

    # in the end remove double spaces
    tweetList = tweetList.str.replace(r'[ ]+', ' ', regex=True)
    tweetList = tweetList.str.replace('  ', ' ', regex=True)

    return tweetList, hashTagList


class TwitterAnalyzer:
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if TwitterAnalyzer.__instance == None:
            TwitterAnalyzer()
        return TwitterAnalyzer.__instance


    def __init__(self):
        credentialFilePath = 'twitter_credentials.json'
        # TODO add that file to .gitignore
        with open(credentialFilePath, "r") as file:
            creds = json.load(file)

        self.twitter = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'],
                               creds['ACCESS_TOKEN'],
                               creds['ACCESS_SECRET'])
        verification = self.twitter.verify_credentials()
        print('Login Verification:' + verification['name'])

        model_state_dict = torch.load(output_model_file)
        self.loaded_model = BertForSequenceClassification.from_pretrained(modelfilePath,
                                                                          state_dict=model_state_dict,
                                                                          num_labels=2)
        self.loaded_model.cuda()

        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
            raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.get_device_name(0)

        if self.loaded_model is None:
            raise ModelError("Sequence Classification Model was not properly loaded!!")

        print("Model loaded")

        if TwitterAnalyzer.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            TwitterAnalyzer.__instance = self

    def search(self, searchstring):
        if self.twitter is None:  # or not self.twitter.verify_credentials():
            raise AuthError('Twitter API was not properly connected!')

        cleanedSearchStr = re.sub('[^A-Za-z0-9 ]+', '', searchstring)

        dict_ = {'user': [], 'date': [], 'text': [], 'favorite_count': []}
        # Attention twitter API used with result_type='popular' only delivers a maximum of 15 tweets... --> therefore only recent/mixed makes sense..
        for status in \
        self.twitter.search(q=cleanedSearchStr, count=100, include_rts=0, lang='en', result_type='mixed')[
            'statuses']:
            dict_['user'].append(status['user']['screen_name'])
            dict_['date'].append(status['created_at'])
            dict_['text'].append(status['text'])
            dict_['favorite_count'].append(status['favorite_count'])

        tweets = pd.DataFrame(dict_)
        tweets.sort_values(by='favorite_count', inplace=True, ascending=False)
        return tweets

    def predictTweetSentiments(self, tweettextList):

        sentences = ["[CLS] " + str(query) + " [SEP]" for query in tweettextList]
        # labels = test_y
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

        # input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
        #                           maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        attention_masks = []

        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

        prediction_inputs = torch.tensor(input_ids, dtype=torch.long)
        prediction_masks = torch.tensor(attention_masks, dtype=torch.long)
        # prediction_labels = torch.tensor(labels)
        batch_size = 32
        prediction_data = TensorDataset(prediction_inputs, prediction_masks)  # prediction_labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

        self.loaded_model.eval()
        predictions, true_labels = [], []
        for batch in prediction_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask = batch
            with torch.no_grad():
                logits = self.loaded_model(b_input_ids, attention_mask=b_input_mask)
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)

        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

        return flat_predictions


    def get_twitter_statistics(self, query):
        tweets = self.search(query)
        tweets['text'], tweets['hashTags'] = preProcessDataAndExtractHashtags(tweets['text'])
        tweets['sentiment'] = self.predictTweetSentiments(tweets['text'])

        hashtagStats = {}
        for index, row in tweets.iterrows():
            for hashtag in row['hashTags']:
                if hashtag not in hashtagStats.keys():
                    hashtagStats[hashtag] = [0, 0, 0, 0]
                hashtagStats[hashtag][row['sentiment']] = hashtagStats[hashtag][row['sentiment']] + 1
                hashtagStats[hashtag][row['sentiment'] + 2] = hashtagStats[hashtag][row['sentiment'] + 2] + row['favorite_count']



        df_hashTagsStats = DataFrame(list(hashtagStats.items()), columns=['Hashtag', 'Sentiment'])
        df1 = pd.concat([pd.DataFrame(df_hashTagsStats['Sentiment'].values.tolist()).add_prefix('Sentiment')], axis=1)
        df_hashTagsStats = pd.concat([df1, df_hashTagsStats.drop(['Sentiment'], axis=1)], axis=1)
        df_hashTagsStats.columns = ['#PositiveTweets', '#NegativeTweets', '#likes for positive tweet',
                                    '#likes for negative tweet', 'Hashtag']
        df_hashTagsStats = df_hashTagsStats[['Hashtag', '#PositiveTweets', '#likes for positive tweet',
                                             '#NegativeTweets', '#likes for negative tweet']]

        return df_hashTagsStats








# #### Possible problem solution?
# I would suggest you to check the input type I had the same issue which solved by converting the input type from int32 to int64.(running on win10) ex:
#
# x = torch.tensor(train).to(torch.int64)






