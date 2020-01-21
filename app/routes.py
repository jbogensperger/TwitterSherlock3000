import json

import pandas as pd
from flask import render_template, url_for, session
from pandas import DataFrame
from flask import request
from app import app
from app.forms import LoginForm, QueryForm
from twitterServant import TwitterAnalyzer, preProcessDataAndExtractHashtags
from flask import g
from flask import render_template, flash, redirect



@app.route('/', methods=['GET', 'POST'])
@app.route('/query', methods=['GET', 'POST'])
def query():
    form = QueryForm()
    if form.validate_on_submit():
        # flash('Search phrase "{}" is going to be analyzed.'.format(form.query.data))
        query=form.query.data
        session['query'] = query=form.query.data
        return redirect(url_for('statistics'))
    return render_template('query.html', title='Analyze Tweets', form=form)


@app.route('/statistics', methods=['GET', 'POST'])
def statistics():
    # query = 'IRAN' #request.args.get('query')
    query = session.get('query', None)

    twitterGuru = TwitterAnalyzer.getInstance()
    df_hashTagsStats = twitterGuru.get_twitter_statistics(query)

    pd.set_option('colheader_justify', 'center')

    html_table = df_hashTagsStats.to_html(index=False, justify='center', classes='table')
    return render_template("statistics.html", name='Tweet Statistics', title='Stats',
                           data=html_table)
