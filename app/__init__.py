from flask import Flask
from config import Config
from twitterServant import TwitterAnalyzer
from flask_bootstrap import Bootstrap

app = Flask(__name__)
app.config.from_object(Config)
bootstrap = Bootstrap(app)

# @app.before_first_request
# def do_something_only_once():
twitterGuru = TwitterAnalyzer.getInstance()


from app import routes
