import os


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or '19249872340103845'
