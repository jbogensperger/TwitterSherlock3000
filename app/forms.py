from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class QueryForm(FlaskForm):
    query = StringField('Enter Query here', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Analyze Tweets')