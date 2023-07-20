from flask import Flask, render_template, url_for, redirect, flash, get_flashed_messages
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, SelectField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from datetime import datetime
import pytz
from wtforms import validators
from flask import flash, request, redirect, url_for, render_template
from joblib import load
import urllib.request
import pickle
from skimage.io import imread
import os
from PIL import Image
from skimage.transform import resize
from werkzeug.utils import secure_filename
from skimage.feature import local_binary_pattern
import cv2
import pandas as pd
import numpy as np
from flask import send_from_directory
from sqlalchemy import inspect




app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///E:/flask/database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'

# Load the machine learning model
model = load('lbp_model.p')

UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secretkey"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

bcrypt = Bcrypt(app)
db = SQLAlchemy(app)


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
#Creating a User table
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    password = db.Column(db.String(80), nullable=False)
    confirm_password = db.Column(db.String(80), nullable=False)
    security_question = db.Column(db.String(100), nullable=False)  
    security_answer = db.Column(db.String(100), nullable=False)  
    registration_date = db.Column(db.DateTime(timezone=True), nullable=False, default=datetime.utcnow().replace(microsecond=0))

    
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

admin = Admin(app)

class UserView(ModelView):
    column_list = ('username', 'first_name', 'last_name', 'security_question', 'security_answer', 'registration_date')
    form_columns = ('username', 'first_name', 'last_name', 'security_question', 'security_answer', 'password', 'confirm_password', 'registration_date')
    column_searchable_list = ['username']
    column_filters = ['username']
    def _format_registration_date(self, context, model, name):
        registration_date = getattr(model, name)
        if registration_date:
            utc_tz = pytz.timezone('UTC')  # Assuming the registration date is stored in UTC
            pakistan_tz = pytz.timezone('Asia/Karachi')
            registration_date = utc_tz.localize(registration_date).astimezone(pakistan_tz)
            return registration_date.strftime('%Y-%m-%d %H:%M:%S')
        return ''

    column_formatters = {
        'registration_date': _format_registration_date
    }
admin.add_view(UserView(User, db.session))


# FlaskForms and Routes

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[
                           InputRequired(), Length(min=4, max=20)])
    first_name = StringField('First Name', validators=[
                            InputRequired(), Length(min=2, max=50)])
    last_name = StringField('Last Name', validators=[
                           InputRequired(), Length(min=2, max=50)])
    password = PasswordField('Password', validators=[
                             InputRequired(), Length(min=8, max=20)])
    confirm_password = PasswordField('Confirm Password', validators=[
                                     InputRequired(), Length(min=8, max=20)])
    security_question = SelectField('Security Question', choices=[
                                    ('', 'Select a security question'),  # Add an empty choice
                                    ('1', 'What is your favorite color?'),
                                    ('2', 'What is your pet\'s name?'),
                                    ('3', 'What city were you born in?'),
                                    ('4', 'What is your mother\'s maiden name?'),
                                    ('5', 'What is the name of your high school?')],
                                    validators=[InputRequired()], default='')
    security_answer = StringField('Security Answer', validators=[
                                  InputRequired(), Length(min=2, max=50)])
    submit = SubmitField('Register')

#Validation for Username and @ symbol
    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')
        if '@' not in username.data:
            raise ValidationError('Username must contain the @ symbol.')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[
                           InputRequired(), Length(min=4, max=20)])
    password = PasswordField('Password', validators=[
                             InputRequired(), Length(min=8, max=20)])
    submit = SubmitField('Login')

class ResetPasswordForm(FlaskForm):
    username = StringField('Username', validators=[
                           InputRequired(), Length(min=4, max=20)])
    security_answer = StringField('Security Answer', validators=[
                                  InputRequired(), Length(min=2, max=100)])
    new_password = PasswordField('New Password', validators=[
                                 InputRequired(), Length(min=8, max=20)])
    confirm_password = PasswordField('Confirm Password', validators=[
                                     InputRequired(), Length(min=8, max=20)])
    security_question = SelectField('Security Question', choices=[
                                    ('', 'Select a security question'),  # Add an empty choice
                                    ('1', 'What is your favorite color?'),
                                    ('2', 'What is your pet\'s name?'),
                                    ('3', 'What city were you born in?'),
                                    ('4', 'What is your mother\'s maiden name?'),
                                    ('5', 'What is the name of your high school?')],
                                    validators=[InputRequired()], default='')
# Validation for Security answer to be case insensitive
    def validate_security_answer(self, field):
        # Perform case-insensitive check for security answer
        if field.data.lower() != self.security_answer.data.lower():
            raise ValidationError('Invalid security answer.')


class EditProfileForm(FlaskForm):
    first_name = StringField('First Name', validators=[
                             InputRequired(), Length(min=2, max=50)])
    last_name = StringField('Last Name', validators=[
                            InputRequired(), Length(min=2, max=50)])
    password = PasswordField('New Password', validators=[
                             Length(min=8, max=20), validators.Optional()])
    confirm_password = PasswordField('Confirm New Password', validators=[
                                     Length(min=8, max=20), validators.EqualTo('password', message='Passwords must match.')])
    submit = SubmitField('Update Profile')

@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if request.method == 'POST':
        # Get the form data
        username = request.form['username']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validate the password and confirm_password
        if password != confirm_password:
            return "Password and Confirm Password do not match."
        
        # Update the user profile
        current_user.username = username
        current_user.first_name = first_name
        current_user.last_name = last_name
        current_user.password = password
        
        # Commit the changes to the database
        db.session.commit()
        
        # Redirect to the profile page
        return redirect(url_for('profile'))
    
    # Render the edit profile page
    return render_template('edit_profile.html', current_user=current_user)

# Image prediction model
#upload1

def predict_adult(image_path):
    print('Image Path:', image_path)
    img = imread(image_path)
    print('Image Shape:', img.shape)
    img_resize = resize(img, (150, 150, 3))
    print('Resized Image Shape:', img_resize.shape)
    l = [img_resize.flatten()]
    print('Flattened Image Shape:', l[0].shape)
    ans = model.predict(l)
    probability = model.predict_proba(l)
    output = '{0:.{1}f}'.format(probability[0][1], 2)

    print('Prediction:', ans)
    print('Probability:', probability)

    if ans[0] == 0:
        prediction = 'Uploaded Image is an Adult Image'
    elif ans[0] == 1:
        prediction = 'Uploaded Image is a Sensitive Image'
    else:
        prediction = 'Uploaded Image is a Non-Adult Image'

    print('Final Prediction:', prediction)
    return prediction


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')


@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.username.data == 'admin' and form.password.data == 'adminpassword':
            # Redirect to the admin dashboard
            return redirect(url_for('admin.index'))

        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('home'))

        flash('Invalid username or password.', 'error')

    flash_messages = get_flashed_messages()
    
    return render_template('login.html', form=form, flash_messages=flash_messages)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')  # Add flash message
    return redirect(url_for('home'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        if form.password.data != form.confirm_password.data:
            flash('Passwords must match.', 'error')
            flash_messages = get_flashed_messages()
            return redirect(url_for('register'))

        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        new_user = User(
            username=form.username.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            password=hashed_password,
            confirm_password=form.confirm_password.data,
            security_question=form.security_question.data,  # Add the security question field
            security_answer=form.security_answer.data  # Add the security answer field
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.security_question == form.security_question.data and user.security_answer.lower() == form.security_answer.data.lower():
            user.password = bcrypt.generate_password_hash(form.new_password.data).decode('utf-8')
            db.session.commit()
            flash('Password reset successful. You can now log in with your new password.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Invalid username or security answer.', 'error')
    
    # Get the user's security question and set it in the form data
    user = User.query.filter_by(username=form.username.data).first()
    if user:
        form.security_question.data = user.security_question
    
    return render_template('forgot_password.html', form=form)



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Call the predict_adult() function with the file path
            prediction = predict_adult(file_path)
            flash(prediction, 'success')

            # Pass the prediction to the template
            return render_template('upload.html', filename=filename, prediction=prediction)

        flash('Invalid file format. Please upload an image with extensions: png, jpg, jpeg, or gif.', 'error')

    flash_messages = get_flashed_messages()
    return render_template('upload.html', flash_messages=flash_messages)

@app.route('/display/<filename>')
def display_image(filename):
    # Logic to display the image
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

#User Dashboard Functionalities
#FAQ
@app.route('/faq')
def faq():
    return render_template('faq.html')

#Upload2.html
@app.route('/upload2', methods=['GET', 'POST'])
def upload2():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            flash('Image successfully uploaded')

            prediction = predict_adult2(filepath)

            return render_template('upload2.html', filename=filename, prediction=prediction)

        else:
            flash('Invalid file format. Allowed formats are png, jpg, jpeg, gif')
            return redirect(request.url)

    return render_template('upload2.html')


lbp_model = pickle.load(open("lbp_model.p", "rb"))

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def extract_lbp_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= sum(lbp_hist)
    return lbp_hist.astype(np.float32)


def predict_adult2(imagepath):
    label_encoding = {0: 'Adult', 1: 'Non_Adult', 2: 'Sensitive'}
    feature_columns = ['lbp_1', 'lbp_2', 'lbp_3', 'lbp_4', 'lbp_5', 'lbp_6', 'lbp_7', 'lbp_8', 'lbp_9', 'lbp_10']

    model = pickle.load(open('lbp_model.p', 'rb'))

    lbp_feature = extract_lbp_features(imagepath)
    lbp_feature = np.array(lbp_feature).reshape(1, -1)
    X = pd.DataFrame(lbp_feature, columns=feature_columns)

    prediction = model.predict(X)
    label = label_encoding[prediction[0]]

    probabilities = model.predict_proba(X)[0]
    percentage = [f'{p * 100:.2f}%' for p in probabilities]

    print(f"Predicted category: {label}")
    print(f"Category probabilities: {percentage}")
    img = imread(imagepath)

    if label == 'Adult':
        prediction = 'Uploaded Image is an Adult Image'
    elif label == 'Sensitive':
        prediction = 'Uploaded Image is a Sensitive Image'
    else:
        prediction= 'Uploaded Image is a Non-Adult Image'

    return prediction


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/display/<filename>')
def display_image2(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

#dashboard

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

#CNN Model Implementation



@app.route('/upload3')
def upload3():
    return render_template('upload3.html')





@app.route('/schema')
def view_schema():
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()

    schema = {}
    for table_name in tables:
        columns = inspector.get_columns(table_name)
        schema[table_name] = [{"name": column["name"], "type": str(column["type"])} for column in columns]

    return render_template('schema.html', schema=schema)

if __name__ == '__main__':
    app.run(debug=True)



















