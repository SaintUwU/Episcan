"""
Forms for EpiScan authentication
"""
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectField, SubmitField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError
from app.models.data_models import User

class RegistrationForm(FlaskForm):
    """Registration form for creating new users"""
    full_name = StringField(
        'Full Name',
        validators=[
            DataRequired(message='Full name is required'),
            Length(min=2, max=100, message='Full name must be between 2 and 100 characters')
        ],
        render_kw={'class': 'form-control', 'placeholder': 'Enter full name'}
    )
    
    email = StringField(
        'Email',
        validators=[
            DataRequired(message='Email is required'),
            Email(message='Please enter a valid email address'),
            Length(max=120, message='Email must be less than 120 characters')
        ],
        render_kw={'class': 'form-control', 'placeholder': 'Enter email address'}
    )
    
    password = PasswordField(
        'Password',
        validators=[
            DataRequired(message='Password is required'),
            Length(min=6, message='Password must be at least 6 characters long')
        ],
        render_kw={'class': 'form-control', 'placeholder': 'Enter password'}
    )
    
    confirm_password = PasswordField(
        'Confirm Password',
        validators=[
            DataRequired(message='Please confirm your password'),
            EqualTo('password', message='Passwords must match')
        ],
        render_kw={'class': 'form-control', 'placeholder': 'Confirm password'}
    )
    
    role = SelectField(
        'Role',
        choices=[
            ('admin', 'Admin'),
            ('user', 'Medical Practitioner/User')
        ],
        validators=[DataRequired(message='Please select a role')],
        render_kw={'class': 'form-control'}
    )
    
    submit = SubmitField(
        'Register',
        render_kw={'class': 'btn btn-primary btn-block'}
    )
    
    def validate_email(self, email):
        """Check if email already exists"""
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email address already exists. Please choose a different one.')

class LoginForm(FlaskForm):
    """Login form for user authentication"""
    email = StringField(
        'Email',
        validators=[
            DataRequired(message='Email is required'),
            Email(message='Please enter a valid email address')
        ],
        render_kw={'class': 'form-control', 'placeholder': 'Enter email address'}
    )
    
    password = PasswordField(
        'Password',
        validators=[DataRequired(message='Password is required')],
        render_kw={'class': 'form-control', 'placeholder': 'Enter password'}
    )
    
    submit = SubmitField(
        'Login',
        render_kw={'class': 'btn btn-primary btn-block'}
    )
