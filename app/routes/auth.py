"""
Authentication routes for EpiScan
"""
from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app import db
from app.models.data_models import User
from app.forms import RegistrationForm, LoginForm
from functools import wraps

auth_bp = Blueprint('auth', __name__)

def admin_required(f):
    """Decorator to require admin role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin():
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if current_user.is_authenticated:
        # Redirect based on user role
        if current_user.is_admin():
            return redirect(url_for('admin.dashboard'))
        else:
            return redirect(url_for('user.dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=True)
            next_page = request.args.get('next')
            if not next_page or not next_page.startswith('/'):
                # Redirect based on user role
                if user.is_admin():
                    next_page = url_for('admin.dashboard')
                else:
                    next_page = url_for('user.dashboard')
            return redirect(next_page)
        else:
            flash('Invalid email or password. Please try again.', 'error')
    
    return render_template('login.html', form=form)

@auth_bp.route('/register', methods=['GET', 'POST'])
@login_required
@admin_required
def register():
    """User registration page (Admin only)"""
    form = RegistrationForm()
    if form.validate_on_submit():
        try:
            user = User(
                full_name=form.full_name.data,
                email=form.email.data,
                role=form.role.data
            )
            user.set_password(form.password.data)
            
            db.session.add(user)
            db.session.commit()
            
            flash(f'User {user.full_name} has been successfully registered!', 'success')
            return redirect(url_for('auth.login'))
            
        except Exception as e:
            db.session.rollback()
            flash('An error occurred while registering the user. Please try again.', 'error')
    
    return render_template('register.html', form=form)

@auth_bp.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('auth.login'))
