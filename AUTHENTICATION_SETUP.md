# EpiScan Authentication System Setup

This document provides instructions for setting up and using the EpiScan authentication system.

## Features

- **Two User Roles**: Admin and Medical Practitioner/User
- **Secure Registration**: Only Admins can register new users
- **Password Hashing**: Uses bcrypt for secure password storage
- **Form Validation**: Server-side validation with Flask-WTF
- **CSRF Protection**: Built-in CSRF protection
- **Role-based Access Control**: Different permissions for different roles

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize Database**:
   ```bash
   python app.py init-db
   ```

3. **Create First Admin User**:
   ```bash
   python app.py create-admin
   ```
   This creates an admin user with:
   - Email: `admin@episcan.ke`
   - Password: `admin123`
   - **Important**: Change this password after first login!

## Usage

### Starting the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### User Roles

#### Admin
- Can access all system features
- Can register new users (both Admin and Medical Practitioner/User)
- Can view and manage all data

#### Medical Practitioner/User
- Can access the dashboard
- Can view outbreak predictions and alerts
- Cannot register new users

### Authentication Flow

1. **Login**: Visit `/login` or the root URL
2. **Registration**: Admins can visit `/register` to create new users
3. **Dashboard**: After login, users are redirected to the main dashboard
4. **Logout**: Click the logout button in the dashboard header

### API Endpoints

- `GET /` - Redirects to login page
- `GET /login` - Login page
- `POST /login` - Process login form
- `GET /register` - Registration page (Admin only)
- `POST /register` - Process registration form (Admin only)
- `GET /logout` - Logout user
- `GET /dashboard` - Main dashboard (requires login)

### Database Schema

The `users` table contains:
- `id` (Primary Key)
- `full_name` (String, 100 chars)
- `email` (String, 120 chars, unique)
- `password_hash` (String, 255 chars)
- `role` (String, 20 chars: 'Admin' or 'Medical Practitioner/User')
- `created_at` (DateTime)

### Security Features

- **Password Hashing**: Uses Werkzeug's secure password hashing
- **CSRF Protection**: All forms include CSRF tokens
- **Session Management**: Flask-Login handles user sessions
- **Role-based Access**: Decorators ensure proper access control
- **Input Validation**: Server-side validation for all form inputs

### Customization

#### Adding New Roles

1. Update the `role` choices in `app/forms.py`
2. Add role checking methods in the `User` model
3. Update the `admin_required` decorator if needed

#### Modifying Registration

1. Edit `app/forms.py` to add/remove form fields
2. Update `app/templates/register.html` for UI changes
3. Modify `app/routes/auth.py` for backend logic

#### Styling

- Templates use Bootstrap 5 for responsive design
- Custom CSS in template files for branding
- Easy to modify colors and layout

### Troubleshooting

#### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
2. **Database Errors**: Run `python app.py init-db` first
3. **Permission Errors**: Ensure the admin user is created
4. **CSRF Errors**: Check that forms include `{{ form.hidden_tag() }}`

#### Reset Admin Password

If you need to reset the admin password:

```python
from app import create_app, db
from app.models.data_models import User

app = create_app()
with app.app_context():
    admin = User.query.filter_by(email='admin@episcan.ke').first()
    if admin:
        admin.set_password('new_password')
        db.session.commit()
        print("Password updated!")
```

### Production Considerations

1. **Change Default Password**: Update the admin password immediately
2. **Environment Variables**: Use environment variables for secret keys
3. **HTTPS**: Enable HTTPS in production
4. **Database**: Use PostgreSQL or MySQL for production
5. **Session Security**: Configure secure session cookies

### Example Usage

```python
# Check if user is admin
if current_user.is_admin():
    # Admin-only code here
    pass

# Require login for a route
@login_required
def protected_route():
    return "This requires login"

# Require admin role
@admin_required
def admin_only_route():
    return "This requires admin role"
```

## Support

For issues or questions about the authentication system, please check the main EpiScan documentation or create an issue in the project repository.


