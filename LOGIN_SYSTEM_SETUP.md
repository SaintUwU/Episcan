# EpiScan Secure Login System Setup Guide

This guide provides complete instructions for setting up and using the EpiScan secure login system with role-based access control.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Database Setup

#### Option A: PostgreSQL (Recommended)
```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE episcan_db;
CREATE USER episcan_user WITH PASSWORD 'episcan_pass';
GRANT ALL PRIVILEGES ON DATABASE episcan_db TO episcan_user;
\q

# Set environment variables
export DATABASE_URL="postgresql://episcan_user:episcan_pass@localhost/episcan_db"
```

#### Option B: SQLite (Development)
```bash
# Set environment variable for SQLite
export DATABASE_URL="sqlite:///episcan.db"
```

### 3. Initialize Database
```bash
python app.py init-db
```

### 4. Create Test Users
```bash
python app.py seed-users
```

### 5. Start Application
```bash
python app.py
```

Visit `http://localhost:5000` to access the login system.

## 🔐 Login Credentials

After running the seed command, you'll have these test accounts:

### Admin User
- **Email**: `admin@episcan.ke`
- **Password**: `admin123`
- **Role**: Admin
- **Dashboard**: `/admin/dashboard`

### Medical Practitioner
- **Email**: `jane.mwangi@episcan.ke`
- **Password**: `user123`
- **Role**: User
- **Dashboard**: `/user/dashboard`

⚠️ **Important**: Change these passwords after first login!

## 🏗️ System Architecture

### User Roles
- **Admin**: Full system access, can register new users, manage all data
- **User (Medical Practitioner)**: Access to dashboard, view predictions and alerts

### Database Schema
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(128) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Route Structure
- `/` - Redirects to login
- `/login` - Login page (GET/POST)
- `/logout` - User logout
- `/register` - User registration (Admin only)
- `/admin/dashboard` - Admin dashboard
- `/user/dashboard` - User dashboard

## 🎨 Features

### Security Features
- ✅ Password hashing with bcrypt
- ✅ CSRF protection on all forms
- ✅ Session management with Flask-Login
- ✅ Role-based access control
- ✅ Input validation and sanitization
- ✅ Secure redirects

### UI Features
- ✅ Bootstrap 5 responsive design
- ✅ Professional branding
- ✅ Role-based dashboards
- ✅ Real-time data visualization
- ✅ Mobile-friendly interface
- ✅ Forgot password placeholder

### Admin Dashboard Features
- User management
- System statistics
- Data monitoring
- User registration
- Comprehensive analytics

### User Dashboard Features
- Disease outbreak predictions
- Active alerts
- County-wise data
- Trend analysis
- Health monitoring tools

## 📁 File Structure

```
app/
├── __init__.py              # App factory with Flask-Login setup
├── models/
│   └── data_models.py       # User model with authentication methods
├── routes/
│   ├── auth.py              # Authentication routes (login, logout, register)
│   ├── admin.py             # Admin dashboard routes
│   ├── user.py              # User dashboard routes
│   ├── api.py               # API endpoints
│   └── dashboard.py         # Legacy dashboard routes
├── forms.py                 # WTForms for validation
└── templates/
    ├── login.html           # Login page with forgot password
    ├── register.html        # User registration form
    ├── admin_dashboard.html # Admin dashboard
    └── user_dashboard.html  # User dashboard
```

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://episcan_user:episcan_pass@localhost/episcan_db

# Flask
SECRET_KEY=your-secret-key-here
FLASK_ENV=development
FLASK_DEBUG=True
```

### Database Configuration
The system supports both PostgreSQL and SQLite:
- **PostgreSQL**: Recommended for production
- **SQLite**: Suitable for development and testing

## 🚀 Deployment

### Production Checklist
1. ✅ Change default passwords
2. ✅ Use strong SECRET_KEY
3. ✅ Enable HTTPS
4. ✅ Use PostgreSQL database
5. ✅ Set up proper environment variables
6. ✅ Configure reverse proxy (nginx)
7. ✅ Set up SSL certificates
8. ✅ Enable database backups

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 🧪 Testing

### Manual Testing
1. **Login Test**: Try logging in with both admin and user accounts
2. **Role Test**: Verify admin can access `/admin/dashboard` and user can access `/user/dashboard`
3. **Registration Test**: Admin should be able to register new users
4. **Logout Test**: Verify logout functionality works correctly
5. **Security Test**: Try accessing protected routes without login

### Automated Testing
```bash
# Run tests (if test suite is available)
python -m pytest tests/
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Database Connection Error
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check database exists
psql -U episcan_user -d episcan_db -c "\dt"
```

#### 2. Import Errors
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 3. Permission Denied
```bash
# Check file permissions
chmod +x seed_users.py
chmod +x create_admin.py
```

#### 4. Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000

# Kill process
kill -9 <PID>
```

### Reset Database
```bash
# Drop and recreate database
dropdb episcan_db
createdb episcan_db
python app.py init-db
python app.py seed-users
```

## 📚 API Documentation

### Authentication Endpoints

#### POST /login
Login with email and password
```json
{
    "email": "admin@episcan.ke",
    "password": "admin123"
}
```

#### GET /logout
Logout current user

#### GET /register (Admin only)
Access registration form

#### POST /register (Admin only)
Register new user
```json
{
    "full_name": "Dr. John Doe",
    "email": "john.doe@episcan.ke",
    "password": "secure_password",
    "confirm_password": "secure_password",
    "role": "user"
}
```

### Dashboard Endpoints

#### GET /admin/dashboard
Admin dashboard with user management

#### GET /user/dashboard
User dashboard with health data

## 🔒 Security Best Practices

1. **Password Policy**: Implement strong password requirements
2. **Session Timeout**: Set appropriate session timeouts
3. **Rate Limiting**: Implement login attempt limiting
4. **Audit Logging**: Log all authentication events
5. **Regular Updates**: Keep dependencies updated
6. **Environment Security**: Use environment variables for secrets

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the logs for error messages
3. Ensure all dependencies are correctly installed
4. Verify database connectivity
5. Check environment variable configuration

## 🎯 Next Steps

1. **Password Reset**: Implement forgot password functionality
2. **Email Verification**: Add email verification for new users
3. **Two-Factor Authentication**: Add 2FA for enhanced security
4. **User Profile**: Add user profile management
5. **Audit Trail**: Implement comprehensive audit logging
6. **API Keys**: Add API key authentication for external access

---

**EpiScan Login System** - Secure, Role-Based Authentication for Disease Surveillance


