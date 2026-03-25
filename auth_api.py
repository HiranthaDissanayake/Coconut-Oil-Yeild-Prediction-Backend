"""
Coconut Oil Prediction - Authentication API
Separate authentication server with JWT tokens and MySQL database
Port: 5001 (different from main prediction API port 5000)
"""

import os
import jwt
import bcrypt
import mysql.connector
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# ══════════════════════════════════════════════════════════════
#  APP SETUP
# ══════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# JWT Secret Key (CHANGE THIS IN PRODUCTION!)
app.config['SECRET_KEY'] = 'YOUR-SUPER-SECRET-KEY-CHANGE-IN-PRODUCTION'
app.config['JWT_EXPIRATION_HOURS'] = 24  # Token expires in 24 hours

# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  MYSQL DATABASE CONNECTION
# ══════════════════════════════════════════════════════════════

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',              # CHANGE THIS
    'password': '12345',  # CHANGE THIS
    'database': 'coconut_oil_db',
    'auth_plugin': 'mysql_native_password'
}

def get_db_connection():
    """Get MySQL database connection"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        log.error(f"Database connection error: {err}")
        return None


# ══════════════════════════════════════════════════════════════
#  PASSWORD HASHING
# ══════════════════════════════════════════════════════════════

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against hashed password"""
    return bcrypt.checkpw(
        password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )


# ══════════════════════════════════════════════════════════════
#  JWT TOKEN MANAGEMENT
# ══════════════════════════════════════════════════════════════

def create_token(user_id: int, email: str) -> str:
    """Create JWT token"""
    payload = {
        'user_id': user_id,
        'email': email,
        'exp': datetime.utcnow() + timedelta(hours=app.config['JWT_EXPIRATION_HOURS']),
        'iat': datetime.utcnow()
    }
    token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
    return token


def decode_token(token: str) -> dict:
    """Decode and verify JWT token"""
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return {'valid': True, 'data': payload}
    except jwt.ExpiredSignatureError:
        return {'valid': False, 'error': 'Token has expired'}
    except jwt.InvalidTokenError:
        return {'valid': False, 'error': 'Invalid token'}


# ══════════════════════════════════════════════════════════════
#  AUTH DECORATOR (for protected routes)
# ══════════════════════════════════════════════════════════════

def token_required(f):
    """Decorator to protect routes with JWT authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Get token from Authorization header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                return jsonify({'message': 'Token format invalid'}), 401

        if not token:
            return jsonify({'message': 'Token is missing'}), 401

        # Verify token
        result = decode_token(token)
        if not result['valid']:
            return jsonify({'message': result['error']}), 401

        # Add user data to request
        request.current_user = result['data']
        return f(*args, **kwargs)

    return decorated


# ══════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.route('/api/auth/health', methods=['GET'])
def health():
    """Health check endpoint"""
    conn = get_db_connection()
    db_status = 'connected' if conn else 'disconnected'
    if conn:
        conn.close()

    return jsonify({
        'status': 'ok',
        'service': 'coconut-oil-auth-api',
        'database': db_status,
        'version': '1.0'
    })


@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user"""
    try:
        data = request.get_json()

        # Validate input
        name = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not name or len(name) < 2:
            return jsonify({'message': 'Name must be at least 2 characters'}), 400

        if not email or '@' not in email:
            return jsonify({'message': 'Invalid email address'}), 400

        if not password or len(password) < 6:
            return jsonify({'message': 'Password must be at least 6 characters'}), 400

        # Connect to database
        conn = get_db_connection()
        if not conn:
            return jsonify({'message': 'Database connection failed'}), 500

        cursor = conn.cursor(dictionary=True)

        # Check if email already exists
        cursor.execute('SELECT id FROM users WHERE email = %s', (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            cursor.close()
            conn.close()
            return jsonify({'message': 'Email already registered'}), 409

        # Hash password
        hashed_password = hash_password(password)

        # Insert new user
        insert_query = """
            INSERT INTO users (name, email, password, created_at)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(insert_query, (name, email, hashed_password, datetime.now()))
        conn.commit()

        # Get new user ID
        user_id = cursor.lastrowid

        # Close database connection
        cursor.close()
        conn.close()

        # Create JWT token
        token = create_token(user_id, email)

        # Return success response
        log.info(f"New user registered: {email}")
        return jsonify({
            'message': 'Registration successful',
            'token': token,
            'user': {
                'id': user_id,
                'name': name,
                'email': email,
            }
        }), 201

    except Exception as e:
        log.error(f"Registration error: {e}")
        return jsonify({'message': f'Registration failed: {str(e)}'}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()

        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({'message': 'Email and password required'}), 400

        # Connect to database
        conn = get_db_connection()
        if not conn:
            return jsonify({'message': 'Database connection failed'}), 500

        cursor = conn.cursor(dictionary=True)

        # Get user by email
        cursor.execute(
            'SELECT id, name, email, password FROM users WHERE email = %s',
            (email,)
        )
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        # Check if user exists
        if not user:
            return jsonify({'message': 'Invalid email or password'}), 401

        # Verify password
        if not verify_password(password, user['password']):
            return jsonify({'message': 'Invalid email or password'}), 401

        # Create JWT token
        token = create_token(user['id'], user['email'])

        # Update last login
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE users SET last_login = %s WHERE id = %s',
                (datetime.now(), user['id'])
            )
            conn.commit()
            cursor.close()
            conn.close()

        # Return success response
        log.info(f"User logged in: {email}")
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user['id'],
                'name': user['name'],
                'email': user['email'],
            }
        }), 200

    except Exception as e:
        log.error(f"Login error: {e}")
        return jsonify({'message': f'Login failed: {str(e)}'}), 500


@app.route('/api/auth/verify', methods=['GET'])
@token_required
def verify_token_endpoint():
    """Verify if token is valid"""
    return jsonify({
        'message': 'Token is valid',
        'user': request.current_user
    }), 200


@app.route('/api/auth/profile', methods=['GET'])
@token_required
def get_profile():
    """Get user profile"""
    try:
        user_id = request.current_user['user_id']

        conn = get_db_connection()
        if not conn:
            return jsonify({'message': 'Database connection failed'}), 500

        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            'SELECT id, name, email, created_at, last_login FROM users WHERE id = %s',
            (user_id,)
        )
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if not user:
            return jsonify({'message': 'User not found'}), 404

        return jsonify({
            'user': {
                'id': user['id'],
                'name': user['name'],
                'email': user['email'],
                'created_at': user['created_at'].isoformat() if user['created_at'] else None,
                'last_login': user['last_login'].isoformat() if user['last_login'] else None,
            }
        }), 200

    except Exception as e:
        log.error(f"Get profile error: {e}")
        return jsonify({'message': f'Failed to get profile: {str(e)}'}), 500


@app.route('/api/auth/profile', methods=['PUT'])
@token_required
def update_profile():
    """Update user profile"""
    try:
        user_id = request.current_user['user_id']
        data = request.get_json()

        name = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()

        if not name and not email:
            return jsonify({'message': 'No data to update'}), 400

        conn = get_db_connection()
        if not conn:
            return jsonify({'message': 'Database connection failed'}), 500

        cursor = conn.cursor(dictionary=True)

        # Build update query dynamically
        updates = []
        params = []

        if name:
            updates.append('name = %s')
            params.append(name)

        if email:
            # Check if email already taken by another user
            cursor.execute(
                'SELECT id FROM users WHERE email = %s AND id != %s',
                (email, user_id)
            )
            if cursor.fetchone():
                cursor.close()
                conn.close()
                return jsonify({'message': 'Email already taken'}), 409

            updates.append('email = %s')
            params.append(email)

        params.append(user_id)

        # Execute update
        update_query = f"UPDATE users SET {', '.join(updates)} WHERE id = %s"
        cursor.execute(update_query, params)
        conn.commit()

        # Get updated user
        cursor.execute(
            'SELECT id, name, email FROM users WHERE id = %s',
            (user_id,)
        )
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        log.info(f"Profile updated for user ID: {user_id}")
        return jsonify({
            'message': 'Profile updated successfully',
            'user': user
        }), 200

    except Exception as e:
        log.error(f"Update profile error: {e}")
        return jsonify({'message': f'Failed to update profile: {str(e)}'}), 500


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  🔐 Coconut Oil Prediction - Authentication API")
    print("  Port: 5001 (separate from prediction API on port 5000)")
    print("  Endpoints:")
    print("    POST   /api/auth/register")
    print("    POST   /api/auth/login")
    print("    GET    /api/auth/verify")
    print("    GET    /api/auth/profile")
    print("    PUT    /api/auth/profile")
    print("="*70 + "\n")

    # Run on port 5001 (prediction API uses 5000)
    app.run(host='0.0.0.0', port=5001, debug=False)