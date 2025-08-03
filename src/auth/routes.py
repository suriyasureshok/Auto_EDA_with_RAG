from fastapi import APIRouter, HTTPException
from firebase_admin import auth as firebase_auth
import pyrebase
import os
from dotenv import load_dotenv

from utils.models import RegisterUser, LoginUser
from .auth_service import create_jwt
from utils.logging import get_logger

load_dotenv()
router = APIRouter(prefix="/auth", tags=["auth"])
logger = get_logger(__name__)

# Load Firebase config from environment variables or .env
firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MSG_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID")
}

firebase = pyrebase.initialize_app(firebase_config)
pyre_auth = firebase.auth()


@router.post("/register")
def register_user(user: RegisterUser):
    """
    Register a new user in Firebase Authentication.

    Args:
        user (RegisterUser): The user registration data.

    Returns:
        dict: Firebase UID and success message.
    """
    try:
        user_record = firebase_auth.create_user(
            email=user.email,
            password=user.password,
            display_name=user.username
        )
        logger.info(f"User registered: {user_record.uid}")
        return {"uid": user_record.uid, "message": "User registered successfully"}
    except Exception as e:
        logger.warning(f"Registration failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")


@router.post("/login")
def login_user(user: LoginUser):
    """
    Authenticate user and return JWT token.

    Args:
        user (LoginUser): The user login credentials.

    Returns:
        dict: Access token and token type.
    """
    try:
        firebase_user = pyre_auth.sign_in_with_email_and_password(user.email, user.password)
        uid = firebase_auth.get_user_by_email(user.email).uid
        token = create_jwt(uid)
        logger.info(f"User logged in: {uid}")
        return {"access_token": token, "token_type": "bearer"}
    except Exception as e:
        logger.warning(f"Login failed for {user.email}: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Login failed: {str(e)}")
