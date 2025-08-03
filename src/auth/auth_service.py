"""
JWT Token generation and verification.

This module provides functions to generate and verify JWT tokens.
"""

import datetime
import os
from typing import Optional
from dotenv import load_dotenv
from jose import JWTError, ExpiredSignatureError, jwt

from utils.logging import get_logger
from utils.exceptions import JWTExpiredError, JWTInvalidError, JWTVerificationError, JWTCreationError

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")

logger = get_logger(__name__)

def create_jwt(uid: str) -> str:
    """
    Generate a JSON Web Token (JWT) for a given user ID.

    Args:
        uid (str): The unique identifier for the user.

    Returns:
        str: The encoded JWT as a string.
    """
    payload = {
        "sub": uid,
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=2)
    }
    try:
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        logger.info(f"JWT successfully created for UID: {uid}")
        return token
    
    except Exception as e:
        logger.exception("Failed to create JWT.")
        raise JWTCreationError("Failed to create JWT.") from e

def verify_token(token: str) -> Optional[str]:
    """
    Verify the JSON Web Token (JWT) and return the user ID if valid.

    Args:
        token (str): The JWT token to verify.

    Returns:
        Optional[str]: The user ID (`sub`) if the token is valid, otherwise None.

    Raises:
        JWTExpiredError: If the token has expired.
        JWTInvalidError: If the token is invalid or verification fails.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        uid: str = payload.get("sub")
        if not uid:
            logger.warning("JWT does not contain 'sub' claim.")
            raise JWTInvalidError("Token payload missing 'sub'")
        logger.info(f"JWT successfully verified for UID: {uid}")
        return uid
    
    #JWT Token Expired
    except ExpiredSignatureError:
        logger.warning("JWT has expired.")
        raise JWTExpiredError("Token has expired.")
    
    #JWT Verification Failed
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        raise JWTInvalidError("Invalid token.")
    
    #Unknown Exception
    except Exception as e:
        logger.exception("Unexpected error during JWT verification.")
        raise JWTVerificationError(str(e))