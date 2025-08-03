"""
JWT authentication dependency for FastAPI routes.

This module defines a custom FastAPI security dependency that verifies
a JWT token from the Authorization header using the Bearer scheme.
"""

from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.status import HTTP_403_FORBIDDEN

from .auth_service import verify_token
from utils.exceptions import JWTExpiredError, JWTInvalidError

class JWTBearer(HTTPBearer):
    """
    Custom HTTPBearer authentication class for FastAPI that validates JWT tokens.
    """

    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        token = credentials.credentials

        try:
            uid = verify_token(token)
            request.state.user = uid  # Store user ID in request for downstream use
            return uid
        
        #JWT Expired Error
        except JWTExpiredError:
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Token has expired.")
        
        #Invalid or malformed JWT Token
        except JWTInvalidError:
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid token.")
        
        #Unknown Error
        except Exception:
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Token verification failed.")
