"""
Custom Exceptions

This module defines the Exceptions that should be raised for each Exception.
"""

class JWTVerificationError(Exception):
    """Base class for JWT verification errors."""
    pass

class JWTExpiredError(JWTVerificationError):
    """Raised when the JWT has expired."""
    pass

class JWTInvalidError(JWTVerificationError):
    """Raised when the JWT is invalid or malformed."""
    pass

class JWTCreationError(JWTVerificationError):
    """Raised when the JWT is unable to be created."""
    pass