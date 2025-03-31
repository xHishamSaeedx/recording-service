from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.oauth2 import id_token
from google.auth.transport import requests
from config import (
    SERVICE_ACCOUNT_EMAIL
)
security = HTTPBearer()


async def validate_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Validates the OIDC token using Google's authentication with cloud run service account
    
    Args:
        credentials: The HTTP Authorization credentials containing the bearer token
        
    Returns:
        dict: Token information including user details if validation is successful
        
    Raises:
        HTTPException: If token is invalid, expired, or verification fails
    """
    try:
        # Verify token with Google using cloud run service account email as audience
        token_info = id_token.verify_oauth2_token(
            credentials.credentials,
            requests.Request(),
            audience=SERVICE_ACCOUNT_EMAIL
        )

        # Verify token expiration
        if not token_info.get('exp'):
            raise HTTPException(
                status_code=401,
                detail="Token has no expiration claim"
            )

        # Return relevant token information
        return {
            "email": token_info.get('email'),
            "user_id": token_info.get('sub'),
            "name": token_info.get('name'),
            "expires_at": token_info.get('exp')
        }

    except Exception as e:
        raise HTTPException(
            status_code=403,
            detail="Invalid or expired token"
        )
