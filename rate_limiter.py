from datetime import datetime, timedelta
from typing import Dict, Tuple
import logging
from fastapi import Request, HTTPException
from collections import defaultdict

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, requests_per_minute: int = 30, requests_per_hour: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_requests: Dict[str, list] = defaultdict(list)
        self.hour_requests: Dict[str, list] = defaultdict(list)
        
    def _cleanup_old_requests(self, user_id: str):
        """Remove old requests from tracking."""
        now = datetime.now()
        
        # Clean up minute requests
        self.minute_requests[user_id] = [
            req_time for req_time in self.minute_requests[user_id]
            if now - req_time < timedelta(minutes=1)
        ]
        
        # Clean up hour requests
        self.hour_requests[user_id] = [
            req_time for req_time in self.hour_requests[user_id]
            if now - req_time < timedelta(hours=1)
        ]
    
    def check_rate_limit(self, request: Request) -> bool:
        """
        Check if the request should be rate limited.
        
        Returns:
            bool: True if request should be allowed, False if rate limited
        """
        try:
            # Get user ID from session
            user_id = request.session.get("user_id")
            if not user_id:
                return True  # Allow requests without session
                
            now = datetime.now()
            
            # Clean up old requests
            self._cleanup_old_requests(user_id)
            
            # Check minute rate limit
            if len(self.minute_requests[user_id]) >= self.requests_per_minute:
                logger.warning(f"Rate limit exceeded for user {user_id}: too many requests per minute")
                return False
                
            # Check hour rate limit
            if len(self.hour_requests[user_id]) >= self.requests_per_hour:
                logger.warning(f"Rate limit exceeded for user {user_id}: too many requests per hour")
                return False
            
            # Record the request
            self.minute_requests[user_id].append(now)
            self.hour_requests[user_id].append(now)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in rate limiting: {str(e)}")
            return True  # Allow request on error 