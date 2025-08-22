"""
Rate limiting utilities to prevent API abuse
Implements token bucket algorithm for rate limiting
"""

import time
import asyncio
from typing import Dict, Optional
from collections import defaultdict
from fastapi import HTTPException

class RateLimiter:
    """
    Token bucket rate limiter
    
    Implements rate limiting using the token bucket algorithm.
    Each user gets a bucket with a certain capacity and refill rate.
    """
    
    def __init__(self, 
                 requests_per_minute: int = 60,
                 burst_size: int = 10):
        """
        Initialize rate limiter
        
        Args:
            requests_per_minute: Number of requests allowed per minute
            burst_size: Maximum burst size (bucket capacity)
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.refill_rate = requests_per_minute / 60.0  # tokens per second
        
        # Store bucket state for each user/token
        self.buckets: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "tokens": float(burst_size),
                "last_refill": time.time()
            }
        )
    
    async def check_rate_limit(self, identifier: str) -> bool:
        """
        Check if request is allowed under rate limit
        
        Args:
            identifier: User/token identifier
            
        Returns:
            True if request is allowed
            
        Raises:
            HTTPException: If rate limit exceeded
        """
        current_time = time.time()
        bucket = self.buckets[identifier]
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = current_time - bucket["last_refill"]
        tokens_to_add = time_elapsed * self.refill_rate
        
        # Update bucket
        bucket["tokens"] = min(
            self.burst_size,
            bucket["tokens"] + tokens_to_add
        )
        bucket["last_refill"] = current_time
        
        # Check if request can be fulfilled
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        else:
            # Calculate wait time
            wait_time = (1.0 - bucket["tokens"]) / self.refill_rate
            
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {wait_time:.1f} seconds",
                headers={"Retry-After": str(int(wait_time) + 1)}
            )
    
    def get_bucket_status(self, identifier: str) -> Dict[str, float]:
        """Get current bucket status for an identifier"""
        current_time = time.time()
        bucket = self.buckets[identifier]
        
        # Calculate current tokens
        time_elapsed = current_time - bucket["last_refill"]
        tokens_to_add = time_elapsed * self.refill_rate
        current_tokens = min(
            self.burst_size,
            bucket["tokens"] + tokens_to_add
        )
        
        return {
            "current_tokens": current_tokens,
            "max_tokens": self.burst_size,
            "refill_rate": self.refill_rate,
            "time_to_full": max(0, (self.burst_size - current_tokens) / self.refill_rate)
        }
    
    def reset_bucket(self, identifier: str):
        """Reset bucket for an identifier (admin function)"""
        if identifier in self.buckets:
            self.buckets[identifier] = {
                "tokens": float(self.burst_size),
                "last_refill": time.time()
            }

class AdvancedRateLimiter:
    """
    Advanced rate limiter with multiple tiers and adaptive limits
    """
    
    def __init__(self):
        self.user_tiers = {
            "admin": {"requests_per_minute": 1000, "burst_size": 50},
            "user": {"requests_per_minute": 60, "burst_size": 10},
            "anonymous": {"requests_per_minute": 20, "burst_size": 5}
        }
        
        self.limiters = {}
        for tier, config in self.user_tiers.items():
            self.limiters[tier] = RateLimiter(
                requests_per_minute=config["requests_per_minute"],
                burst_size=config["burst_size"]
            )
    
    async def check_rate_limit(self, identifier: str, user_role: str = "user") -> bool:
        """
        Check rate limit based on user role
        
        Args:
            identifier: User identifier
            user_role: User role (admin, user, anonymous)
            
        Returns:
            True if request is allowed
        """
        limiter = self.limiters.get(user_role, self.limiters["user"])
        return await limiter.check_rate_limit(identifier)
    
    def get_limits_for_role(self, role: str) -> Dict[str, int]:
        """Get rate limits for a specific role"""
        return self.user_tiers.get(role, self.user_tiers["user"])

# Global rate limiter instance
default_rate_limiter = RateLimiter()

# Advanced rate limiter for different user tiers
advanced_rate_limiter = AdvancedRateLimiter()

# For testing
if __name__ == "__main__":
    import asyncio
    
    async def test_rate_limiter():
        limiter = RateLimiter(requests_per_minute=10, burst_size=3)
        
        # Test normal usage
        for i in range(5):
            try:
                await limiter.check_rate_limit("test_user")
                print(f"Request {i+1}: Allowed")
            except HTTPException as e:
                print(f"Request {i+1}: Denied - {e.detail}")
            
            await asyncio.sleep(0.1)
    
    asyncio.run(test_rate_limiter())
