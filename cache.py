import redis
import hashlib
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Redis configuration from environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
CACHE_TTL = int(os.getenv("CACHE_TTL", "1200"))  # Default 20 minutes

# Global Redis client
redis_client: Optional[redis.Redis] = None
cache_enabled = False

def init_redis():
    """Initialize Redis connection"""
    global redis_client, cache_enabled

    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2
        )
        # Test connection
        redis_client.ping()
        cache_enabled = True
        logger.info(f"Redis cache initialized successfully at {REDIS_HOST}:{REDIS_PORT} with TTL={CACHE_TTL}s")
    except Exception as e:
        logger.warning(f"Redis connection failed: {str(e)}. Continuing without cache.")
        redis_client = None
        cache_enabled = False

def generate_cache_key(request_data: dict) -> str:
    """
    Generate a unique cache key from request data

    Args:
        request_data: Dictionary containing request fields

    Returns:
        Cache key string in format "prediction:<hash>"
    """
    # Extract relevant fields and normalize
    normalized = {
        "comment": request_data.get("comment", "").strip(),
        "type": request_data.get("type", "{}"),
        "organization": request_data.get("organization", ""),
        "district": request_data.get("district", ""),
        "subdistrict": request_data.get("subdistrict", ""),
        "timestamp": request_data.get("timestamp", "")
    }

    # Create sorted JSON string for consistent hashing
    json_str = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    hash_value = hashlib.md5(json_str.encode('utf-8')).hexdigest()

    return f"prediction:{hash_value}"

def get_cached_prediction(cache_key: str) -> Optional[dict]:
    """
    Retrieve cached prediction result

    Args:
        cache_key: Cache key to lookup

    Returns:
        Cached prediction dict or None if not found/error
    """
    if not cache_enabled or redis_client is None:
        return None

    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            logger.info(f"Cache HIT: {cache_key}")
            return json.loads(cached_data)
        else:
            logger.info(f"Cache MISS: {cache_key}")
            return None
    except Exception as e:
        logger.warning(f"Cache read error: {str(e)}")
        return None

def set_cached_prediction(cache_key: str, prediction_result: dict, ttl: int = CACHE_TTL) -> bool:
    """
    Store prediction result in cache

    Args:
        cache_key: Cache key
        prediction_result: Prediction result to cache
        ttl: Time to live in seconds

    Returns:
        True if successful, False otherwise
    """
    if not cache_enabled or redis_client is None:
        return False

    try:
        redis_client.setex(
            cache_key,
            ttl,
            json.dumps(prediction_result, ensure_ascii=False)
        )
        logger.info(f"Cache SET: {cache_key} (TTL={ttl}s)")
        return True
    except Exception as e:
        logger.warning(f"Cache write error: {str(e)}")
        return False

def get_cache_stats() -> dict:
    """
    Get cache statistics

    Returns:
        Dictionary with cache stats
    """
    if not cache_enabled or redis_client is None:
        return {
            "enabled": False,
            "connected": False
        }

    try:
        info = redis_client.info()
        return {
            "enabled": True,
            "connected": True,
            "host": REDIS_HOST,
            "port": REDIS_PORT,
            "db": REDIS_DB,
            "ttl_seconds": CACHE_TTL,
            "total_keys": info.get("db0", {}).get("keys", 0) if isinstance(info.get("db0"), dict) else 0,
            "used_memory_human": info.get("used_memory_human", "N/A")
        }
    except Exception as e:
        logger.warning(f"Failed to get cache stats: {str(e)}")
        return {
            "enabled": True,
            "connected": False,
            "error": str(e)
        }

def clear_cache():
    """Clear all prediction cache entries"""
    if not cache_enabled or redis_client is None:
        return False

    try:
        # Delete only prediction keys
        for key in redis_client.scan_iter("prediction:*"):
            redis_client.delete(key)
        logger.info("Cache cleared successfully")
        return True
    except Exception as e:
        logger.warning(f"Cache clear error: {str(e)}")
        return False
