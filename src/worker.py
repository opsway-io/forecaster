import os
import json
import redis
import time
from src.train import train_forecaster_for_monitor

def get_redis_client():
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", "") or None
    sentinel_mode = os.getenv("REDIS_SENTINEL_MODE", "false").lower() in ("true", "1", "yes")
    master_name = os.getenv("REDIS_MASTER_NAME", "mymaster")

    if sentinel_mode:
        from redis.sentinel import Sentinel
        sentinel = Sentinel([(redis_host, redis_port)], socket_timeout=5.0, password=redis_password)
        return sentinel.master_for(master_name, socket_timeout=10.0, password=redis_password, decode_responses=True)

    return redis.Redis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)

def main():
    r = get_redis_client()
    
    stream_key = "TrainForecasterTask"
    group_name = "forecaster_group"
    consumer_name = "forecaster_worker_1"
    
    while True:
        try:
            r.xgroup_create(stream_key, group_name, id='0', mkstream=True)
            break
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                break
            print(f"Redis error creating group: {e}, retrying in 5s...")
            time.sleep(5)
            r = get_redis_client()
        except Exception as e:
            print(f"Failed to connect to Redis to create group: {e}, retrying in 5s...")
            time.sleep(5)
            r = get_redis_client()
            
    print(f"Listening to Redis stream '{stream_key}' as group '{group_name}'...")
    
    while True:
        try:
            # Block for 5 seconds waiting for a new message
            messages = r.xreadgroup(group_name, consumer_name, {stream_key: '>'}, count=1, block=5000)
            
            for stream, msg_list in messages:
                for msg_id, msg_data in msg_list:
                    print(f"Received message: {msg_id} - {msg_data}")
                    
                    try:
                        # Watermill stores the payload in a field, usually 'payload' or 'data'
                        payload_str = msg_data.get('payload') or msg_data.get('data') or list(msg_data.values())[0]
                        payload = json.loads(payload_str)
                        
                        monitor_id = payload.get('monitor_id')
                        if monitor_id is not None:
                            train_forecaster_for_monitor(int(monitor_id))
                        else:
                            print("Missing monitor_id in payload")
                            
                    except Exception as e:
                        print(f"Error processing message {msg_id}: {e}")
                    
                    # Acknowledge the message
                    r.xack(stream_key, group_name, msg_id)
        
        except redis.exceptions.ConnectionError:
            print("Redis connection error, retrying in 5s...")
            time.sleep(5)
            r = get_redis_client()
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
