import os
import json
import redis
import time
from src.train import train_forecaster_for_monitor

def main():
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    
    stream_key = "TrainForecasterTask"
    group_name = "forecaster_group"
    consumer_name = "forecaster_worker_1"
    
    try:
        r.xgroup_create(stream_key, group_name, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise e
            
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
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
