def extract_clip(start_time, end_time):
    # Add validation for clip duration
    if start_time >= end_time:
        raise ValueError(f"Invalid clip duration: start time ({start_time}s) must be less than end time ({end_time}s)")
    
    # ... rest of extraction code ...

def process_clip(feedback_id, start_time, end_time):
    try:
        print(f"Processing clip for feedback_id: {feedback_id} from {start_time}s to {end_time}s")
        
        # Validate clip duration before attempting extraction
        if start_time >= end_time:
            print(f"Skipping clip {feedback_id}: Invalid duration (start: {start_time}s, end: {end_time}s)")
            return False
            
        print(f"Extracting clip from {start_time}s to {end_time}s")
        # ... rest of processing code ...
        
    except Exception as e:
        print(f"Error processing clip: {str(e)}")
        return False 