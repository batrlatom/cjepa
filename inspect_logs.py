
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys

def main():
    log_dir = "runs/nlp_demo_v2"
    if not os.path.exists(log_dir):
        print(f"Log dir {log_dir} not found.")
        return

    # Find the event file
    event_files = [f for f in os.listdir(log_dir) if "tfevents" in f]
    if not event_files:
        print("No event files found.")
        return
        
    path = os.path.join(log_dir, event_files[0])
    print(f"Reading {path}...")
    
    ea = EventAccumulator(path)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    print(f"Found tags: {tags}")
    
    for tag in ['Train/Loss_Total', 'Train/Loss_History', 'Train/Loss_Future']:
        if tag in tags:
            events = ea.Scalars(tag)
            if events:
                first = events[0].value
                last = events[-1].value
                print(f"{tag}: Start={first:.4f} -> End={last:.4f} (Steps: {len(events)})")

if __name__ == "__main__":
    main()
