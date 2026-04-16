import os
import json
import time
import argparse
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# 1. Setup the Client
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Make sure it's set in your .env file.")

client = genai.Client(api_key=api_key)

# 2. Define the System Instruction (The "Sensei" Logic)
SYSTEM_INSTRUCTION = """
You are an Expert Trajectory Optimization Engine for a 3.5-inch FPV drone.
Drone Profile: Volador VX3.5, 320g AUW, 12Hz Control Frequency.

Observation Space Scaling:
- obs[0] & obs[5]: 1.0 = 1.0m (Altitude).
- obs[1-2]: 1.0 = 1.0m (Drift).
- obs[3-4]: 1.0 = 5.0m/s (Velocity).

Action Space & Logic:
- action[0]: Normalized Altitude Setpoint. Formula: desired_alt_norm = (action[0] + 1) / 2.
- action[1-2]: Pitch and Roll corrections to counter drift in obs[1-4].

Rules:
1. DO NOT modify observations; keep the original noisy data.
2. Output exactly 24 steps (pad with stable hover if needed).
3. Ensure the final action[0] maps perfectly to the target_alt.
"""

def refine_episode(raw_data):
    """Sends one episode to Gemini for expert relabeling with retry logic for quota limits."""
    while True:
        try:
            response = client.models.generate_content(
                model='gemini-robotics-er-1.5-preview',
                contents=f"Refine this log for Behavior Cloning: {json.dumps(raw_data)}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json"
                    # thinking=True is not supported in this SDK/Model combination
                )
            )
            return json.loads(response.text)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                print(f"Quota exceeded (429 RESOURCE_EXHAUSTED). Waiting 5 minutes before retrying...")
                time.sleep(300)  # Wait for 5 minutes
                continue
            else:
                print(f"Error during API call: {e}")
                return None

def get_episode_number(filename):
    """Extracts episode number from filename (e.g., 'episode_005.json' -> 5)."""
    match = re.search(r'episode_(\d+)\.json', filename)
    if match:
        return int(match.group(1))
    return None

# 3. The Batch Processing Loop
def process_data_factory(start_episode=None, end_episode=None):
    # Resolve paths relative to the script's location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    input_dir = os.path.join(project_root, 'input', 'raw_logs')
    output_dir = os.path.join(project_root, 'input', 'expert_dataset')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Gather and sort files to ensure consistent processing
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".json")])
    
    for filename in files:
        ep_num = get_episode_number(filename)
        
        # Skip files that don't match the episode pattern if a range is specified
        if ep_num is None and (start_episode is not None or end_episode is not None):
            continue
            
        # Filter by episode range if provided
        if start_episode is not None and ep_num < start_episode:
            continue
        if end_episode is not None and ep_num > end_episode:
            continue

        output_filename = f"expert_{filename}"
        output_path = os.path.join(output_dir, output_filename)

        # Check if file already exists to skip re-processing (simple resume logic)
        if os.path.exists(output_path):
            print(f"Skipping {filename} (already exists)...")
            continue

        print(f"Refining {filename} (Episode {ep_num})...")
        
        with open(os.path.join(input_dir, filename), 'r') as f:
            raw_log = json.load(f)
        
        expert_log = refine_episode(raw_log)
        
        if expert_log:
            with open(output_path, 'w') as f:
                json.dump(expert_log, f, indent=2)
            print(f"Successfully saved {output_filename}")
        
        # 4. Respect Rate Limits (10 seconds for Free Tier)
        time.sleep(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate expert data using Gemini API.")
    parser.add_argument("--start", type=int, help="Starting episode number (inclusive).")
    parser.add_argument("--end", type=int, help="Ending episode number (inclusive).")
    
    args = parser.parse_args()
    
    process_data_factory(start_episode=args.start, end_episode=args.end)