import datetime
import subprocess
import hashlib
import time

# Cache dictionary to store recommendations
# Keys: hash of prompt, Value: (timestamp, recommendation)
recommendation_cache = {}
CACHE_EXPIRY_SECONDS = 300  # e.g., cache entries expire after 5 minutes

def generate_prompt(posture_label, detection_count=1, last_alert_time=None):
    """
    Generate a dynamic prompt for generating posture recommendations.
    
    Parameters:
        posture_label (str): The detected posture (e.g., "slouching").
        detection_count (int): Number of times this posture has been detected recently.
        last_alert_time (datetime.datetime, optional): Timestamp of the last alert sent.
    
    Returns:
        str: A dynamically generated prompt.
    """
    
    # Add frequency context if the posture has been detected multiple times
    frequency_info = ""
    if detection_count > 1:
        frequency_info = f"This posture has been detected {detection_count} times recently. "
    
    # Include last alert context if available
    alert_info = ""
    if last_alert_time is not None:
        alert_info = f"The previous alert was sent at {last_alert_time.strftime('%I:%M %p on %B %d, %Y')}. "
    
    # Build the dynamic prompt
    prompt = (
        f"The user is currently {posture_label}. "
        f"Explain briefly how {posture_label} can negatively affect the musculoskeletal system and provide "
        f"personalized, practical tips to correct it. {frequency_info}. Keep your response within 100 words."
    )
    
    return ' '.join(prompt.split())

def get_prompt_hash(prompt):
    """Generate a SHA-256 hash for a given prompt."""
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()

def get_cached_recommendation(prompt):
    """
    Retrieve a recommendation from cache if it exists and is not expired.
    """
    prompt_hash = get_prompt_hash(prompt)
    current_time = time.time()
    if prompt_hash in recommendation_cache:
        timestamp, recommendation = recommendation_cache[prompt_hash]
        if current_time - timestamp < CACHE_EXPIRY_SECONDS:
            return recommendation
        else:
            # Remove expired entry
            del recommendation_cache[prompt_hash]
    return None

def set_cached_recommendation(prompt, recommendation):
    """Store the recommendation in the cache with the current timestamp."""
    prompt_hash = get_prompt_hash(prompt)
    recommendation_cache[prompt_hash] = (time.time(), recommendation)

def call_deepseek(prompt):
    """
    Call the locally running deepseek model using Ollama.
    Adjust the subprocess call if your Ollama setup requires different parameters.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", "deepseek-v2:16b", prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60  # or adjust as needed
        )
        if result.returncode == 0:
            recommendation = result.stdout.strip()
            return recommendation
        else:
            print("Error from deepseek:", result.stderr)
            return None
    except Exception as e:
        print("Exception when calling deepseek:", e)
        return None

def get_recommendation(posture_label, detection_count=1, last_alert_time=None):
    """
    Get a recommendation by first checking the cache; if not present,
    call the deepseek model via Ollama.
    """
    print("Generating DeepSeek recommendation...")
    prompt = generate_prompt(posture_label, detection_count, last_alert_time)
    cached = get_cached_recommendation(prompt)
    if cached:
        return cached
    recommendation = call_deepseek(prompt)
    if recommendation:
        set_cached_recommendation(prompt, recommendation)
        return recommendation
    # Fallback recommendation in case of error
    return "Please maintain proper posture to avoid musculoskeletal issues."

# Example usage:
if __name__ == "__main__":
    # For demonstration, assume "slouching" posture detected 3 times,
    # with no previous alert time.
    rec = get_recommendation("crossed_legs", detection_count=3)
    print("Generated recommendation:")
    print(rec)
