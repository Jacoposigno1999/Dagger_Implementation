import os
import re

def find_last_episode(directory, pattern=r'model_(\d+)\.pth'):
    """
    Find the highest episode number from saved model files in the directory.
    
    Args:
        directory (str): Path to the directory containing model files.
        pattern (str): Regex pattern to match model file names.

    Returns:
        int: The highest episode number found, or -1 if no files match.
    """
    last_episode = 0
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            match = re.match(pattern, filename)
            if match:
                episode = int(match.group(1))  # Extract the episode number
                last_episode = max(last_episode, episode)
    return last_episode

