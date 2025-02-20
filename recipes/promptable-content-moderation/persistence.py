import json
import os

def save_detection_data(data, output_file):
    """
    Saves the detection data to a JSON file.

    Args:
        data (dict): The complete detection data structure.
        output_file (str): Path to the output JSON file.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Detection data saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return False

def load_detection_data(input_file):
    """
    Loads the detection data from a JSON file.

    Args:
        input_file (str): Path to the JSON file.
        
    Returns:
        dict: The loaded detection data, or None if there was an error.
    """
    try:
        with open(input_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None 