import function_calls
import re

def main():
    filename = function_calls.record_audio()
    if filename:
        engine = function_calls.AudioInferenceEngine()
        result_string = engine.process_file(filename)
        print(f"Model Output: {result_string}")
        degrees = None
        distance = None
        dist_match = re.search(r"Distance: (\d+\.?\d*) ft", result_string)
        if dist_match:
            distance = float(dist_match.group(1))
        deg_match = re.search(r"Source \d+: (\d+\.?\d*)°", result_string)
        if deg_match:
            degrees = float(deg_match.group(1))
        if degrees is not None and distance is not None:
            coordinates = (distance, degrees)
        else:
            print("No active sources detected.")

if __name__ == "__main__":
    main()