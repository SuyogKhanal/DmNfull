import json

input_file = "meeting.json"
output_file = "output.json"

# Correct mappings for YOUR JSON
speaker_replacements = {
    "speaker0": "Santu",
    "speaker1": "Suyog",
    "speaker2": "Arun",
    "unknown": "Santu/Arun"
}

def replace_speakers(obj):

    if isinstance(obj, dict):

        for key, value in obj.items():

            # Replace speaker values
            if key == "speaker":
                if isinstance(value, str):
                    obj[key] = speaker_replacements.get(value, value)

            # Continue recursion
            replace_speakers(value)

    elif isinstance(obj, list):

        for item in obj:
            replace_speakers(item)

# Load JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Replace all speaker values recursively
replace_speakers(data)

# Save updated JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("Done.")