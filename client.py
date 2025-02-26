import json
import time
import sys
import os
from gradio_client import Client, handle_file

# Replace with the actual server URL if different
ip = "127.0.0.1"
port = "8080"

# Define the user prompt (caption)
user_prompt = "Thoroughly and carefully describe this image."

files = []
output_file = "output.json"

# Hyperparameters
temperature = 0.6
top_k = 50
top_p = 0.9
max_tokens = 100

startAt = 0

argumentStart = 1
if len(sys.argv) > 1:
    for i in range(0, len(sys.argv)):
        if sys.argv[i] == "--ip":
            ip = sys.argv[i + 1]
            argumentStart += 2
        if sys.argv[i] == "--directory":
            directory = sys.argv[i + 1]
            argumentStart += 2
            # Populate files with image (.jpg, .png) contents of directory
            if os.path.isdir(directory):
                directoryList = os.listdir(directory)
                directoryList.sort()
                for file in directoryList:
                    if file.lower().endswith(('.jpg', '.png', '.jpeg', '.txt')):
                        files.append(os.path.join(directory, file))
            else:
                print(f"Error: Directory '{directory}' does not exist.")
                sys.exit(1)
        elif sys.argv[i] == "--start":
            startAt = int(sys.argv[i + 1])
            argumentStart += 2
        elif sys.argv[i] == "--port":
            port = sys.argv[i + 1]
            argumentStart += 2
        elif sys.argv[i] == "--prompt":
            user_prompt = sys.argv[i + 1]
            argumentStart += 2
        elif sys.argv[i] == "--temperature":
            temperature = float(sys.argv[i + 1])
            argumentStart += 2
        elif sys.argv[i] == "--top_k":
            top_k = int(sys.argv[i + 1])
            argumentStart += 2
        elif sys.argv[i] == "--top_p":
            top_p = float(sys.argv[i + 1])
            argumentStart += 2
        elif sys.argv[i] == "--max_tokens":
            max_tokens = int(sys.argv[i + 1])
            argumentStart += 2
        elif sys.argv[i] in ("--output", "-o"):
            output_file = sys.argv[i + 1]
            argumentStart += 2

# Initialize the Gradio client with the server URL
client = Client(f"http://{ip}:{port}")

results = {"prompt": user_prompt}

for i in range(argumentStart, len(sys.argv)):
    files.append(sys.argv[i])

# Make sure the list is sorted
files.sort()

# Possibly start at a specific index
for i in range(startAt, len(files)):
    # Grab the next image path
    image_path = files[i]

    # Count start time
    start = time.time()

    # Make query to VLLM
    try:
        imageFile = None
        this_user_prompt = user_prompt
        if image_path.endswith('.txt'):
            with open(image_path, 'r') as txt_file:
                this_user_prompt = txt_file.read().strip()
        else:
            imageFile = handle_file(image_path)

        # Reset state 
        result = client.predict(api_name="/reset_state" )

        # Send the image file path and the prompt to the Gradio app for processing
        result = client.predict(
            input_images=[imageFile],           # Provide the file path directly
            input_text=this_user_prompt,     # Adapted prompt parameter
            api_name="/transfer_input"
        )

        result = client.predict(
		    chatbot=[],
            temperature=temperature,
            #top_k=top_k,
            top_p=top_p,
            max_length_tokens=max_tokens, # Adapted max_tokens parameter
		    repetition_penalty=1.1,
		    max_context_length_tokens=4096,
		    #model_select_dropdown="deepseek-ai/deepseek-vl2-tiny",
            api_name="/predict"
        )



    except Exception as e:
        print(f"Failed to complete job at index {i}: {e}")
        output_file = f"partial_until_{i}_{output_file}"
        break

    # Calculate elapsed time
    seconds = time.time() - start
    remaining = (len(files) - i) * seconds
    hz = 1 / (seconds + 0.0001)

    # Output the result
    #print("result[0][0][1] ",result[0][0][1])
    question = this_user_prompt #Don't try to recover it from the list..
    response = result[0][0][1]

    # Print on screen
    print(f"Processing {1 + i}/{len(files)} | {hz:.2f} Hz / remaining {remaining / 60:.2f} minutes")
    print(f"Image: {image_path}\nResponse: {response}")

    # Store each path as the key pointing to each description
    results[image_path] = response

# Save results to JSON
print(f"\n\n\nStoring results in JSON file {output_file}")
with open(output_file, "w") as outfile:
    json.dump(results, outfile, indent=4)

