import requests
from typing import List, Optional, Union
import requests
from typing import List, Optional, Union
import jsonlines
from tqdm.auto import tqdm
import numpy as np

def get_response(prompt: str, system: Optional[str] = "", temperature: Optional[float] = 1.0, greedy: Optional[bool] = False):
    """
    Get a response from the model.
    """
    body = {
        "instruction": prompt,
        "system": system,
        "temperature": temperature,
        "greedy": greedy,
    }
    output = requests.post("http://35.204.127.181:35020/instruct", json=body).json()
    response = output['response']
    print(output)
    try:
        score = response.split("[RESULT]", 1)[1].strip()
        if "\n" in score:
            score = score.split("\n", 1)[0]
        score = int(score)  # Ensure score is an integer
    except IndexError:
        print("Error parsing score from response:", response)
        score = 0  # Default score or handle appropriately
    return output['response'], score


input_text = """###The instruction to evaluate: 
You are a well-known emotional assistant who has a reputation for being empathetic and understanding. 
Now you are having a conversation with a client. Generate an appropriate answer that will follow user's utterance.

[User]
{dialog}

###Response to evaluate: 
{response}

###Reference Answer (Score 5): 
{reference}

###Score Rubrics: 
[Comforting: How effective is the model at providing comfort and emotional support in its response?]
Score 1: The model's response fails to offer comfort or support, possibly exacerbating the user's distress.
Score 2: The model's attempt to provide comfort is minimal and may not resonate with or adequately support the user's emotional needs.
Score 3: The model offers a moderate level of comfort and support, somewhat addressing the user's emotional needs but lacking depth.
Score 4: The model effectively provides comfort and emotional support, closely addressing the user's needs with thoughtful responses.
Score 5: The model excels at offering profound comfort and emotional support, deeply resonating with and fully addressing the user's emotional state. """


# Function to remove strategy text, if present
def remove_strategy(x):
    return x.split("]", 1)[1].strip() if "]" in x else x

# Load the JSON file format
results = list(jsonlines.open("gen2.txt"))
total_scores = []

for r in tqdm(results):
    try:
        # Map the fields from the JSON entry to input_text variables
        dialog = r.get('post', '')
        prediction = r.get('generation', '')
        reference = r.get('response', '')

        # Process the dialogue, prediction, and reference through remove_strategy if needed
        dialog = remove_strategy(dialog)
        prediction = remove_strategy(prediction)
        reference = remove_strategy(reference)

        feedback, score = get_response(input_text.format(dialog=dialog, response=prediction, reference=reference))
        total_scores.append(score)
    except Exception as e:
        print("An error occurred:", e)
        break

if total_scores:
    print("Mean score:", np.mean(total_scores))
else:
    print("No scores to calculate mean.")
