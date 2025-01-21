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
    output = requests.post("http://34.147.57.161:35020/instruct", json=body).json()
    response = output['response']
    print(output)
    score = response.split("[RESULT]", 1)[1].strip()
    if "\n" in score:
        score = score.split("\n", 1)[0]
    return output['response'], int(score)


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
# [Fluency: How well does the model's response demonstrate command over language in a way that is coherent and fluid?]
# Score 1: The response is incoherent, filled with grammatical errors, and difficult to understand.
# Score 2: The response has noticeable grammatical issues and lacks smoothness in delivery, impacting comprehension.
# Score 3: The response is mostly coherent with some minor grammatical errors; the flow is acceptable but not seamless.
# Score 4: The response is coherent, with minimal grammatical mistakes, and demonstrates a good flow that is easy to follow.
# Score 5: The response is perfectly fluent, with no grammatical errors, and demonstrates excellent command of language, making it very engaging and easy to understand."""

def remove_strategy(x):
    return x.split("]", 1)[1].strip() if "]" in x else x

# Load the new JSON file format
results = list(jsonlines.open("./GeneratedDataset/Ashokajou51_esconv-tinyllama-plm_epoch-3.json"))
total_scores = []

for r in tqdm(results):
    try:
        # Process each record according to the new structure
        content = r.get('content', '')
        prediction = r.get('prediction', '').split("|", 1)[-1].strip()  # Adjusting based on the new prediction format
        
        # Assuming 'content' and 'prediction' are directly available and formatted as needed
        dialog = remove_strategy(content)  # Assume 'content' contains the dialogue context for this turn
        prediction = remove_strategy(prediction)  # Process the prediction as before
        reference = ""  # Adjust based on how you want to handle references in the new format

        feedback, score = get_response(input_text.format(dialog=dialog, response=prediction, reference=reference))
        total_scores.append(score)
    except:
        from traceback import print_exc
        print_exc()
        break

print("Mean score:", np.mean(total_scores))
