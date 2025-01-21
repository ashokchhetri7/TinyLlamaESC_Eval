import requests
from typing import List, Optional, Union
import jsonlines
from tqdm.auto import tqdm
import numpy as np

def get_response(
    prompt: str,
    system: Optional[str] = "",
    temperature: Optional[float] = 1.0,
    greedy: Optional[bool] = False,
):
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
    score = response.split("[RESULT]", 1)[1].strip()
    if "\n" in score:
        score = score.split("\n", 1)[0]

    return output['response'], int(score)

def remove_strategy(content):
    """
    Removes strategy markers from the content.
    """
    if "]" in content:
        return content.split("]", 1)[1].strip()
    return content

def format_prediction(prediction):
    """
    Formats the prediction by removing any leading markers and trailing special tokens.
    """
    prediction = prediction.split("|", 1)[-1] if "|" in prediction else prediction
    prediction = prediction.split("</s>", 1)[0].strip()
    return remove_strategy(prediction)

input_text = """###The instruction to evaluate: 
You are a well-known emotional assistant who has a reputation for being empathetic and understanding. 
Now you are having a conversation with a client. Generate an appropriate answer that will follow user's utterance.

[User]
{dialog}

###Response to evaluate: 
{response}

###Reference Answer (Generate your own) (Score 5): 

 ###Score Rubrics: 
[Fluency: How well does the model's response demonstrate command over language in a way that is coherent and fluid?]
Score 1: The response is incoherent, filled with grammatical errors, and difficult to understand.
Score 2: The response has noticeable grammatical issues and lacks smoothness in delivery, impacting comprehension.
Score 3: The response is mostly coherent with some minor grammatical errors; the flow is acceptable but not seamless.
Score 4: The response is coherent, with minimal grammatical mistakes, and demonstrates a good flow that is easy to follow.
Score 5: The response is perfectly fluent, with no grammatical errors, and demonstrates excellent command of language, making it very engaging and easy to understand."""

results = list(jsonlines.open("./GeneratedDataset/heegyu_esconv-tinyllama_epoch-3.json"))  # Load the specific file
total_scores = []
processed_records = 0  # Counter for the processed records


for r in tqdm(results):
    try:
        dialog = remove_strategy(r["content"])
        prediction = format_prediction(r["prediction"])

        reference = ""  # Placeholder for the reference, adjust as needed based on your evaluation framework

        feedback, score = get_response(input_text.format(dialog=dialog, response=prediction, reference=reference))
        total_scores.append(score)
        processed_records += 1
        if processed_records >= 2295:  # Stop processing after 2295 records
            break
    except Exception as e:
        print(f"An error occurred: {e}")
        break

mean_score = np.mean(total_scores) if total_scores else 0
print(f"Processed {processed_records} records.")
print("Mean score:", mean_score)



