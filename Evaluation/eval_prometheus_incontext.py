# Evaluation\eval_prometheus_incontext.py

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




input_text = """###The instruction to evaluate: 
You are a well-known emotional assistant who has a reputation for being empathetic and understanding. 
Now you are having a conversation with a client. Generate an appropriate answer that will follow user's utterance.

[User]
{dialog}

###Response to evaluate: 
{response}

###Reference Answer (Score 5): 
{reference}

# [Suggestion: How well does the model offer constructive suggestions or advice relevant to the user's expressed concerns?]
# Score 1: The model provides irrelevant or unhelpful suggestions that do not address the user's concerns.
# Score 2: The model offers suggestions that may have a vague relevance but largely miss the mark in terms of usefulness or applicability.
# Score 3: The model gives generally relevant suggestions, but they may lack specificity or full applicability to the user's situation.
# Score 4: The model provides relevant and practical suggestions that are mostly applicable to the user's expressed concerns.
# Score 5: The model offers insightful, specific, and highly relevant suggestions that directly address and provide actionable advice for the user's concerns."""

def remove_strategy(x):
    x = x.split("]", 1)[1].strip() if "]" in x else x
    x = x.replace("</s>", "")
    return x

def remove_knowledge(x):
    x = x.split("[", 1)[0].strip() if "[" in x else x
    x = x.replace("User:", "").strip()
    return x

results = list(jsonlines.open("./GeneratedDataset/Ashokajou51_esconv-sorted-incontext-tinyllama-plm_epoch-2.json"))
total_scores = []

for r in tqdm(results):
    # remove system prompt
    r["context"].pop(0)

    if len(r["context"]) < 2:
        continue

    try:
        dialog = remove_knowledge(r["context"][-2]['content'])
        prediction = remove_strategy(r["prediction"])
        reference = remove_strategy(r["context"][-1]['content'])

        feedback, score = get_response(input_text.format(dialog=dialog, response=prediction, reference=reference))
        total_scores.append(score)
    except:
        from traceback import print_exc
        print_exc()
        break

print("Mean score:", np.mean(total_scores))

