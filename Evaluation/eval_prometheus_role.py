import requests
from typing import List, Optional, Union
import jsonlines
from tqdm.auto import tqdm
import numpy as np
import pdb
import json
import sys



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
    output = requests.post("http://34.147.57.161:35020/instruct", json=body).json()
    response = output['response']
    print(output)
    score = response.split("[RESULT]", 1)[1].strip()
    if "\n" in score:
        score = score.split("\n", 1)[0]

    return output['response'], int(score)

input_text = """###The instruction to evaluate: 
You are a well-known emotional assistant who has a reputation for being empathetic and understanding. 
Now you are reviewing a conversation with a client. Your job is to generate an appropriate score given the user's utterance, response and the reference on the rubrics suggested below;

[User]
{dialog}

###Response to evaluate: 
{response}

###Reference Answer (Score 5): 
{reference}

# ###Score Rubrics: 
[Suggestion: How well does the model offer constructive suggestions or advice relevant to the user's expressed concerns?]
Score 1: The model provides irrelevant or unhelpful suggestions that do not address the user's concerns.
Score 2: The model offers suggestions that may have a vague relevance but largely miss the mark in terms of usefulness or applicability.
Score 3: The model gives generally relevant suggestions, but they may lack specificity or full applicability to the user's situation.
Score 4: The model provides relevant and practical suggestions that are mostly applicable to the user's expressed concerns.
Score 5: The model offers insightful, specific, and highly relevant suggestions that directly address and provide actionable advice for the user's concerns."""


def remove_strategy(x):
    x = x.split("]", 1)[1].strip() if "]" in x else x
    x = x.replace("</s>", "")
    return x

def remove_knowledge(x):
    x = x.split("[", 1)[0].strip() if "[" in x else x
    x = x.replace("User:", "").strip()
    return x


results = list(jsonlines.open("./GeneratedDataset/Ashokajou51_esconv-sorted-role-tinyllama-plm_epoch-3.json"))
total_scores = []

counter = 0
for r in tqdm(results):

    r["context"].pop(0)

    if len(r["context"]) < 2:
        continue

    try:
        dialog = ""
        for turn in r["context"]:
            if turn["role"] == "user":
                dialog = remove_knowledge(turn["content"])
                break

        prediction = remove_strategy(r["prediction"])
        reference = remove_strategy(r["context"][-1]["content"])

        for _ in range(5):
            print(input_text)
        
        sys.exit()



    

        feedback, score = get_response(input_text.format(dialog=dialog, response=prediction, reference=reference))
        total_scores.append(score)
    except Exception as e:
        print(f"Error processing data: {e}")
        continue

print("Mean score:", np.mean(total_scores))








        # print(input_text.format(dialog=dialog, response=prediction, reference=reference))
        
        # if counter > 5:
        #     sys.out()

        # counter = counter+1
        # continue

