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
    #Server1 http://34.147.57.161:35020/instruct
    #Server1 http://35.204.127.181:35020/instruct
    # output = requests.post("http://35.204.127.181:35020/instruct", json=body).json()
    
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
[Fluency: How well does the model's response demonstrate command over language in a way that is coherent and fluid?]
Score 1: The response is incoherent, filled with grammatical errors, and difficult to understand.
Score 2: The response has noticeable grammatical issues and lacks smoothness in delivery, impacting comprehension.
Score 3: The response is mostly coherent with some minor grammatical errors; the flow is acceptable but not seamless.
Score 4: The response is coherent, with minimal grammatical mistakes, and demonstrates a good flow that is easy to follow.
Score 5: The response is perfectly fluent, with no grammatical errors, and demonstrates excellent command of language, making it very engaging and easy to understand.  """

test_type = "Fluency"

#Before it was suggestion t1, then identification, now it's fluency

def remove_strategy(x):
    return x.split("]", 1)[1].strip() if "]" in x else x

results = list(jsonlines.open("/hdd1/ashok/TINY_EVAL/heegyu_esconv-tinyllama_epoch-3.json"))
total_scores = []

for r in tqdm(results):
    try:
        dialog = ""
        if len(r["context"]) > 1:
            for u in r["context"][-2:]:
                if 'content' in u:
                    dialog += remove_strategy(u['content']) + " "
        else:
            dialog = ""

        prediction = remove_strategy(r["prediction"])

        if 'content' in r["context"][-1]:
            reference = remove_strategy(r["context"][-1]['content'])
        else:
            reference = ""

        feedback, score = get_response(input_text.format(dialog=dialog, response=prediction, reference=reference))
        total_scores.append(score)
    except:
        from traceback import print_exc
        print_exc()
        break

print("Mean score:", np.mean(total_scores))
print("{test_type} Checking: in  Tinyllama")
