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

# ###Score Rubrics: 
#[Fluency: How well does the model's response demonstrate command over language in a way that is coherent and fluent?] 
# Score 1: The response is incoherent, filled with grammatical errors, and difficult to understand.
# Score 2: The response has noticeable grammatical issues and lacks smoothness in delivery, impacting comprehension.
# Score 3: The response is mostly coherent with some minor grammatical errors; the flow is acceptable but not seamless.
# Score 4: The response is coherent, with minimal grammatical mistakes, and demonstrates a good flow that is easy to follow.
# Score 5: The response is perfectly fluent, with no grammatical errors, and demonstrates excellent command of language, making it very engaging and easy to understand."""

def remove_strategy(x):
    return x.split("]", 1)[1].strip() if "]" in x else x

results = list(jsonlines.open("./GeneratedDataset/Ashokajou51_esconv-sorted-role-tinyllama-plm_epoch-3.json"))
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
        
        # print(f"{input_text.format(dialog=dialog, response=prediction, reference=reference)}")


        feedback, score = get_response(input_text.format(dialog=dialog, response=prediction, reference=reference))
        total_scores.append(score)
    except:
        from traceback import print_exc
        print_exc()
        break

print("Mean score:", np.mean(total_scores))




# INPUT
"""
'###Task Description:\n
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.\n1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.\n2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.\n3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"\n4. Please do not generate any other opening, closing, and explanations.\n

###The instruction to evaluate: \n
You are a well-known emotional assistant who has a reputation for being empathetic and understanding. \n
Now you are having a conversation with a client. Generate an appropriate answer that will follow user\'s utterance.\n\n

[User]\nnervous [xIntent] none [xWant] to relax [xNeed] none [xEffect] none\\n [Heal]: [resp] Depressed [str] sadness, fine, sad, normal, angry [aff] Sad Well with the holidays coming up i have been very stressed and nervous about what i am going to do I really feel you there. Holidays are so hard.. especially with the way this year has been. Anything specific? \n\n

###Response to evaluate: \n
That\'s understandable. Are you having trouble making decisions?</s>\n\n

###Reference Answer (Score 5): \n
I really feel you there. Holidays are so hard.. especially with the way this year has been. Anything specific?\n\n

####Score Rubrics: \n
#[Fluency: How well does the model\'s response demonstrate command over language in a way that is coherent and fluent?] \n
# Score 1: The response is incoherent, filled with grammatical errors, and difficult to understand.\n
# Score 2: The response has noticeable grammatical issues and lacks smoothness in delivery, impacting comprehension.\n
# Score 3: The response is mostly coherent with some minor grammatical errors; the flow is acceptable but not seamless.\n
# Score 4: The response is coherent, with minimal grammatical mistakes, and demonstrates a good flow that is easy to follow.\n
# Score 5: The response is perfectly fluent, with no grammatical errors, and demonstrates excellent command of language, making it very engaging and easy to understand.

###Feedback: ', 'response': "\n
The response is perfectly fluent, with no grammatical errors, and demonstrates excellent command of language, making it very engaging and easy to understand. The assistant's empathetic tone is evident, and the response is coherent, addressing the user's feelings of stress and nervousness. The assistant also asks a relevant question to help the user relax. 
So the overall score is 5. [RESULT] 5", 'tokens_used': 93}
"""