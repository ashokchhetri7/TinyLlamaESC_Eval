#Testing with giving knowledge and Heal in the input:

import fire
import torch
import json, jsonlines
from tqdm.auto import tqdm
from copy import deepcopy

torch.set_grad_enabled(False)

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from typing import List, Dict, Any, Optional


def main(
    model_name: str = "Ashokajou51/esconv-tinyllama-plm",  # Update your model name
    revision: Optional[str] = "epoch-1",  # Update your model name

):
    dataset_name = "heegyu/esconv-KEMI"
    dataset = load_dataset(dataset_name, split="test")

    def map_conversations(item):
        dialog = item["dialog"]
        convs = []
        speaker2role = {
            "sys": "assistant",
            "usr": "user"
        }
        prev_heal = None

        for i in range(len(dialog)):
            uttr = dialog[i]
            speaker = uttr["speaker"]
            text = uttr["text"]

            if speaker == "sys":
                text = f"[{uttr['strategy']}] {text}"
                prev_heal = uttr.get("heal", None)

            content = text
            if speaker == "usr":
                if "knowledge" in uttr:
                    content += f"\n{uttr['knowledge']}"
                if prev_heal:
                    content += f"\n{prev_heal}"

            convs.append({"role": speaker2role[speaker],
                          "content": content})

        return {"conversations": convs,
                "dialog": dialog}

    dataset = dataset.map(map_conversations)

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    tokenizer.truncation_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision).eval().half().to(device)

    gen_kwargs = dict(
        do_sample=True,
        max_new_tokens=128,
    )

    if revision:
        model_name = model_name.replace("/", "_")
        save_file = f"./GeneratedDataset{model_name}_{revision}.json"
    else:
        save_file = f"./GeneratedDataset{model_name}.json"

    fout = jsonlines.open(save_file, "w")

    for item in tqdm(dataset):
        dialog = item["dialog"]
        convs = item["conversations"]
        for i in range(1, len(convs)):
            context = convs[:i]

            if convs[i]['role'] == 'user':
                continue

            # Pass the user utterance along with knowledge and heal values as the prompt
            prompt = context[-1]['content'] if context else ""
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            gens = model.generate(**inputs, **gen_kwargs)

            response = gens[:, inputs["input_ids"].shape[1]:]

            response = tokenizer.decode(response[0])
            # print(prompt, "-->")
            # print("Prediction", response)
            # print("GT", convs[i]['content'])

            fout.write({
                "turn_index": i,
                **convs[i],
                "prediction": response
            })

    fout.close()

if __name__ == "__main__":
    fire.Fire(main)

#Testing with the knowledge and Heal is given:
# hi i am okay, a little bit sad though
# [xReact] sad [xIntent] none [xWant] to cry [xNeed] none [xEffect] none -->
# Well with the holidays coming up i have been very stressed and nervous about what i am going to do
# [xReact] nervous [xIntent] none [xWant] to relax [xNeed] none [xEffect] none
# [resp] Depressed [str] sadness, fine, sad, normal, angry [aff] Sad -->
# I have no choice but to go live at home over the break but I am very scared about it
# [xReact] worried [xIntent] none [xWant] to stay home [xNeed] none [xEffect] none
# [resp] Relax and enjoy. [str] calm, correct, enery, longtime, feelin [aff] Anxious -->
# [Question] Do you feel scared for your own self? Or of the situation in general? I'm sorry to hear that. Relationships with parents can be very difficult sometimes. -->
# and i have no car to escape it
# [xReact] sad [xIntent] none [xWant] to get a new car [xNeed] none [xEffect] gets arrested
# [resp] I want you to survive. [str] commit, killing, death, painless, option [aff] Afraid -->
# [Self-disclosure] It's like we live the same life. I also have no car to escape! It seems so small, but it's such a huge stressor when you feel trapped in an environment you're not positive in. I feel your pain and I empathize with you completely. -->
# Not many to be honest. I have a hamster but he is at school with me so nothing at home to go back to
# [xReact] bored [xIntent] none [xWant] to do something else [xNeed] none [xEffect] none
# [resp] And it never gets easy. [str] posting, pointless, ripped, passion, goal [aff] Sad -->