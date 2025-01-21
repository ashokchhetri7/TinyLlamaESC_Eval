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
    model_name: str = "Ashokajou51/mi-tinyllama-plm",  # Update your model name
    revision: Optional[str] = "epoch-1",  # Update your model name
):
    dataset_name = "heegyu/mi-KEMI"
    dataset = load_dataset(dataset_name, split="test")

    def map_conversations(item):
        dialog = item["dialog"]
        knowledge = item["knowledge"]
        heal = item["heal"]
        target = item["target"]
        strategy = item["strategy"]

        convs = []

        user_prompt = f"{item['problem_type']}\n{dialog}\n{knowledge}\n{heal}"
        convs.append({"role": "user", "content": user_prompt})

        convs.append({"role": "assistant", "content": f"[{strategy}] {target}"})

        return {"conversations": convs}

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
        convs = item["conversations"]

        for i, conv in enumerate(convs):
            if conv["role"] == "user":
                prompt = conv["content"]
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
                gens = model.generate(**inputs, **gen_kwargs)

                response = gens[:, inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(response[0])

                # print(prompt, ":;;;;;;;;;;;;;;;;;;;;;;;;;;;")

                fout.write({
                    "turn_index": i,
                    **conv,
                    "prediction": response
                })

    fout.close()

if __name__ == "__main__":
    fire.Fire(main)