import fire
import torch
import json, jsonlines
from tqdm.auto import tqdm
from copy import deepcopy
import sys

torch.set_grad_enabled(False)

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from typing import List, Dict, Any, Optional

# Ashokajou51/esconv-sorted-incontext-tinyllama-plm

def main(                     
  model_name: str = "Ashokajou51/esconv-tinyllama-plm",  # Update your model name
  revision: Optional[str] = "epoch-3",  # Update your model name
):
  # dataset_name = "heegyu/augesc"
  dataset_name = "Ashokajou51/ESConv_Original"
  dataset = load_dataset(dataset_name, split="test")

  def map_conversations(item):
      # item = json.loads(item["text"])
      dialog = item["dialog"]
      convs = []
      speaker2role = {
          'sys': "assistant",
          'usr': 'user'
      }

      for i in range(0, len(dialog) - 1):
          uttr = dialog[i]
          speaker = uttr["speaker"]

          if speaker == "sys":
              text = "[{strategy}] {text}".format(**uttr)
          else:
              text = uttr["text"]

          convs.append({
              "role": speaker2role[speaker],
              "content": text,
          })

      return {
          "conversations": convs,
          "dialog": dialog
      }

  dataset = dataset.map(map_conversations)

  device="cuda:0"
  tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
  tokenizer.truncation_side = 'left'
  model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision).eval().half().to(device)

  gen_kwargs = dict(
      do_sample=True,
      max_new_tokens=128,
  )
  
  if revision:
    model_name = model_name.replace("/", "_")
    save_file = f"{model_name}_{revision}.json"
  else:
    save_file = f"{model_name}.json"

  fout = jsonlines.open(save_file, "w")

  for item in tqdm(dataset):
    dialog = item["dialog"]
    convs = item["conversations"]
    for i in range(0, len(convs)):
      context = convs[:i]

      if convs[i]['role'] == 'user':
        continue

      prompt = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
      # prompt = prompt + "["
      inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
      gens = model.generate(**inputs, **gen_kwargs)

      response = gens[:, inputs["input_ids"].shape[1]:]


      response = tokenizer.decode(response[0])
      # print(prompt, "----------->")

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