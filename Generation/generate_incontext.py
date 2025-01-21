import fire
import torch
import json, jsonlines
from tqdm.auto import tqdm
from copy import deepcopy

torch.set_grad_enabled(False)

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from typing import List, Dict, Any, Optional


def main( #makes two optional arguements
  model_name: str = "Ashokajou51/esconv-sorted-incontext-tinyllama-plm", # the nameo fo lakdfjlasdf 
  revision: Optional[str] = "epoch-3", # the sepci version of model to use
):
  # dataset_name = "heegyu/augesc"
  dataset_name = "Ashokajou51/ESConv_Sorted"
  dataset = load_dataset(dataset_name, split="test")

  def map_conversations(item):  #while copying remember to remove the self in here
      dialog = item["dialog"]
      convs = []
      speaker2role = {"sys": "assistant", "usr": "user"}
      prev_heal = None
      for i in range(len(dialog)):
          uttr = dialog[i]
          speaker = uttr["speaker"]
          text = uttr["text"]
          content= ""
          if speaker == "usr":
              content += f"User: {text}"

              if "knowledge" in uttr:
                  content += f"\\n  Considering the user's statements, the below given [Knowledge] and [Heal] context is used to generate the next strategy prediction and assistant response. Here, [xReact] is the user's emotional state, comparable to [aff]. [xIntent] is the reason behind the user's situation, similar to [str]. [xWant], [xNeed], and [xEffect] represent the user's cognitive needs and desired outcomes, which should inform the final [resp] response. [resp] provides some guidance on the nature of the expected response. \\n [Knowledge]: {uttr['knowledge']}"
                  content += f"{uttr['knowledge']}"
              if prev_heal:
                  content += f"\\n [Heal]: {prev_heal}"
          if speaker == "sys":
              text = f"[{uttr['strategy']}] {text}"
              prev_heal = uttr.get("heal", None)
          content += text
          convs.append({"role": speaker2role[speaker], "content": content})

      return {"conversations": convs, "dialog": dialog}  # and remember to add the "dialog" here.

  dataset = dataset.map(map_conversations)

  device="cuda:0"
  tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
  tokenizer.truncation_side = 'left'
  model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision).eval().half().to(device)

  gen_kwargs = dict(
      do_sample=False,
      max_new_tokens=128,
  )
  
  if revision:
    model_name = model_name.replace("/", "_")
    save_file = f"./GeneratedDataset{model_name}_{revision}.json"
  else:
    save_file = f"./GeneratedDataset{model_name}.json"

  fout = jsonlines.open(save_file, "w")

  for item in tqdm(dataset, position=0, desc="Generating"):
    convs = item["conversations"]
    for i in range(0, len(convs)):
      context = convs[:i]

      if i == 0 and convs[0]["role"] == "assistant":
        prompt = "<|assistant|>\n"
      else:
        if convs[i]['role'] == 'user':
          continue
        prompt = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)

      # prompt = prompt + "["
      inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
      gens = model.generate(**inputs, **gen_kwargs)

      response = gens[:, inputs["input_ids"].shape[1]:]


      response = tokenizer.decode(response[0])
      # print(prompt, "-->")
      # print("Prediction", response)
      # print("GT", convs[i]["content"])

      fout.write({
          "turn_index": i,
          "context": convs[:i + 1],
          "prediction": response
      })

  fout.close()

if __name__ == "__main__":
   fire.Fire(main)