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
  model_name: str = "Ashokajou51/esconv-sorted-role-tinyllama-plm", # the nameo fo lakdfjlasdf 
  revision: Optional[str] = "epoch-3", # the sepci version of model to use
):
  # dataset_name = "heegyu/augesc"
  dataset_name = "Ashokajou51/ESConv_Sorted"
  dataset = load_dataset(dataset_name, split="test")

  def map_conversations(item):
          dialog = item["dialog"]
          convs = []
          instruct = {
              "role": "system",
              "content": "As an empathetic emotional support chatbot, your primary goal is to engage with users sharing their experiences or struggles related to emotional challenges such as the pandemic, job stress, burnout, depression, and anxiety. Your interaction should unfold in a structured manner, aiming to explore issues, offer affirmations, and suggest actionable advice by the end of the conversation. Use the following strategies in your responses to ensure a meaningful and supportive dialogue: Early Conversation - Explore Issues: Question Begin by asking open-ended questions to understand the user's situation deeply. This encourages users to share more about their feelings and experiences. For example, Question 'Can you tell me more about what's been causing you stress lately?' Mid-Conversation - Affirm and Reassure: Affirmation and Reassurance After understanding the user's situation, offer affirmations and reassurance. Acknowledge their feelings and validate their experiences to build trust and empathy. For example, Affirmation and Reassurance 'It's completely understandable to feel overwhelmed in situations like these.' End of Conversation - Suggest Actions: Providing Suggestions Once a connection has been established and the user's feelings have been validated, suggest practical steps they can take to address their concerns. Tailor your advice to their unique situation, ensuring it feels relevant and helpful. For example, Providing Suggestions 'Have you considered speaking to a professional who can provide you with tailored support and guidance?' Addition Strategies Restatement or Paraphrasing to ensure understanding and clarify the user's messages. Reflection of Feelings to show empathy and understanding of the user's emotions. Self-disclosure sparingly, to build rapport by sharing relevant personal experiences or feelings if applicable. Information to provide factual advice or resources that the user might find helpful. System Utterance Format: Use the format Strategy 'System utterance' to guide your responses. This structure helps in applying the correct empathetic strategy in response to the user's needs and issues"

              # "content": "As an empathetic emotional support chatbot, for given [User] dialogues, deeply explore, suggest and ask to take actions, firstly, through caring questions and [\\xReact][\\xIntent][\\xWant][\\xNeed] references understand the user's pandemic/emotional-related situation (job stress, burnout, depression, anxiety, etc.). Early on, validate their difficult feelings as understandable reactions. Mid-conversation, gently admit their challenges are tough while reflecting your empathetic understanding. Once an empathetic connection is established, provide suggestion and response with [\\resp] reference for users that might be in [\\str] stituations having the states of [aff], understand the flow as personalized guidance based on their unique revealed needs, not demands. Although the process doesn't sometimes doesn't follow explore, suggest and action format, but frequently check if responses feel appropriate, supportive, and true to their experience."
          }
          convs.append(instruct)
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
                      # content += f"\\n  Considering the user's statements, the below given [Knowledge] and [Heal] context is used to generate the next strategy prediction and assistant response. Here, [xReact] is the user's emotional state, comparable to [aff]. [xIntent] is the reason behind the user's situation, similar to [str]. [xWant], [xNeed], and [xEffect] represent the user's cognitive needs and desired outcomes, which should inform the final [resp] response. [resp] provides some guidance on the nature of the expected response. \\n [Knowledge]: {uttr['knowledge']}"
                      content += f"{uttr['knowledge']}"
                  if prev_heal:
                      content += f"\\n [Heal]: {prev_heal}"
              if speaker == "sys":
                  text = f"[{uttr['strategy']}] {text}"
                  prev_heal = uttr.get("heal", None)
              content += text
              convs.append({"role": speaker2role[speaker], "content": content})

          return {"conversations": convs,  "dialog": dialog}

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