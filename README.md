# Conversational Model Evaluation with ESConv

This repository provides a script for evaluating conversational AI models using the ESConv dataset. The script leverages the TinyLlama pre-trained language model to predict responses in a conversational context and compares them with the dataset's ground truth. It utilizes the Hugging Face Transformers library and PyTorch.

---

## Features
- **Dataset Handling**: Uses the ESConv dataset from Hugging Face.
- **Pre-trained Model**: Supports TinyLlama or any other Hugging Face-compatible language model.
- **In-Context Learning**: Generates responses using conversational history.
- **Customizable Parameters**: Adjust model name, dataset, and generation settings.
- **Results Logging**: Outputs predictions to a JSONLines file for further analysis.

---

## Requirements

Ensure you have the following dependencies installed:

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- Fire
- TQDM
- jsonlines

You can install the dependencies using the following command:

```bash
pip install torch transformers datasets fire tqdm jsonlines
```

---

## Usage

To run the script, use the following command in your terminal:

```bash
python <script_name>.py --model_name "Ashokajou51/esconv-tinyllama-plm" --revision "epoch-3"
```

### Arguments

- `model_name` (str): The Hugging Face model name. Default: `Ashokajou51/esconv-tinyllama-plm`
- `revision` (Optional[str]): The model revision or checkpoint. Default: `epoch-3`

### Example

```bash
python main.py --model_name "Ashokajou51/esconv-tinyllama-plm" --revision "epoch-3"
```

---

## Output

The script generates a JSONLines file named `<model_name>_<revision>.json`. Each line contains the following information:

- `turn_index`: The turn index in the conversation.
- `role`: The role of the speaker (user/assistant).
- `content`: The ground truth content of the response.
- `prediction`: The model's predicted response.

---

## How It Works

1. **Dataset Loading**: The ESConv dataset is loaded and preprocessed into conversational turns.
2. **Mapping Conversations**: Dialogues are structured into roles (`user` and `assistant`) with contextual metadata.
3. **Tokenization**: Conversational history is tokenized using the specified model's tokenizer.
4. **Response Generation**: The model generates responses based on the conversational context.
5. **Result Logging**: Predictions are saved alongside the ground truth for evaluation.

---

## Customization

### Modify Dataset
You can replace the dataset with another Hugging Face dataset by changing the `dataset_name` in the script:

```python
dataset_name = "your_dataset_name"
```

### Adjust Generation Parameters
Edit the `gen_kwargs` dictionary to fine-tune the model's generation behavior:

```python
gen_kwargs = dict(
    do_sample=True,
    max_new_tokens=128,
)
```

---

## Notes
- Ensure that your system has a compatible GPU with CUDA enabled for faster processing.
- Modify the `device` variable in the script if you want to use a different hardware setup (e.g., CPU).

---

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [ESConv Dataset](https://huggingface.co/datasets/Ashokajou51/ESConv_Original)
- [Ashokajou51](https://huggingface.co/Ashokajou51)

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contributing

Feel free to open issues or submit pull requests to improve this project. Feedback and suggestions are always welcome!

