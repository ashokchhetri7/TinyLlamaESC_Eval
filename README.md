# Rubric-based ESC Analysis on Context-size and External Knowledge in LLM

This repository provides a comprehensive framework for generating dialogues, evaluating model performance, and analyzing results based on rubric-based scoring. The flow of the program is structured as follows:

1. **Generate Dialogues**: Run the script to generate conversational responses using the model and dataset.
2. **Process the Generated JSON Dataset**: Save the outputs as a JSONLines file for further evaluation.
3. **Evaluate and Analyze Results**: Run the evaluation script to score the generated responses. Pre-calculated results are available in the `/Results` folder for reference.

   - The evaluation script processes the generated dataset, sends requests to a scoring server, and calculates the mean score based on rubric criteria. The results are displayed in the terminal, and pre-calculated results are saved in the `/Results` folder.

---

## Features

- **Dataset Handling**: Uses the ESConv dataset from Hugging Face.
- **Pre-trained Model**: Supports TinyLlama or any other Hugging Face-compatible language model.
- **In-Context Learning**: Generates responses using conversational history.
- **Customizable Parameters**: Adjust model name, dataset, and generation settings.
- **Results Logging**: Outputs predictions to a JSONLines file for further analysis.
- **Rubric-based Evaluation**: Scores generated responses based on fluency, relevance, and usefulness.

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

### Step 1: Generate Dialogues
Run the script to generate dialogues based on the provided dataset and model:

```bash
python <script_name>.py --model_name "Ashokajou51/esconv-sorted-incontext-tinyllama-plm" --revision "epoch-3"
```

#### Arguments

- `model_name` (str): The Hugging Face model name. Default: `Ashokajou51/esconv-sorted-incontext-tinyllama-plm`
- `revision` (Optional[str]): The model revision or checkpoint. Default: `epoch-3`

### Step 2: Process the Generated Dataset
The output will be saved as a JSONLines file named `GeneratedDataset/<model_name>_<revision>.json`. This file contains the generated dialogues and relevant metadata:

- `turn_index`: The turn index in the conversation.
- `context`: The conversation context leading to the prediction.
- `prediction`: The model's predicted response.
- `reference`: The ground truth response for the given context.

### Step 3: Evaluate and Analyze Results
Run the evaluation script to score the generated dialogues using rubric-based criteria:

```bash
python Evaluation/eval_prometheus_incontext.py
```

The evaluation script processes the generated dataset, calculates scores based on rubric criteria, and displays the results in the terminal. Pre-calculated results are saved in the `/Results` folder for further analysis.

### Results Table

#### Table: Results for Different Model Configurations

\label{table:tableforablation}
\begin{tabular*}{\hsize}{@{}@{\extracolsep{\fill}}ccc|ccccc@{}}
\toprule
\multicolumn{3}{c|}{\textbf{Model Configuration}} & \multicolumn{5}{c}{\textbf{Rubrics Scores}} \\
\cmidrule(lr){1-3}
\cmidrule(lr){4-8}
\textbf{Model} & \textbf{Context Size} & \textbf{External Knowledge} & \textbf{Empathy} & \textbf{Fluency} & \textbf{Identify} & \textbf{Comfort} & \textbf{Suggest}\\
\midrule
KEMI & Blocks & strategy-comet-heal & 2.04 & 2.16 & 1.85 & 1.75 & 1.51 \\
KEMI & Blocks & strategy & 2.04 & 2.50 & 1.79 & 1.98 & 1.51 \\
\midrule
Tiny-Know & Blocks & strategy-comet-heal & 2.24 & 2.48 & 1.86 & 1.70 & 1.79 \\
Tiny-Stand & Blocks & strategy & 2.29 & 2.56 & 2.11 & 2.23 & 2.44 \\
\midrule
Tiny-Role & Whole & strategy-comet-heal-role & 1.87 & 2.03 & 1.72 & 1.50 & 1.48 \\
Tiny-Stand & Whole & strategy & \textbf{2.30} & \textbf{2.57} & \textbf{2.14} & \textbf{2.25} & \textbf{2.74} \\
\bottomrule
\end{tabular*}
\end{table}

---

## How It Works

1. **Dataset Loading**: The ESConv dataset is loaded and preprocessed into conversational turns.
2. **Mapping Conversations**: Dialogues are structured into roles (`user` and `assistant`) with contextual metadata.
3. **Tokenization**: Conversational history is tokenized using the specified model's tokenizer.
4. **Response Generation**: The model generates responses based on the conversational context.
5. **Evaluation**: Predictions are scored using a rubric-based system, considering fluency, relevance, and constructiveness.
6. **Result Logging**: Predictions and evaluation scores are saved for analysis.

---

## Rubric-based Evaluation

Generated responses are evaluated based on the following criteria:

1. **Fluency**: The coherence and smoothness of the response.
2. **Identification**: The effectiveness of the speaker in identifying and addressing the seeker’s problems.
3. **Empathy**: The level of empathetic understanding displayed by the speaker towards the seeker’s feelings and situation.
4. **Suggestion**: The quality of the suggestions offered by the speaker.
5. **Comforting**: The ability to make the seeker calm and relaxed by providing emotional support.

Evaluation scores and feedback are recorded in the `/Results` folder for detailed analysis.

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
    do_sample=False,
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
- [ESConv Dataset](https://huggingface.co/datasets/Ashokajou51/ESConv_Sorted)
- [Ashokajou51](https://huggingface.co/Ashokajou51)

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contributing

Feel free to open issues or submit pull requests to improve this project. Feedback and suggestions are always welcome!

