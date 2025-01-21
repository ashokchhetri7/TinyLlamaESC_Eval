import json

def calculate_stats(file_path):
    total_utterances = 0
    total_words_per_utterance = 0
    total_turns_per_dialogue = 0
    total_words_per_dialogue = 0
    num_dialogues = 0

    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            dialog = data['dialog']

            num_turns = len(dialog)
            total_turns_per_dialogue += num_turns
            num_dialogues += 1

            words_per_dialogue = 0
            for utterance in dialog:
                text = utterance['text']
                words = len(text.split())
                words_per_dialogue += words
                total_words_per_utterance += words
                total_utterances += 1

            total_words_per_dialogue += words_per_dialogue

    avg_words_per_utterance = total_words_per_utterance / total_utterances
    avg_turns_per_dialogue = total_turns_per_dialogue / num_dialogues
    avg_words_per_dialogue = total_words_per_dialogue / num_dialogues

    return avg_words_per_utterance, avg_turns_per_dialogue, avg_words_per_dialogue, num_dialogues, total_utterances

# Example usage
file_path = '/hdd1/ashok/KEMI/blenderbot/_reformat/esconv/Original_Sorted/test_sorted_turn.txt'
avg_words_per_utterance, avg_turns_per_dialogue, avg_words_per_dialogue, num_dialogues, total_utterances = calculate_stats(file_path)

print(f"Average words per utterance: {avg_words_per_utterance}")
print(f"Average turns per dialogue: {avg_turns_per_dialogue}")
print(f"Average words per dialogue: {avg_words_per_dialogue}")
print(f"Average words per dialogue: {num_dialogues}")
print(f"Total utterances: {total_utterances}")
