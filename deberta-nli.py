import pickle as pkl
import torch
from tqdm import tqdm
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

with open("pair_claims.pkl", 'rb') as f:
    claim_pairs = pkl.load(f)

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xxlarge-mnli")

def DeBERTaNLI(tokenizer, model, claim, verbose=False):
    premise, hypothesis = claim
    # e.g., premise, hypothesis = ["I love you.", "I like you."]
    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis, return_token_type_ids=True, truncation=True)
    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)
    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=None)

    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one
    argmax_id = predicted_probability.index(max(predicted_probability))
    if argmax_id == 0:
        pred = "contradict"
    elif argmax_id == 1:
        pred = "neutral"
    elif argmax_id == 2:
        pred = "entail"

    if verbose:
        print("Premise:", premise)
        print("Hypothesis:", hypothesis)
        print("Contradiction:", predicted_probability[0])
        print("Neutral:", predicted_probability[1])
        print("Entailment:", predicted_probability[2])
        print("Prediction:", pred)

    return [premise, hypothesis, predicted_probability[0], predicted_probability[1], predicted_probability[2], pred]

with open("deberta-v2-xxlarge-nli-outputs.tsv", "wt") as f:
    tsv_writer = csv.writer(f, delimiter="\t")
    for c in tqdm(claim_pairs):
        out = DeBERTaNLI(tokenizer, model, c)
        tsv_writer.writerow(out)

