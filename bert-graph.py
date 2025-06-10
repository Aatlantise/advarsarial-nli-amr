import penman
from torch import nn
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import GATv2Conv
from transformers import BertModel, TrainingArguments
from torch_geometric.data import Data
from bert import load_data_from_pickle, set_seed
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch_geometric.data import Data as GeoData
import torch
from tqdm import tqdm
import re
import os
from sklearn.metrics import  confusion_matrix
import numpy as np
import argparse

def concept_embeddings(bert, tokenizer, all_amr_graphs):
    if 'concept2vec.pt' in os.listdir():
        concept2vec = torch.load("concept2vec.pt")
    else:
        # Load BERT
        bert.eval().cuda()

        # Function to strip sense ID (e.g., 'want-01' -> 'want')
        def strip_sense(concept):
            return re.sub(r"-\d\d$", "", concept)

        # Assume you have all AMR graphs in a list `all_amr_graphs`, each with nodes having `.concept`
        amr_vocab = set()
        for graph in all_amr_graphs:
            concepts, _ = parse_amr_graph(graph)
            for concept in concepts:
                base = strip_sense(concept)
                amr_vocab.add(base)

        # Compute CLS embedding for each stripped concept
        def get_embedding(word):
            inputs = tokenizer(word, return_tensors="pt", truncation=True).to(bert.device)
            with torch.no_grad():
                outputs = bert(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()  # CLS token

        concept2vec = {concept: get_embedding(concept) for concept in tqdm(amr_vocab)}

        # Save to file
        torch.save(concept2vec, "concept2vec.pt")

    return concept2vec

class AMRBERTforClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        special_tokens_dict = {'additional_special_tokens': [
            '[TXT]', '[AMR]',
            '[NEW]', '[TAB]',
            ':ARG0', ':ARG1', ':ARG2', ':ARG3', ':ARG4', ':ARG5',
            ':op1', ':op2', ':op3',
            ':mod', ':location', ':time', ':name', ':value', ':topic', ':poss'
                                                                       '(', ')', '/', ':conj', ':and']
        }
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_tokens} tokens.")
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.gnn = GATv2Conv(768, 256)  # Shared GNN for both AMRs
        self.classifier = nn.Linear(768 + 256 * 2, 3)  # 3 classes for MNLI

    def forward(self, input_ids, attention_mask, premise_amr, hypothesis_amr):
        # Text features
        text_features = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0]  # CLS token

        # AMR processing
        premise_emb = torch.stack([
            self.gnn(amr.x, amr.edge_index).mean(dim=0) # ChatGPT recommends torch_geometric.nn.global_mean_pool(...)
            for amr in premise_amr
        ])

        hypothesis_emb = torch.stack([
            self.gnn(amr.x, amr.edge_index).mean(dim=0)
            for amr in hypothesis_amr
        ])

        # Concatenate features
        combined = torch.cat([text_features, premise_emb, hypothesis_emb], dim=1)
        return self.classifier(combined)

def strip_sense(concept):
    return re.sub(r"-\d\d$", "", concept)

def build_graph(concepts, edge_index, concept2vec, embedding_dim):
    x = [concept2vec.get(strip_sense(c), torch.zeros(embedding_dim)) for c in concepts]
    x = torch.stack(x)

    if len(edge_index) > 0:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).T  # shape [2, num_edges]
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)  # empty edge_index if needed

    return GeoData(x=x, edge_index=edge_index_tensor)

def parse_amr_graph(penman_str):
    """Parses AMR string into (concepts, edge_index)"""
    graph = penman.decode(penman_str)
    node_dict = {var: idx for idx, (var, _, _) in enumerate(graph.instances())}

    edge_index = [
        [node_dict[edge.source], node_dict[edge.target]]
        for edge in graph.edges()
        if edge.source in node_dict and edge.target in node_dict
    ]

    concepts = [concept for _, _, concept in graph.instances()]
    return concepts, edge_index

class MNLIWithAMRDataset(Dataset):
    def __init__(self, df, tokenizer, parse_amr_graph, concept2vec, embedding_dim):
        self.data = df
        self.tokenizer = tokenizer
        self.parse_amr_graph = parse_amr_graph
        self.concept2vec = concept2vec
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Tokenize text
        encoding = self.tokenizer(
            row["premise"], row["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        # Parse AMRs
        prem_concepts, prem_edges = self.parse_amr_graph(row["premise_amr"])
        hyp_concepts, hyp_edges = self.parse_amr_graph(row["hypothesis_amr"])

        prem_graph = build_graph(prem_concepts, prem_edges, self.concept2vec, self.embedding_dim)
        hyp_graph = build_graph(hyp_concepts, hyp_edges, self.concept2vec, self.embedding_dim)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "premise_amr": prem_graph,
            "hypothesis_amr": hyp_graph,
            "label": torch.tensor(row["label"], dtype=torch.long)
        }

def create_datasets(tokenizer, train_df, dev_df, hans_df, parse_amr_graph, concept2vec, embedding_dim, batch_size=16):
    train_ds = MNLIWithAMRDataset(train_df, tokenizer, parse_amr_graph, concept2vec, embedding_dim)
    dev_ds = MNLIWithAMRDataset(dev_df, tokenizer, parse_amr_graph, concept2vec, embedding_dim)
    hans_ds = MNLIWithAMRDataset(hans_df, tokenizer, parse_amr_graph, concept2vec, embedding_dim)

    return train_ds, dev_ds, hans_ds

class BERTGraphTrainer:
    def __init__(self, model, train_dataset, dev_dataset, test_dataset, device='cuda', lr=2e-5):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.criterion = CrossEntropyLoss()

    def step(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['label'].to(self.device)

        premise_amr = [g.to(self.device) for g in batch['premise_amr']]
        hypothesis_amr = [g.to(self.device) for g in batch['hypothesis_amr']]

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             premise_amr=premise_amr, hypothesis_amr=hypothesis_amr)

        return {"labels": labels,
                "outputs": outputs}

    def train(self, epochs=3, batch_size=32):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
                _outputs = self.step(batch)

                labels = _outputs["labels"]
                outputs = _outputs["outputs"]

                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}: Training loss = {avg_loss:.4f}")
            self.evaluate(batch_size=batch_size)

    def evaluate(self, batch_size=32):
        dev_loader = DataLoader(self.dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            all_labels = []
            all_preds = []
            for batch in tqdm(dev_loader, desc="Evaluating"):
                _outputs = self.step(batch)

                labels = _outputs["labels"]
                outputs = _outputs["outputs"]

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_labels.append(labels)
                all_preds.append(preds)

        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()

        cm = confusion_matrix(
            all_labels,
            all_preds,
            labels=[0, 1, 2]
        )

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")
        print("Validation Confusion Matrix:\n", cm.tolist())

        hans_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        with torch.no_grad():
            all_labels = []
            all_preds = []
            for batch in tqdm(hans_loader, desc="Evaluating"):
                _outputs = self.step(batch)

                labels = _outputs["labels"]
                outputs = _outputs["outputs"]

                preds = torch.argmax(outputs, dim=1)
                collapsed_preds = np.where(preds.cpu().numpy() == 0, 0, 1)
                correct += (collapsed_preds == labels.cpu().numpy()).sum().item()
                total += labels.size(0)

                all_labels += labels.cpu().tolist()
                all_preds += labels.cpu().tolist()

        cm = confusion_matrix(
            all_labels,
            all_preds,
            labels=[0, 1]
        )

        accuracy = correct / total
        print(f"HANS Accuracy: {accuracy:.4f}")
        print("HANS Confusion Matrix:\n", cm.tolist())


        self.model.train()

    def collate_fn(self, batch):
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'premise_amr': [item['premise_amr'] for item in batch],
            'hypothesis_amr': [item['hypothesis_amr'] for item in batch],
            'label': torch.tensor([item['label'] for item in batch], dtype=torch.long)
        }


def main(args):
    set_seed(args.seed)
    model = AMRBERTforClassification()

    mnli_train, mnli_dev, hans_df = load_data_from_pickle(args)

    if args.debug:
        mnli_train = mnli_train[:1000]
        mnli_dev = mnli_dev[:1000]
        hans_df = hans_df[:1000]

    # Load precomputed BERT embeddings of AMR concepts
    concept2vec = concept_embeddings(model.bert, model.tokenizer, mnli_train["premise_amr"] + mnli_train["hypothesis_amr"])
    embedding_dim = next(iter(concept2vec.values())).shape[0]

    train_ds, dev_ds, hans_ds = create_datasets(
        tokenizer=model.tokenizer,
        train_df=mnli_train,
        dev_df=mnli_dev,
        hans_df=hans_df,
        parse_amr_graph=parse_amr_graph,  # from above
        concept2vec=concept2vec,
        embedding_dim=embedding_dim,
        batch_size=32
    )

    trainer = BERTGraphTrainer(
        model=model,
        train_dataset=train_ds,
        dev_dataset=dev_ds,
        test_dataset=hans_ds
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--debug", type=bool, default=False, help="Whether to use debug mode")
    parser.add_argument("--tqdm", type=bool, default=False, help="Whether to use debug mode")
    parser.add_argument("--eval_only", type=bool, default=False, help="Does not train model if false")
    args = parser.parse_args()

    args.desc = "amr-with-text-nospace"

    print(args)
    main(args)

