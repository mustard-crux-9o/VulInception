"""UnixCoder (RoBERTa) based code embedder for sequence and batch embedding."""
from typing import List
import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer


class CodeBERTEmbedder:
    def __init__(self, model_name: str, device: torch.device) -> None:
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def get_embedding_sequence(self, code: str, max_length: int = 512) -> np.ndarray:
        if not code or not isinstance(code, str):
            return np.zeros((1, self.model.config.hidden_size), dtype=np.float32)
        inputs = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def get_batch_embeddings(self, codes: List[str], max_length: int = 512) -> List[np.ndarray]:
        """Return per-token embeddings for each code string in the batch."""
        embeddings = [None] * len(codes)
        valid_indices = []
        valid_codes = []
        for idx, code in enumerate(codes):
            if not code or not isinstance(code, str):
                embeddings[idx] = np.zeros((1, self.model.config.hidden_size), dtype=np.float32)
            else:
                valid_indices.append(idx)
                valid_codes.append(code)
        if not valid_codes:
            return embeddings
        inputs = self.tokenizer(valid_codes, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state.detach().cpu()
        attention_mask = inputs["attention_mask"].detach().cpu()
        for batch_idx, original_idx in enumerate(valid_indices):
            seq_len = int(attention_mask[batch_idx].sum().item())
            emb = hidden_states[batch_idx, :seq_len].numpy()
            embeddings[original_idx] = emb.astype(np.float32, copy=False)
        return embeddings
