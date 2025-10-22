import torch
from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertModel
from typing import List, Union

class TextEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the text encoder with either CLIP or BERT
        Args:
            model_name: Name of the model to use (CLIP or BERT)
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if "clip" in model_name.lower():
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name).to(self.device)

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of texts into embeddings
        Args:
            texts: List of text strings to encode
        Returns:
            Tensor of text embeddings
        """
        if "clip" in self.model_name.lower():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_text_features(**inputs)
            return outputs / outputs.norm(dim=-1, keepdim=True)
        else:
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)  # Mean pooling

    def encode_single_text(self, text: str) -> torch.Tensor:
        """
        Encode a single text string into an embedding
        Args:
            text: Text string to encode
        Returns:
            Tensor of text embedding
        """
        return self.encode_texts([text])

    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of the text embeddings
        Returns:
            Dimensionality of embeddings
        """
        if "clip" in self.model_name.lower():
            return self.model.text_projection.shape[1]
        else:
            return self.model.config.hidden_size

