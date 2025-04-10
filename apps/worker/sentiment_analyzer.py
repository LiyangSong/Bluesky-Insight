import asyncio
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class PostTextsDataset(Dataset):
    def __init__(self, texts):
        super().__init__()
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class SentimentAnalyzer:
    def __init__(self):
        load_dotenv()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 32

        self.tokenizer = AutoTokenizer.from_pretrained(os.getenv("SENTIMENT_MODEL"))
        self.model = AutoModelForSequenceClassification.from_pretrained(os.getenv("SENTIMENT_MODEL"))
        if self.device == 'cuda':
            self.model = self.model.half()  # FP16
            self.model = torch.compile(self.model)  # Optimize with TorchDynamo

        self.executor = ThreadPoolExecutor(max_workers=4)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.executor.shutdown(wait=True)

    async def parallel_analyze(self, trend_posts):
        futures = [
            asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.batch_analyze,
                trend, posts
            ) for trend, posts in trend_posts.items()
        ]
        await asyncio.gather(*futures)

    def batch_analyze(self, trend, posts):
        dataset = PostTextsDataset(posts)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn
        )

        sentiment_counts = defaultdict(int)
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Analyzing sentiment for trend: {trend}", leave=False):
                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                }
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=1)

                probs = probs.cpu().float().numpy()

                for prob in probs:
                    max_idx = np.argmax(prob)
                    sentiment_counts[max_idx] += 1

        total = sum(sentiment_counts.values())
        sentiment_stats = {
            "trend": trend,
            "sample_size": total,
            "positive": sentiment_counts[0] / total,
            "neutral": sentiment_counts[1] / total,
            "negative": sentiment_counts[2] / total
        }

        print(f"[SentimentAnalyzer] {trend} — Pos: {sentiment_stats['positive']:.2%}, "
              f"Neu: {sentiment_stats['neutral']:.2%}, Neg: {sentiment_stats['negative']:.2%}")

        with open(f'output/sentiment_{trend}_{int(time.time())}.json', 'w') as f:
            json.dump(sentiment_stats, f)

    def _collate_fn(self, batch):
        tokenized_inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
        }


