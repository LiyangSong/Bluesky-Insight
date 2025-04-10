import asyncio
import os
import time
from collections import deque, defaultdict

import numpy as np
from dotenv import load_dotenv
from pyarrow import parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer

from apps.worker.sentiment_analyzer import SentimentAnalyzer


class TrendAnalyzer:
    def __init__(self, queue):
        load_dotenv()
        self.window_size = int(os.getenv("ANALYSIS_WINDOW_SIZE"))  # Analyze data within current window
        self.windows = deque(maxlen=6)  # Keep data for 6 hours
        self.queue = queue
        self.persist_interval = int(os.getenv("PERSIST_INTERVAL"))
        self.top_n = int(os.getenv("TOP_N_TRENDS"))

    def get_current_window_files(self):
        return [
            w['file_path'] for w in self.windows
            if time.time() - w['timestamp'] <= self.window_size
        ]

    async def monitor_queue_and_analysis(self):
        while True:
            await asyncio.sleep(self.persist_interval)
            file_path = await self.queue.dequeue()
            timestamp = int(os.path.basename(file_path).split('.')[0].split('_')[1])
            self.windows.append({
                "file_path": file_path,
                "timestamp": timestamp,
            })
            print(f"[TrendAnalyzer] Added file: {file_path}")

            start = time.time()
            await self.run_analysis()
            print(f"[TrendAnalyzer] Analysis took {time.time() - start:.2f}s")

    async def run_analysis(self):
        trends = self.get_trending_words()
        print(f"[TrendAnalyzer] Trends: {trends}")

        trend_posts = defaultdict(list)
        for trend in trends:
            trend_posts[trend] = self.fetch_posts_by_trend(trend)

        async with SentimentAnalyzer() as analyzer:
            await analyzer.parallel_analyze(trend_posts)

    def get_trending_words(self):
        print(f"[TrendAnalyzer] Scanning trending words...")
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Catch phrases
            max_df=0.7,  # Ignore very common tokens
            min_df=10  # Ignore very rare tokens
        )

        # Read post texts from current window
        files = self.get_current_window_files()
        print(f"[TrendAnalyzer] Analyzing {len(files)} files in current window...")

        all_texts = []
        for file in files:
            table = pq.read_table(f"temp_storage/{file}")
            text = table['text_for_trend'].to_pylist()
            all_texts.extend(text)
            print(f"[TrendAnalyzer] Loaded {len(text)} posts from {file}")
        print(f"[TrendAnalyzer] Total texts for analysis: {len(all_texts)}")

        # Calculate TF-IDF
        tf_idf_vectors = vectorizer.fit_transform(all_texts)
        tfidf_scores = tf_idf_vectors.sum(axis=0).A1
        sorted_indices = np.argsort(tfidf_scores)[::-1]
        return vectorizer.get_feature_names_out()[sorted_indices[:self.top_n]].tolist()

    def fetch_posts_by_trend(self, trend):
        print(f"[TrendAnalyzer] Fetching posts for trend: {trend}...")
        """Fetch posts in current window that contain the trend word."""
        files = self.get_current_window_files()
        post_texts = []
        for file in files:
            table = pq.read_table(f"temp_storage/{file}")
            posts = table.to_pydict()
            for idx, text_for_trend in enumerate(posts["text_for_trend"]):
                if trend in text_for_trend:
                    post_texts.extend(posts["text_for_sentiment"][idx])
        print(f"[TrendAnalyzer] Fetched {len(post_texts)} posts for {trend} in current window")
        return post_texts
