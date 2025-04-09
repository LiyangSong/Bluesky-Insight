import os
import time
from collections import deque, defaultdict

from pyarrow import parquet as pq
from tqdm import tqdm


class TrendAnalyzer:
    def __init__(self, queue, sentiment_analyzer):
        self.window_size = 900  # 15 minutes
        self.windows = deque(maxlen=8)  # Save data for 2 hours
        self.queue = queue
        self.sentiment_analyzer = sentiment_analyzer

    def get_current_window_files(self):
        return [
            w['file_path'] for w in self.windows
            if time.time() - w['timestamp'] <= self.window_size
        ]

    async def monitor_queue_and_analysis(self):
        while True:
            file_path = await self.queue.dequeue()
            timestamp = int(os.path.basename(file_path).split('.')[0].split('_')[1])
            self.windows.append({
                "file_path": file_path,
                "timestamp": timestamp,
            })
            print(f"[TrendAnalyzer] Added file: {file_path}")

            start = time.time()
            if len(self.windows) >= 3:
                await self.run_analysis()
                print(f"[TrendAnalyzer] Analysis took {time.time() - start:.2f}s")

    async def run_analysis(self):
        trends = [t[0]for t in self.get_trending_words()]
        print(f"[TrendAnalyzer] Trends: {trends}")

        trend_posts = defaultdict(list)
        for trend in trends:
            trend_posts[trend] = self.fetch_posts_by_trend(trend)

        await self.sentiment_analyzer.parallel_analyze(trend_posts)

    def get_trending_words(self, top_n=10):
        print(f"[TrendAnalyzer] Scanning trending words...")
        word_counts = defaultdict(int)
        # Read post texts from current window
        files = self.get_current_window_files()
        for file in files:
            table = pq.read_table(f"temp_storage/{file}")
            texts = table['text_for_trend'].to_pylist()
            # Count word frequencies
            for text in texts:
                for word in text.split():
                    word_counts[word] += 1

        return sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def fetch_posts_by_trend(self, trend):
        print(f"[TrendAnalyzer] Fetching posts for {trend}...")
        """Fetch posts in current window that contain the trend word."""
        files = self.get_current_window_files()
        post_texts = []
        for file in files:
            table = pq.read_table(f"temp_storage/{file}")
            posts = table.to_pydict()
            post_texts.extend(text for text in posts["text_for_sentiment"]
                              if trend.lower() in text.lower())
        print(f"[TrendAnalyzer] Fetched {len(post_texts)} posts for {trend} in current {self.window_size}s time window")
        return post_texts
