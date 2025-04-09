import json
import os
import re
import time
from contextlib import asynccontextmanager

import nltk
import pyarrow as pa
from pyarrow import parquet as pq
from bloom_filter import BloomFilter

import websockets
from dotenv import load_dotenv
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from tqdm.asyncio import tqdm


class JetstreamProcessor:
    def __init__(self, queue):
        load_dotenv()
        self.jetstream_wss = os.getenv('JETSTREAM_WSS')

        nltk.download('punkt_tab')
        nltk.download('stopwords')

        # Use Bloom Filter to reduce memory usage
        self.seen_posts = BloomFilter(
            max_elements=100_000,
            error_rate=0.001
        )

        self.temp_storage = []

        # Queue service that stores post temp storage files
        self.queue = queue

        # Periodically persist data from temp storage
        self.persist_interval = 300  # 5 minutes
        self.last_saved_time = time.time()

    @asynccontextmanager
    async def _auto_managed_connection(self):
        """Connect to Jetstream websocket that supports auto retries."""
        retry_count = 0
        max_retries = 5

        while True:
            try:
                async with websockets.connect(
                    self.jetstream_wss,
                    ping_interval=30,
                    ping_timeout=60,
                    max_queue=1024
                ) as ws:
                    print("[JetstreamProcessor] Connected to Jetstream WS")
                    retry_count = 0
                    yield ws
            except (websockets.ConnectionClosed, OSError) as e:
                if retry_count > max_retries:
                    raise RuntimeError(f"[JetstreamProcessor] Failed to connect to Jetstream WS exceeding max retry times: {e}")
                print(f"[JetstreamProcessor] Connection closed. Retrying...")
                retry_count += 1

    async def process_jetstream(self):
        """Read post message from Jetstream websocket and process it."""
        async with self._auto_managed_connection() as ws:
            await ws.send(json.dumps({
                "wantedCollections": ["app.bsky.feed.post"]
            }))

            try:
                # Progress bar for streaming posts
                self.tqdm_bar = tqdm(desc="[JetstreamProcessor] Processing post messages from Jetstream WS:",
                                 bar_format='{desc} {n_fmt} posts [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                                 unit="posts", dynamic_ncols=True, leave=False)

                async for message in ws:
                    await self._process_post(message)
            except websockets.exceptions.ConnectionClosed as e:
                print(f"[JetstreamProcessor] Connection Closed: {e}")
            finally:
                await self.persist_storage()

    async def _process_post(self, post_data):
        """Key logic to process single post message."""
        data = json.loads(post_data)

        # Only process new post operations
        if (data is None or data["kind"] != "commit" or
                data["commit"]["operation"] != "create" or
                data["commit"]["collection"] != "app.bsky.feed.post"):
            return

        # Only process English contents
        if ("langs" not in data["commit"]["record"].keys() or
                "en" not in data["commit"]["record"]["langs"]):
            return

        cid = data["commit"]["cid"]
        if cid in self.seen_posts:
            return
        self.seen_posts.add(cid)

        text = data["commit"]["record"]["text"]

        #  Temp storage
        self.temp_storage.append({
            "cid": cid,
            "text_raw": text,
            "text_for_trend": self._preprocess_text_for_trend(text),
            "text_for_sentiment": self._preprocess_text_for_sentiment(text),
            "timestamp": data["commit"]["record"]["createdAt"]
        })

        # Update progress bar
        self.tqdm_bar.update(1)

        # Persist storage periodically
        if time.time() - self.last_saved_time > self.persist_interval:
            tqdm.write('')
            await self.persist_storage()

    @staticmethod
    def _preprocess_text_for_trend(text):
        """Preprocess text for trending analysis. Apply multiple preprocessing steps to keep only keywords."""
        # Remove URL and @
        preprocessed_text = re.sub(r'http\S+|@\w+', '', text)

        preprocessed_text = preprocessed_text.lower()

        # Remove punctuation
        preprocessed_text = re.sub(r'[^\w\s]', '', preprocessed_text)

        # Remove Stop words
        tokens = word_tokenize(preprocessed_text)
        stops = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stops]

        # Stem words
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

        return ' '.join(tokens)

    @staticmethod
    def _preprocess_text_for_sentiment(text):
        """Preprocess text for sentiment analysis. Only apply simple preprocessing to keep original meanings."""
        # Replace URL and @ with labels
        preprocessed_text = re.sub(r'http\S+', '[URL]', text)
        preprocessed_text = re.sub(r'@\w+', '[MENTION]', preprocessed_text)

        # Keep basic punctuations
        preprocessed_text = re.sub(r"[^a-zA-Z0-9\s!?.,']", '', preprocessed_text)

        return preprocessed_text

    async def persist_storage(self):
        # Persist data from temp storage
        if not self.temp_storage:
            return

        filename = f"jetstream_{int(time.time())}.parquet"

        try:
            table = pa.Table.from_pydict({
                "cid": pa.array([post["cid"] for post in self.temp_storage]),
                "text_raw": pa.array([post["text_raw"] for post in self.temp_storage]),
                "text_for_trend": pa.array([post["text_for_trend"] for post in self.temp_storage]),
                "text_for_sentiment": pa.array([post["text_for_sentiment"] for post in self.temp_storage]),
                "timestamp": pa.array([post["timestamp"] for post in self.temp_storage])
            })
            pq.write_table(table, f"temp_storage/{filename}")

            # Add file to queue
            await self.queue.enqueue(filename)

            print(f"[JetstreamProcessor] Saved {len(self.temp_storage)} posts to {filename}")

            # Reset temp storage
            self.temp_storage = []
            self.last_saved_time = time.time()

        except IOError as e:
            print(f"[JetstreamProcessor] Failed to persist temp storage: {e}")

