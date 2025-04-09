import json
import os
import re
import time
from contextlib import asynccontextmanager

import nltk
from bloom_filter import BloomFilter

import websockets
from dotenv import load_dotenv
from nltk import word_tokenize
from nltk.corpus import stopwords


class JetstreamProcessor:
    def __init__(self):
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

        # Periodically persist data from temp storage
        self.persist_interval = 300  # 5 minutes
        self.last_saved_time = time.time()

    @asynccontextmanager
    async def _auto_managed_connection(self):
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
                    print("Connected to Jetstream WS")
                    retry_count = 0
                    yield ws
            except (websockets.ConnectionClosed, OSError) as e:
                if retry_count > max_retries:
                    raise RuntimeError(f"Failed to connect to Jetstream WS exceeding max retry times: {e}")
                print(f"Connection closed. Retrying...")
                retry_count += 1

    async def process_jetstream(self):
        async with self._auto_managed_connection() as ws:
            await ws.send(json.dumps({
                "wantedCollections": ["app.bsky.feed.post"]
            }))

            try:
                async for message in ws:
                    await self._process_post(message)
            except websockets.exceptions.ConnectionClosed as e:
                print(f"Connection Closed: {e}")
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
        preprocessed_text = self._preprocess_text(text)
        created_at = data["commit"]["record"]["createdAt"]

        #  Temp storage
        self.temp_storage.append({
            "cid": cid,
            "created_at": created_at,
            "text": preprocessed_text
        })

        # Persist storage periodically
        if time.time() - self.last_saved_time > self.persist_interval:
            await self.persist_storage()

    @staticmethod
    def _preprocess_text(text):
        preprocessed_text = re.sub(r'http\S+', '', text)  # Remove URL
        preprocessed_text = re.sub(r'@\w+', '', preprocessed_text)  # Remove @
        preprocessed_text = preprocessed_text.lower()
        preprocessed_text = re.sub(r'[^\w\s]', '', preprocessed_text)  # Remove punctuation
        # Remove Stop words
        tokens = word_tokenize(preprocessed_text)
        stops = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stops]
        return ' '.join(tokens)

    async def persist_storage(self):
        # Persist data from temp storage
        if not self.temp_storage:
            return

        filename = f"jetstream_{int(time.time())}.json"

        try:
            with open(f"temp_storage/{filename}", 'w') as f:
                for post in self.temp_storage:
                    f.write(json.dumps(post) + "\n")

            print(f"Saved {len(self.temp_storage)} posts to {filename}")

            # Reset temp storage
            self.temp_storage = []
            self.last_saved_time = time.time()

        except IOError as e:
            print(f"Failed to persist temp storage: {e}")

