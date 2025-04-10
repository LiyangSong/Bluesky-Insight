import asyncio
import os

from apps.worker.jetstream_processor import JetstreamProcessor
from apps.worker.trend_analyzer import TrendAnalyzer


class StreamAnalysisQueue():
    def __init__(self):
        self.queue = asyncio.Queue()

    async def enqueue(self, item):
        await self.queue.put(item)

    async def dequeue(self):
        return await self.queue.get()


async def main():
    os.makedirs("output", exist_ok=True)
    os.makedirs("temp_storage", exist_ok=True)

    queue = StreamAnalysisQueue()
    jetstream_processor = JetstreamProcessor(queue)
    trend_analyzer = TrendAnalyzer(queue)

    await asyncio.gather(
        jetstream_processor.process_jetstream(),
        trend_analyzer.monitor_queue_and_analysis()
    )

if __name__ == '__main__':
    asyncio.run(main())




