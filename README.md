# Bluesky Insight

This project is a real-time trend and sentiment analysis system for Bluesky posts, designed to extract trending topics from the Jetstream of Bluesky and analyze public sentiment toward them streamingly.

The system continuously collects live posts from Bluesky's Jetstream, detects trending keywords and hashtags using TF-IDF algorithm, and applies state-of-the-art deep learning models for sentiment classification.

The pipeline is built for production-quality data engineering and machine learning workflows, following real-world practices in streaming data processing and modular architecture.

We are still working on:
- Cloud deployment with distributed queue systems (Redis / AWS SQS).
- Web dashboard using FastAPI + React for real-time trend and sentiment visualization.
- Hashtag segmentation and topic clustering for higher-level trend abstraction.
- Trend history storage and time-series analysis for growth tracking.

