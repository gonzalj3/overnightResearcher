"""Entry point for the iMessage chat daemon.

Usage:
    python -m research.chat_daemon [--chat-id 4] [--db-path ...] [--reports-dir ...]
"""

import argparse
import logging
import os
import signal
import sys
import threading

from research.chat_handler import run_watcher
from research.orchestrate import setup_logging


def main():
    parser = argparse.ArgumentParser(description="iMessage chat daemon for research agent")
    parser.add_argument("--chat-id", type=int, default=4, help="iMessage chat ID to watch")
    parser.add_argument("--db-path", default="research/research.db", help="Path to SQLite DB")
    parser.add_argument("--reports-dir", default="~/reports", help="Path to reports directory")
    parser.add_argument("--log-dir", default=None, help="Path to log directory")
    args = parser.parse_args()

    setup_logging(args.log_dir)
    logger = logging.getLogger("research.chat_daemon")

    stop_event = threading.Event()

    def handle_signal(signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    logger.info("Starting chat daemon (chat-id=%d, db=%s, reports=%s)",
                args.chat_id, args.db_path, args.reports_dir)

    run_watcher(
        chat_id=args.chat_id,
        db_path=args.db_path,
        reports_dir=args.reports_dir,
        stop_event=stop_event,
    )

    logger.info("Chat daemon stopped")


if __name__ == "__main__":
    main()
