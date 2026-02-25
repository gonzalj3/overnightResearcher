"""Entry point for the iMessage chat daemon.

Tails the NDJSON output from a separate imsg watch launchd agent
(which has Full Disk Access) and responds via Ollama 8B.

Usage:
    python -m research.chat_daemon [--watch-file ...] [--db-path ...] [--reports-dir ...]
"""

import argparse
import logging
import signal
import threading

from research.chat_handler import tail_watch_file
from research.orchestrate import setup_logging

WATCH_FILE = "/Users/jmg/GitHub/overnightResearcher/research/logs/imsg-watch-stdout.log"


def main():
    parser = argparse.ArgumentParser(description="iMessage chat daemon for research agent")
    parser.add_argument("--watch-file", default=WATCH_FILE,
                        help="Path to imsg watch NDJSON output file")
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

    logger.info("Starting chat daemon (watch-file=%s, db=%s, reports=%s)",
                args.watch_file, args.db_path, args.reports_dir)

    tail_watch_file(
        watch_file=args.watch_file,
        db_path=args.db_path,
        reports_dir=args.reports_dir,
        stop_event=stop_event,
    )

    logger.info("Chat daemon stopped")


if __name__ == "__main__":
    main()
