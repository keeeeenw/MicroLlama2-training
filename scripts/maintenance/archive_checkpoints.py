"""Continuously archive completed local checkpoints to permanent storage."""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from microllama.training.checkpoint_archive import run_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Local checkpoints root, such as NVMe/checkpoints")
    parser.add_argument("--destination", type=Path, required=True,
                        help="Permanent archive root, such as TrainData/checkpoints")
    parser.add_argument("--train-pid", type=int,
                        help="Exit once this trainer stops and every published checkpoint is archived")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    run_worker(args.source, args.destination, train_pid=args.train_pid,
               poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
