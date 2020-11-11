import multiprocessing
import sys


def worker(num):
    """thread worker function"""
    msg = f"Worker: {num}"
    print(msg)
    sys.stdout.flush()
    return msg