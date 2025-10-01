#!/usr/bin/env python3
"""
Run both feeder and ingest_predict in parallel threads.
This allows both to run in a single terminal window.
"""

import threading
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def run_feeder():
    """Run the feeder in a separate thread."""
    print("[FEEDER] Starting data feeder...")
    try:
        from src import feeder
        # Override sys.argv to pass default args
        sys.argv = ["feeder.py"]
        feeder.main()
    except KeyboardInterrupt:
        print("\n[FEEDER] Stopped.")
    except Exception as e:
        print(f"[FEEDER] Error: {e}")

def run_ingest():
    """Run the ingest_predict in a separate thread."""
    # Give feeder a head start to create some files
    print("[INGEST] Waiting 3 seconds for feeder to start...")
    time.sleep(3)
    print("[INGEST] Starting prediction engine...")
    try:
        from src import ingest_predict
        # Override sys.argv to pass default args
        sys.argv = ["ingest_predict.py"]
        ingest_predict.main()
    except KeyboardInterrupt:
        print("\n[INGEST] Stopped.")
    except Exception as e:
        print(f"[INGEST] Error: {e}")

def main():
    print("=" * 60)
    print("  GigShield Risk Monitor - Pipeline Runner")
    print("=" * 60)
    print("\nStarting both services in parallel...")
    print("- Feeder: Generates synthetic job transactions")
    print("- Ingest: Processes transactions and makes predictions")
    print("\nPress Ctrl+C to stop both services\n")

    # Create threads for both services
    feeder_thread = threading.Thread(target=run_feeder, daemon=True)
    ingest_thread = threading.Thread(target=run_ingest, daemon=True)

    # Start both threads
    feeder_thread.start()
    ingest_thread.start()

    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("  Shutting down pipeline...")
        print("=" * 60)
        print("\nBoth services stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()
