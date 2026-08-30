"""Compatibility CLI for the canonical V-Shield data collector."""


def main() -> None:
    """Load the camera stack lazily, then start collection."""
    print("Loading OpenCV/MediaPipe; this may take several seconds...", flush=True)
    from vshield.data.collector import main as run_collector

    run_collector()


if __name__ == "__main__":
    main()
