"""CLI.

  python -m hoopsvision run    --video in.mp4 --out out/ [--config config.yaml]
  python -m hoopsvision render --video in.mp4 --out out/   # re-render from cache only
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hoopsvision", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("run", "full pipeline (cached stages are reused)"),
                        ("render", "re-render outputs from cached stages; never runs the models")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--video", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--config", default="config.yaml")
        s.add_argument("--max-frames", type=int, default=None, help="override video.max_frames")
        if name == "run":
            s.add_argument("--force", nargs="*", default=[],
                           help="stages to recompute: detect scenecut track teams calibrate metrics | all")
        s.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("ultralytics", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    overrides = {"video": {"max_frames": a.max_frames}} if a.max_frames else None
    cfg = load_config(a.config, overrides)

    from .pipeline import Pipeline

    pipe = Pipeline(a.video, a.out, cfg, force=tuple(getattr(a, "force", []) or ()))
    if a.cmd == "render":
        pipe.render_only()
    else:
        pipe.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
