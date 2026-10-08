"""Recompute and validate poses for a saved Stage 17 scan, without a camera."""
import argparse
from pathlib import Path
from processing.room_trajectory import optimise_trajectory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",type=Path)
    parser.add_argument("--stride",type=int,default=3)
    parser.add_argument("--max-frames",type=int,default=0)
    parser.add_argument("--intrinsic",type=Path,help="Experimental replay projection")
    args = parser.parse_args()
    if args.stride < 1 or args.max_frames < 0:
        parser.error("stride must be positive; max-frames must be nonnegative")
    optimise_trajectory(args.dataset,args.stride,args.max_frames,intrinsic_path=args.intrinsic)
