"""Extract fitted wall candidates and floor from a previously saved scan PLY."""
from datetime import datetime
from pathlib import Path
import argparse
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import open3d as o3d
from processing.room_layout import extract_layout

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error('Saved model does not exist')
    cloud = o3d.io.read_point_cloud(str(args.model))
    if not cloud.has_points():
        parser.error('PLY contains no point positions')
    out = Path(__file__).resolve().parent/'report_evidence'/'stage_20_room_layout'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    extract_layout(cloud,out,source_model=args.model)
    print(f'Layout evidence: {out}',flush=True)
