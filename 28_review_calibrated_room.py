"""Open the latest completed calibrated replay without starting any camera."""
from pathlib import Path
import json
import importlib.util


def main():
    root=Path(__file__).resolve().parent
    candidates=[]
    for path in (root/'report_evidence'/'stage_18_final_room').glob('*/processing_report.json'):
        report=json.loads(path.read_text(encoding='utf-8'))
        if report.get('projection_override_used') and (path.parent/'room_mesh.ply').is_file():
            candidates.append((report.get('created_at_utc',''),path.parent))
    if not candidates:
        print('No completed calibrated room replay yet. Wait for reconstruction to finish.')
        return
    folder=max(candidates,key=lambda p:p[0])[1]
    print(f'Opening calibrated room evidence: {folder}',flush=True)
    print('Drag to orbit; mouse wheel to zoom. C: coverage, L: layout, Q: close.',flush=True)
    print('Experimental geometry: inspect duplicated walls and missing regions.',flush=True)
    spec=importlib.util.spec_from_file_location('calibrated_room_review',root/'18_finalize_room_scan.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.review(folder)


if __name__=='__main__':main()
