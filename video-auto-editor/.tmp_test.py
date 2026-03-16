import sys, traceback
sys.path.insert(0, 'src')
try:
    from auto_video_editor.renderer import _subtitle_overlays
    from auto_video_editor.models import PlannedSegment
    segs = [PlannedSegment(start=0.0, end=2.0, text='Test owl fact here', duration=2.0)]
    result = _subtitle_overlays(segs, width=1080, height=1920, final_duration=2.0, caption_style='beast')
    print('OK', len(result))
except Exception:
    traceback.print_exc()
