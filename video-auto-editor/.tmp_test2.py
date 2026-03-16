import sys; sys.path.insert(0,'src')
from auto_video_editor.ffmpeg_setup import ensure_pillow_compatibility
ensure_pillow_compatibility()
from auto_video_editor.renderer import _subtitle_overlays, _apply_micro_zooms
from auto_video_editor.models import PlannedSegment
from moviepy.editor import ColorClip

segs=[
    PlannedSegment(start=0.0,end=2.0,text='NEVER underestimate the FIRST step',duration=2.0,emphasis=True),
    PlannedSegment(start=2.0,end=4.0,text='Five amazing things you never knew',duration=2.0),
]
r=_subtitle_overlays(segs,width=1080,height=1920,final_duration=4.0,caption_style='beast')
print('Caption overlays OK:', len(r))

clip = ColorClip(size=(1080,1920), color=[0,0,0], duration=3.0).set_fps(30)
zoomed = _apply_micro_zooms(clip)
frame = zoomed.get_frame(0.5)
print('Micro-zoom frame shape:', frame.shape)
print('ALL PASS')
