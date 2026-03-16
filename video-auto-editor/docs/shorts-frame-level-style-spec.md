# Shorts Frame-Level Style Spec

Date: 2026-03-15
Source analyzed: local MP4 in workspace root (59.62s, 9:16, 30fps)
Target: premium retention-driven short suitable for top creator teams.

## 1) Baseline Findings From This Exact Video

- Runtime: 59.62s
- Resolution/FPS: 720x1280 @ 30fps
- Voiceover blocks: 13 segments (about 4-5s each)
- Cut density (scene threshold 0.25): 36 cuts total
- Average gap between cuts: 1.40s
- Pacing profile: calmer first half, aggressive montage spike near the end

### Segment-Level Rhythm Snapshot

- 00.00-04.00: 0 cuts (static opener)
- 04.00-08.64: 1 cut
- 08.64-12.80: 2 cuts
- 12.80-17.84: 1 cut
- 17.84-22.40: 2 cuts
- 22.40-27.84: 3 cuts
- 27.84-31.04: 0 cuts
- 31.04-36.00: 2 cuts
- 36.00-40.80: 3 cuts
- 40.80-44.56: 2 cuts
- 44.56-49.76: 11 cuts (major acceleration)
- 49.76-55.68: 8 cuts (major acceleration)
- 55.68-59.68: 1 cut

Interpretation: this short is strong in late-stage velocity, but under-optimized in first 3-5 seconds where Shorts retention is won.

## 2) MrBeast-Level Edit Grammar

## Hook Window (0.0s-3.0s)

- Show payoff first. No setup sentence before visual proof.
- First visual event by 0.3-0.8s.
- At least 3 distinct visual events in first 3s:
  - Reframe/zoom
  - B-roll evidence shot
  - Caption emphasis pulse
- Never let the first 2 seconds stay on a single static composition.

## Mid-Body Rhythm (3.0s-45.0s)

- Target average visual event every 0.9-1.4s.
- Alternate cadence to avoid mechanical feel:
  - Burst section: 0.5-0.9s gaps
  - Relief section: 1.2-1.8s gaps
- Each voiceover clause must trigger a proof visual within 3-8 frames.

## Climax + Payoff (45.0s-end)

- Keep acceleration, but keep logic clear.
- Every rapid cut must still answer one narrative question:
  - What changed?
  - Why should I care?
  - What is the next surprising fact?
- End with loop bait in final 0.8-1.2s (question, tease, or callback).

## 3) Transition Decision Tree

- Default: hard cuts on speech beat.
- Use transition effects only when they preserve comprehension.

Allowed transition timings:

- Hard cut: 0f
- Motion blur/whip: 3-6f
- Push/zoom: 5-9f

Rules:

- If transition draws attention to itself, remove it.
- Direction continuity required for whip/pan transitions.
- Max 2-4 stylized transitions in a 60s short.

## 4) Caption Highlighting System

Goal: captions should increase understanding and urgency, not decorate the frame.

Typography and placement:

- Max 2 lines
- 26-34 chars per line target
- Bottom-third safe zone with dynamic vertical offset when subject blocks text

Highlight behavior:

- Active word: high contrast
- Inactive words: 35-50% dimmer
- Emphasis words only (numbers, verbs, stakes, novelty terms)
- Active word pulse: scale 1.02-1.06 for 80-120ms

Timing:

- Word highlight sync tolerance: +/-40ms target
- Avoid static full-line dwell >1.4s unless intentionally dramatic

## 5) Audio Mix Targets

- VO always primary and intelligible
- Music ducking tied to VO presence
- Micro-SFX on key cut points (under VO)
- No perceived loudness jumps between sections

Practical loudness targets (short-form social):

- Integrated loudness around -14 to -12 LUFS
- True peak <= -1.0 dBTP

## 6) Quality Gates Before Delivery

A video is not client-ready unless all pass:

- Hook gate:
  - 3+ visual events in first 3s
  - First caption appears by <=0.7s
- Motion gate:
  - No static shot >1.8s without a text/motion/composition change
- Caption gate:
  - No overlap with key subject face/object
  - Emphasis words visually distinct
- Coherence gate:
  - Every cut has a narrative reason
- Export gate:
  - Deliver 1080x1920 master, clean audio, no dropped frames

## 7) Concrete Upgrade Plan For This Owl Video Style

- Keep the existing late acceleration pattern (44.5s onward).
- Rebuild first 5 seconds with outcome-first visuals.
- Add stricter caption emphasis for ordinal list beats (First, Second, Third, Fourth, Fifth).
- Reduce any non-informational transition effects.
- Add loop bait ending to increase rewatches.

## 8) Shot-Level Blueprint Template (Use For New Shorts)

- 0.00-0.80: Outcome visual + hard claim caption
- 0.80-2.00: Proof cut #1 + highlighted keyword
- 2.00-3.00: Proof cut #2 + escalation text
- 3.00-15.00: Fact loop with one visual proof per clause
- 15.00-45.00: Alternating burst/relief cadence
- 45.00-58.50: High-speed montage with clear semantic anchors
- 58.50-60.00: Loop bait outro

Use this as the default production grammar until performance data suggests otherwise.
