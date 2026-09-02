"""Final correctness + performance comparison."""
import json, math, numpy as np

with open('_phase_artifacts/phase_xxc_optimized.json') as f:
    baseline = json.load(f)
with open('_phase_artifacts/phase_xxe_run1.json') as f:
    optimized = json.load(f)

print('=' * 70)
print('CORRECTNESS: XX-C baseline vs XX-E cached (Run 1)')
print('=' * 70)

status_changed = 0
center_deltas = []
radius_deltas = []
for b, o in zip(baseline, optimized):
    if b['pupil_detected'] != o['pupil_detected']:
        status_changed += 1
        print(f'  frame {b["frame_idx"]}: pupil CHANGED ({b["pupil_detected"]} -> {o["pupil_detected"]})')
    if b['limbus_detected'] != o['limbus_detected']:
        status_changed += 1
        print(f'  frame {b["frame_idx"]}: limbus CHANGED ({b["limbus_detected"]} -> {o["limbus_detected"]})')
    if b['pupil_detected'] and o['pupil_detected']:
        b_c = b.get('pupil_center', [0, 0])
        o_c = o.get('pupil_center', [0, 0])
        delta = math.sqrt((b_c[0] - o_c[0])**2 + (b_c[1] - o_c[1])**2)
        center_deltas.append(delta)
        b_r = b.get('pupil_radius', 0)
        o_r = o.get('pupil_radius', 0)
        radius_deltas.append(abs(b_r - o_r))

print(f'\n  Status changes: {status_changed}')
if center_deltas:
    print(f'  Centre delta: max={max(center_deltas):.4f} mean={np.mean(center_deltas):.4f}')
if radius_deltas:
    print(f'  Radius delta: max={max(radius_deltas):.4f} mean={np.mean(radius_deltas):.4f}')

b_times = [r['total_ms'] for r in baseline]
o_times = [r['total_ms'] for r in optimized]
print(f'\n  PERFORMANCE:')
print(f'    Baseline mean:   {np.mean(b_times):.0f} ms')
print(f'    Optimized mean:  {np.mean(o_times):.0f} ms')
print(f'    Improvement:     {np.mean(b_times) - np.mean(o_times):.0f} ms ({(np.mean(b_times) - np.mean(o_times)) / np.mean(b_times) * 100:.1f}%)')
print(f'    Baseline median: {np.median(b_times):.0f} ms')
print(f'    Optimized median:{np.median(o_times):.0f} ms')
print(f'    Baseline worst:  {np.max(b_times):.0f} ms')
print(f'    Optimized worst: {np.max(o_times):.0f} ms')

if status_changed == 0:
    print(f'\n  CORRECTNESS: PASS')
else:
    print(f'\n  CORRECTNESS: FAIL ({status_changed} changes)')
