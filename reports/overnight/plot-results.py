import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot the saved, measured overnight evidence; no inference.')
parser.add_argument('--evidence-root', type=Path, default=Path(__file__).resolve().parent)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
night = args.evidence_root
out = args.output or night / 'results.png'
comparison = json.loads((night / 'robustness-hybrid-comparison.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.spines.left': False, 'axes.spines.bottom': False})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), gridspec_kw={'width_ratios': [1, 1.2]})
fig.patch.set_facecolor('#faf9f5')
for ax in axes:
    ax.set_facecolor('#faf9f5')
    ax.set_xlim(0, 50)
    ax.set_xlabel('Macro F1 (%)')
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#e3e5df', linewidth=.7)
    ax.tick_params(axis='y', length=0, pad=8)

labels = ['Original operational', 'Current operational', 'Current text alone']
hybrid = json.loads((night / 'model-study/hybrid-reused-test-results.json').read_text())['metrics']['all']
values = [hybrid[name]['macro_f1'] * 100 for name in ['baseline_operational', 'hybrid', 'new_text']]
bars = axes[0].barh(labels, values, color=['#aeb7b1', '#365d4d', '#b39a76'], height=.52)
axes[0].invert_yaxis()
axes[0].set_title('Same 2,610 reused-test utterances', loc='left', fontweight='bold', pad=20)
for bar, value in zip(bars, values):
    axes[0].text(value+.6, bar.get_y()+bar.get_height()/2, f'{value:.2f}', va='center', fontsize=9)

names = ['Original video', 'Silent stream copy', 'H.264 re-encode', 'Half resolution', 'Half brightness', 'Gaussian blur', 'Horizontal flip', 'Black video']
rows = list(comparison['summaries'].values())
y = np.arange(8)
axes[1].barh(y-.17, [r['baseline']['macro_f1']*100 for r in rows], height=.30, color='#aeb7b1', label='Original operational')
axes[1].barh(y+.17, [r['candidate']['macro_f1']*100 for r in rows], height=.30, color='#365d4d', label='Current operational')
axes[1].set_yticks(y, names)
axes[1].invert_yaxis()
axes[1].set_title('Same 95 stratified dev clips × 8 conditions', loc='left', fontweight='bold', pad=20)
axes[1].legend(loc='lower right', frameon=False, fontsize=8)
fig.suptitle('Measured changes, with their trade-offs', x=.07, ha='left', fontsize=17, fontweight='bold', color='#253c32')
fig.text(.07, .095, 'Left: improved macro F1 comes from text fallback; current text alone remains stronger. Accuracy and weighted F1 decrease slightly.', fontsize=9, color='#505951')
fig.text(.07, .058, 'Right: identical saved features; visual predictions unchanged. Altered-video conditions are stress diagnostics, not independent accuracy.', fontsize=9, color='#505951')
fig.text(.07, .021, 'Official test was already observed. No clinical or webcam validation. Both panels retain failures and null results.', fontsize=9, color='#505951')
fig.subplots_adjust(left=.16, right=.985, top=.81, bottom=.24, wspace=.78)
fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
print(out)
