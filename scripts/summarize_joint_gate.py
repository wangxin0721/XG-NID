"""Summarize a completed joint experiment without importing tensor libraries."""
import argparse,json
from pathlib import Path

def summarize(root):
    r=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    rows=[]
    for name,m in [('N0',r['N0']),('frozen_V2',r['frozen_V2'])]+list(r['joint'].items()):
        rows.append({'model':name,'macro_f1':m['best_validation']['macro_f1'],
            'per_class':m['best_validation']['per_class'],'best_epoch':m['best_epoch'],
            'updates':m.get('updates',m.get('optimizer_updates')),
            'wall_seconds':m.get('joint_wall_seconds')})
    lines=['| model | validation Macro-F1 | best epoch | updates | joint seconds |','|---|---:|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['model']} | {100*row['macro_f1']:.6f}% | {row['best_epoch']} | {row['updates']} | {row['wall_seconds']} |")
    lines+=['','Stage A accounting: '+json.dumps(r['stage_A']),
        'Missing historical durations are unavailable, not zero. J2 includes extra warmup. No isolated component attribution.']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    return '\n'.join(lines)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);print(summarize(p.parse_args().root))
