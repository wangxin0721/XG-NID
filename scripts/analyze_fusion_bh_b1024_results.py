"""Reuse validation diagnostics while declaring the batch-size comparison boundary."""
from analyze_fusion_bh_results import analyze as analyze_base

def analyze(repo, output, config, binding):
    result = analyze_base(repo, output, config, binding)
    result['schema'] = 'xgnid.fusion-bh-b1024-analysis.v1'
    result['comparison_boundary'] = {
        'candidate_training_batch': 1024, 'historical_controls_training_batch': 512,
        'candidate_updates_per_epoch': 106, 'historical_controls_updates_per_epoch': 211,
        'evaluation_batch': 16, 'same_learning_rate': .001,
        'interpretation': 'Differences against historical B/C/E2/E0 combine batch and model changes; they do not isolate H-statistic effects.'}
    result['decision']['batch512_threshold_action_for_reference_only'] = result['decision']['action']
    result['decision']['action'] = 'review_exploratory_batch1024_result_with_batch512_BH'
    result['decision']['automatic_additional_training'] = False
    lines = ['# B-H batch1024 seed42', '',
             f"Validation Macro-F1: {100*result['metrics']['metrics']['fusion']['macro_f1']:.6f}%", '',
             'Training batch1024, evaluation batch16, lr0.001, 106 optimizer steps/epoch.',
             'Historical B/C/E2/E0 controls used batch512 and 211 steps/epoch.',
             'Compare this run with B-H batch512 to study batch effects; historical-control score differences cannot isolate the gate input change.',
             'Full paired diagnostics and class metrics: analysis.json. No further training is automatically scheduled.']
    (output/'decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
