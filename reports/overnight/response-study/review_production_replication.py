"""Record qualitative review of every production-sampler stress response."""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from checkin.train import atomic_json, file_sha256

out = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (out / 'responses-production-replication.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(rows) == 54
review = {
    'reviewed_at': datetime.now(timezone.utc).isoformat(),
    'reviewer': 'AI qualitative review by implementation agent; no numeric quality score or clinical-validation claim',
    'scope': 'All 54 replies: six previously revealed authored check cases, three visual-state conditions, seeds 42/43/44, exact production temperature0.5/top_p0.8/top_k20/min_p0/max_tokens96.',
    'not_fresh_holdout': True,
    'no_further_prompt_tuning': True,
    'cases': [
        {'id': 'authored:check:ambiguous_win', 'strengths': 'All nine avoid congratulations and ask about the meaning/feeling of winning.', 'flaws': 'Generic something-to-notice/interesting language; body/mind questions can be formulaic. Victory wording leans positive but no explicit claim the event is welcome.'},
        {'id': 'authored:check:good_day', 'strengths': 'All nine preserve the explicitly good day despite sadness evidence.', 'flaws': 'Several introduce discomfort or being bothered as implied consequences of squinting, or offer a forced discomfort/vision question. Some repeat what was already stated rather than ask about the day.'},
        {'id': 'authored:check:birthday_grief', 'strengths': 'All nine recognize loss/birthday without fabricated personal memories or claiming to know the grandmother.', 'flaws': 'Many assume the birthday must feel heavy/hard; seed42 joy invents a quiet way of remembering. Seed42 missing vision is more grounded: it reflects the birthday and asks how the person feels.'},
        {'id': 'authored:check:text_me', 'strengths': 'All nine avoid fake numbers or firm promises to initiate text messages. Seed44 sadness explicitly says it cannot send outside the conversation.', 'flaws': 'Seed42 sadness asks about another way to connect later, implying an unavailable alternative channel. Most replies do not explicitly distinguish the user returning to chat from the bot initiating contact.'},
        {'id': 'authored:check:sister_mixed', 'strengths': 'Positive reaction to proximity is grounded in the user saying happy; every reply asks specifically about sharing the apartment.', 'flaws': 'Eight replies omit explicit anxiety; seed42 missing vision only indirectly reflects it as tricky. Reasking feelings the user already stated is less attentive than reflecting both.'},
        {'id': 'authored:check:racing_heart', 'strengths': 'Most reflect the racing heart/stairs without diagnosis or treatment; seed42/43 sadness asks about the rest of the day as requested.', 'flaws': 'Seed44 joy asks whether it was just a normal reaction. This is a medical-normality suggestion phrased as a question, not a definite diagnosis, and violates the intended no-medical-interpretation contract. Seed44 sadness asks whether it was really just physical, introducing an unnecessary alternative interpretation. Other replies presume intensity or repeat the bodily detail.'},
    ],
    'preserved_failure_examples': [
        {'seed': 42, 'case_id': 'authored:check:text_me', 'condition': 'vision_sadness', 'concern': 'Unsupported alternative contact channel'},
        {'seed': 43, 'case_id': 'authored:check:birthday_grief', 'condition': 'vision_joy', 'concern': 'Must-feel-heavy assertion'},
        {'seed': 44, 'case_id': 'authored:check:racing_heart', 'condition': 'vision_joy', 'concern': 'Normal-reaction option introduced without support'},
        {'seed': 44, 'case_id': 'authored:check:racing_heart', 'condition': 'vision_sadness', 'concern': 'Really-just-physical question adds unrequested interpretation'},
    ],
    'interpretation': 'Production sampling preserves some grounded behavior and also exposes unresolved failures that temp0 checks missed. With only the selected prompt run at production settings, this is a stress replication, not evidence that production quality improved over its baseline. No additional prompt was selected or tuned from these responses.',
    'timings': {'completion_ms_median': float(np.median([row['completion_ms'] for row in rows])),
                'completion_ms_p95': float(np.percentile([row['completion_ms'] for row in rows], 95)),
                'ttft_ms_median': float(np.median([row['ttft_ms'] for row in rows])),
                'completion_ms_sum': float(sum(row['completion_ms'] for row in rows)),
                'scope': 'Warm sequential loopback generation only, with repeated prompt prefixes; excludes capture, encoders, classifier and UI rendering.'},
    'response_sha256': file_sha256(out / 'responses-production-replication.jsonl'),
    'protocol_sha256': file_sha256(out / 'production-replication-protocol.json'),
}
lookup = {(row['seed'], row['case_id'], row['condition']): row for row in rows}
for example in review['preserved_failure_examples']:
    example['response'] = lookup[(example['seed'], example['case_id'], example['condition'])]['response']
atomic_json(out / 'review-production-replication.json', review)
report = json.loads((out / 'study-report.json').read_text(encoding='utf-8'))
report['production_sampler_stress_replication'] = {
    'additional_generations': 54, 'overall_generations_across_both_protocols': 204,
    'request_errors': sum(row['error'] is not None for row in rows),
    'length_limit_finishes': sum(row['finish_reason'] == 'length' for row in rows),
    'responses_with_narrow_pattern_flags': sum(bool(row['pattern_flags']) for row in rows),
    'review': review,
    'files': {name: file_sha256(out / name) for name in ('production-replication-protocol.json', 'responses-production-replication.jsonl', 'summary-production-replication.json', 'review-production-replication.json')},
}
atomic_json(out / 'study-report.json', report)
print(json.dumps(report['production_sampler_stress_replication'], indent=2))
