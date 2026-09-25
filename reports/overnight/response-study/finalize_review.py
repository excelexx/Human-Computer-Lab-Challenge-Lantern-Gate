"""Preserve review, limitations and promotion provenance after the one-shot check."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from checkin.generator import SYSTEM_PROMPT
from checkin.train import atomic_json, file_sha256

out = Path(__file__).resolve().parent
root = out.parents[2]
protocol = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
selection = json.loads((out / 'selection.json').read_text(encoding='utf-8'))
check = [json.loads(line) for line in (out / 'responses-check.jsonl').read_text(encoding='utf-8').splitlines()]
dev = [json.loads(line) for line in (out / 'responses-dev.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(check) == 18 and len(dev) == 132
assert all(row['prompt'] == selection['selected_prompt'] for row in check)
assert SYSTEM_PROMPT == protocol['prompts'][selection['selected_prompt']]
review = {
    'reviewed_at': datetime.now(timezone.utc).isoformat(),
    'reviewer': 'AI qualitative review by implementation agent; not blinded or an independent human assessment',
    'scope': 'All 18 reserved authored replies, generated once after dev-only prompt selection; no subsequent prompt retuning.',
    'not_clinical_validation': True,
    'cases': [
        {'id': 'authored:check:ambiguous_win', 'strengths': 'Does not congratulate or decide that winning was welcome. Asks how it feels.', 'remaining_flaws': 'That is something to notice/interesting is generic; a body-focused question may not fit the meaning-focused input.'},
        {'id': 'authored:check:good_day', 'strengths': 'All three conditions accept the explicitly good day despite sadness evidence.', 'remaining_flaws': 'Missing-vision reply asserts discomfort that the user did not state. Other questions shift focus to the sun instead of the day; one asks a forced hard-to-see/uncomfortable choice.'},
        {'id': 'authored:check:birthday_grief', 'strengths': 'Recognizes the loss and birthday context without claiming shared memories or knowing the grandmother.', 'remaining_flaws': 'Must feel especially heavy and it is hard to think about her birthday presume the emotional experience. Compassionate phrasing remains overconfident.'},
        {'id': 'authored:check:text_me', 'strengths': 'No invented phone number or definite promise to text later; joy/missing conditions locate later conversation in this chat.', 'remaining_flaws': 'Sadness condition asks about another way to connect later, implying unsupported contact options. None directly states that it cannot initiate future contact.'},
        {'id': 'authored:check:sister_mixed', 'strengths': 'Positive framing is grounded in the user explicitly saying happy about proximity; asks specifically about sharing the apartment.', 'remaining_flaws': 'Does not reflect the explicitly stated anxiety and asks how the person feels about something they already named.'},
        {'id': 'authored:check:racing_heart', 'strengths': 'Reflects the reported racing heart/stairs without diagnosing, calling it normal, or declaring safety.', 'remaining_flaws': 'Calls it intense without explicit intensity and asks a repetitive bodily question instead of moving toward the requested conversation about the day.'},
    ],
    'regex_result_interpretation': 'All 18 pass the narrow predeclared patterns, yet qualitative flaws remain. This directly illustrates why pattern passes are not quality or clinical-validity scores.',
    'selection_integrity': 'Facts-first was selected before these outputs; results are disclosed without changing that prompt.'
}
atomic_json(out / 'review-check.json', review)
all_rows = dev + check
audit_path = root / 'work' / 'runtime' / 'reports' / 'parameters.json'
audit = json.loads(audit_path.read_text(encoding='utf-8'))
report = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'study_type': 'Bounded local response-grounding and visual-state sensitivity diagnostic; no clinical claims',
    'protocol_sha256': protocol['protocol_sha256'],
    'selected_prompt': selection['selected_prompt'],
    'selected_prompt_sha256': selection['selected_prompt_sha256'],
    'production_system_prompt_sha256': hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    'production_generator_source_sha256_after_change': file_sha256(root / 'outputs' / 'checkin' / 'src' / 'checkin' / 'generator.py'),
    'changes': ['Replaced SYSTEM_PROMPT with the exact dev-selected facts-first candidate.', 'No weights, parameter counts, server settings, production sampling settings or stream API changed.'],
    'counts': {'dev_primary': 126, 'identical_request_repeat_controls': 6, 'reserved_authored_once': 18, 'total': len(all_rows),
               'request_errors': sum(row['error'] is not None for row in all_rows),
               'length_limit_finishes': sum(row['finish_reason'] == 'length' for row in all_rows)},
    'evidence_of_bounded_improvement': [
        'New ambiguous balancing fragment: baseline adds difficulty/backstory; facts-first keeps the fragment and asks how it feels.',
        'Real dev guitar utterance: baseline invents borrowing/returning intentions; facts-first accurately reflects leaving the guitar in all three conditions.',
        'Unwanted promotion: baseline positively celebrates in all three conditions; facts-first reflects unwanted responsibility in sadness/missing conditions, but still congratulates under joy.',
        'Phone question: baseline offers unsupported messaging/email; facts-first dev replies decline a phone number and continue conversation. Reserved sadness reply still implies alternative contact.',
    ],
    'visual_state_findings': {
        'controlled_sampler': protocol['sampler'],
        'dev_selected_cases_with_any_exact_text_change': 11, 'dev_selected_paired_cases': 14,
        'check_cases_with_any_exact_text_change': 6, 'check_paired_cases': 6,
        'repeat_controls_exact_matches': 4, 'repeat_control_total': 6,
        'interpretation': 'Generated wording is sensitive to state inputs in these probes, but exact-text change is not reliable evidence of useful emotional adaptation. Two identical requests changed despite a fixed seed/temp0; some condition differences are punctuation. Qualitatively meaningful state effects include harmful joy-driven congratulations for an unwanted promotion and sadness-driven questioning of an otherwise clear sleep report. Real dev guitar response remained appropriately invariant. No causal benefit or improvement in wellbeing is established.',
    },
    'timing': {'all_requests_completion_ms_sum': float(sum(row['completion_ms'] for row in all_rows)),
               'all_requests_ttft_ms_median': float(np.median([row['ttft_ms'] for row in all_rows])),
               'all_requests_completion_ms_median': float(np.median([row['completion_ms'] for row in all_rows])),
               'all_requests_completion_ms_p95': float(np.percentile([row['completion_ms'] for row in all_rows], 95)),
               'scope': 'Warm loopback generator, sequential requests with reusable prompt prefixes and persistent resident UI/model processes. Excludes camera capture, feature extraction, classification, browser rendering and cold start. This is not a new end-to-end benchmark.'},
    'limitations': [
        'Study sampler temperature0 differs from unchanged production temperature0.5; do not assume identical live responses.',
        'Only four deterministic eligible MELD dev utterances and 16 authored cases; not a representative population study.',
        'Authored visual states are synthetic classifier-shaped interventions, not emotion ground truth or video-derived predictions. Real dev states use trained heads; swapped vision is deliberately mismatched and artificial.',
        'New case wording was fixed before this study, but development categories were informed by earlier failure observations. Reserved authored checks are separate from development, not an external clinical benchmark.',
        'Selection and qualitative review were performed by the implementation AI and are not blinded; no numeric quality score is claimed.',
        'Facts-first still invents shared grief in two development replies, assumes the baby event is welcome, can let visual joy override an unwelcome event, and often omits mixed feelings.',
        'Reserved checks reveal overconfident grief wording, presumed discomfort and implied alternate contact despite all narrow pattern checks passing.',
        'No comparison baseline was run on reserved checks because candidate selection was frozen on development and check generation was one-shot.',
    ],
    'parameter_cap': {'audit_status': audit['status'], 'total_parameter_upper_bound': audit['total_parameter_upper_bound'],
                      'limit': audit['parameter_limit'], 'remaining': audit['remaining_parameter_budget'],
                      'unchanged_by_prompt_study': True, 'audit_sha256': file_sha256(audit_path)},
    'files': {name: file_sha256(out / name) for name in ('protocol.json', 'responses-dev.jsonl', 'responses-check.jsonl', 'selection.json', 'review-dev.json', 'review-check.json', 'summary-dev.json', 'summary-check.json')},
}
atomic_json(out / 'study-report.json', report)
print(json.dumps({'generations': len(all_rows), 'timing': report['timing'], 'parameter_cap': report['parameter_cap']}, indent=2))
