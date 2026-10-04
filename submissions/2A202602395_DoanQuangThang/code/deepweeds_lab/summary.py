"""Readable comparison costs with explicit provenance and missing measurements."""
import pandas as pd


def build_summary(backbones, training, inference, finals, selection):
    rows = []
    selected_id = selection['training_config']['exp_id']
    single = next((r for r in inference if r['method'] == 'single'), None)
    baseline = next((r for r in finals if r['exp_id'] == 'T00' and
                     r['seed'] == selection['seeds'][0]), None)
    for row in backbones + training:
        latency = row.get('latency_ms')
        source = row['exp_id'] if latency is not None else 'Chưa đo riêng'
        if row['exp_id'] == selected_id and single is not None:
            latency, source = single['p50'], 'I00: đúng checkpoint seed đầu, single-view'
        elif row['exp_id'] == 'T00' and baseline is not None:
            latency, source = baseline['latency']['p50'], 'T00: single-view, seed đầu vòng cuối'
        rows.append(dict(exp_id=row['exp_id'], stage='training', backbone=row['backbone'],
                         macro_f1_val=row['macro_f1_val'], top1_val=row['top1_val'],
                         latency_ms=latency if latency is not None else 'Chưa đo riêng',
                         latency_percentile='p50' if latency is not None else 'Chưa đo',
                         latency_source=source, params_m=row.get('params_m'),
                         gmac=row.get('gmac'),
                         train_seconds_per_epoch=row.get('train_seconds_per_epoch'),
                         is_baseline=row['exp_id'] == 'T00'))
    for row in inference:
        rows.append(dict(exp_id=row['exp_id'], stage='inference', backbone=selection['backbone'],
                         macro_f1_val=row['macro_f1'], top1_val=row['top1'], latency_ms=row['p95'],
                         latency_percentile='p95', latency_source=row['exp_id'] + ': batch 1',
                         params_m=None, gmac=None, train_seconds_per_epoch=None,
                         is_baseline=False))
    frame = pd.DataFrame(rows).sort_values('macro_f1_val', ascending=False)
    top = frame.head(10)
    # Preserve the top ten and also show the baseline even if it ranks below them.
    missing_baseline = frame[frame.is_baseline & ~frame.exp_id.isin(top.exp_id)]
    return pd.concat([top, missing_baseline], ignore_index=True)
