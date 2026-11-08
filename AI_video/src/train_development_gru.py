"""Explicit train/valid-only pipeline. Default invocation only checks readiness.

The frozen local image encoder is reused; no API, downloads, or test predictions.
Features are cached once. New heads are separate from the deployed candidate.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import baseline_common  # Project-local packages and bounded CPU environment.
import numpy as np
from development_video_data import load_split, sample_development_frames
from video_pipeline import file_sha256, save_npz

CONFIG = ROOT / 'configs/train_development_gru_v1.json'
REPORTS = ROOT / 'reports/training/development_gru_v1'
CACHE = ROOT / 'data/cache/features/development_gru_v1'


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def configuration():
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    if not 1 <= cfg['frames_per_video'] <= 64 or cfg['image_size'] != 256:
        raise ValueError('Encoder requires 256px; frame count must be 1..64')
    if not 1 <= cfg['epochs'] <= 1000 or not 1 <= cfg['batch_size'] <= 32:
        raise ValueError('Invalid epochs/batch size')
    if not 1 <= cfg['patience'] <= cfg['epochs'] or cfg['threshold'] != 0.5:
        raise ValueError('Invalid stopping policy or changed threshold')
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Independent seeds must be unique')
    return cfg


def protected_hashes():
    snapshot = json.loads((ROOT / 'data/metadata/training_collection_v1/preserved_test.json').read_text(encoding='utf-8'))
    # Old train/valid hashes in that snapshot intentionally predate activation.
    names = ['data/sample/test.csv', 'test_latest_model.bat',
             'checkpoints/adoption_augmented_v1/residual.keras']
    actual = {name: file_sha256(ROOT / name) for name in names}
    for name in names:
        if actual[name] != snapshot['protected_files'][name]:
            raise ValueError('Protected file changed: ' + name)
    return actual


def preflight(cfg):
    rows = {s: load_split(s) for s in ('train', 'valid')}
    expected = {'train': {'real': 500, 'fake': 500}, 'valid': {'real': 100, 'fake': 100}}
    for split, part in rows.items():
        if dict(Counter(r['label'] for r in part)) != expected[split]:
            raise ValueError('Unexpected split counts: ' + split)
        if file_sha256(ROOT / f'data/sample/{split}.csv') != cfg[f'{split}_manifest_sha256']:
            raise ValueError('Frozen manifest changed: ' + split)
        if len({r['video_id'] for r in part}) != len(part):
            raise ValueError('Duplicate video IDs')
        # Valid rows are deliberately marked False: never use them for gradients.
        gradient_allowed = 'True' if split == 'train' else 'False'
        if any(not r['group_id'] or r.get('training_allowed') != gradient_allowed
               or r.get('source_review_completed') != 'True' for r in part):
            raise ValueError('Missing group/source review or incorrect gradient-use role')
    for field in ('sha256', 'group_id', 'video_id'):
        if {r[field] for r in rows['train']} & {r[field] for r in rows['valid']}:
            raise ValueError('Train/valid overlap: ' + field)
    protected = protected_hashes()
    if protected['data/sample/test.csv'] != cfg['test_manifest_sha256']:
        raise ValueError('Test manifest changed')
    from image_model_backbone import source_identity
    identity = source_identity()
    contract = {
        'schema': 1, 'frames': cfg['frames_per_video'], 'size': cfg['image_size'],
        'sampling': cfg['sampling'], 'frozen_image_files': identity,
        'reader_sha256': file_sha256(ROOT / 'src/development_video_data.py'),
        'backbone_code_sha256': file_sha256(ROOT / 'src/image_model_backbone.py'),
        'video_reader_sha256': file_sha256(ROOT / 'src/video_pipeline.py'),
        'encoder_adapter_code_sha256': file_sha256(Path(__file__)),
        'feature_schema': '256 image embedding then P(fake)=1-P(real); RGB float32 0..255',
    }
    info = {'ready': True, 'counts': expected, 'config': cfg, 'feature_contract': contract,
            'config_sha256': file_sha256(CONFIG), 'code_sha256': file_sha256(Path(__file__)),
            'protected_files': protected, 'training_started': False,
            'test_predictions_computed': False, 'media_hash_policy': 'Every source hashed during feature preparation, including cache reuse'}
    return rows, info


def cache_path(row, contract):
    return CACHE / digest(contract) / (row['sha256'] + '.npz')


def read_feature(path, row, contract):
    with np.load(path, allow_pickle=False) as item:
        if str(item['source_sha256']) != row['sha256'] or str(item['contract_sha256']) != digest(contract):
            raise ValueError('Feature cache identity mismatch')
        values = item['features'].copy()
        times = item['timestamps_sec'].copy()
    if values.shape != (contract['frames'], 257) or not np.isfinite(values).all():
        raise ValueError('Invalid feature shape/values')
    if times.shape != (contract['frames'],) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid cached timestamps')
    if np.any((values[:, -1] < 0) | (values[:, -1] > 1)):
        raise ValueError('Invalid image score')
    return values


def prepare_features(rows, cfg, contract):
    import tensorflow as tf
    backbone = None
    started = time.monotonic()
    output = {}
    all_count = sum(map(len, rows.values()))
    count = reused = 0
    for split, part in rows.items():
        values = []
        for row in part:
            if file_sha256(ROOT / row['file_path']) != row['sha256']:
                raise ValueError('Source changed: ' + row['video_id'])
            path = cache_path(row, contract)
            if path.exists():
                feature = read_feature(path, row, contract)
                reused += 1
            else:
                if backbone is None:
                    from image_model_backbone import load_image_backbone
                    backbone, _ = load_image_backbone()
                frames, times, _ = sample_development_frames(row, cfg['frames_per_video'], cfg['image_size'])
                parts = []
                for frame in frames:
                    embedding, real_score = backbone(tf.convert_to_tensor(frame[None], dtype=tf.float32))
                    parts.append(np.concatenate([embedding.numpy()[0], [1 - float(real_score.numpy()[0, 0])]]))
                feature = np.asarray(parts, np.float32)
                path.parent.mkdir(parents=True, exist_ok=True)
                save_npz(path, features=feature, timestamps_sec=times,
                         source_sha256=row['sha256'], contract_sha256=digest(contract))
                feature = read_feature(path, row, contract)
            values.append(feature)
            count += 1
            if count == 1 or count % 10 == 0 or count == all_count:
                print(f'FEATURES {count}/{all_count}; cached={reused}; elapsed={time.monotonic()-started:.0f}s', flush=True)
        output[split] = (np.asarray(values, np.float32), np.asarray([int(r['label_id']) for r in part], np.float32))
    return output


def build_head(cfg, x_train):
    import tensorflow as tf
    inputs = tf.keras.Input((cfg['frames_per_video'], 257), name='frame_sequence')
    norm = tf.keras.layers.Normalization(mean=x_train.mean(axis=(0, 1)),
        variance=np.maximum(x_train.var(axis=(0, 1)), 1e-4), name='train_only_normalization')(inputs)
    temporal = tf.keras.layers.GRU(cfg['gru_units'], dropout=cfg['dropout'],
        kernel_regularizer=tf.keras.regularizers.L2(cfg['l2']), name='temporal_gru')(norm)
    score = tf.keras.layers.Dense(1, activation='sigmoid', name='probability_fake')(temporal)
    return tf.keras.Model(inputs, score, name='frozen_image_features_video_gru')


def predict_batches(model, x, batch):
    return np.concatenate([model(x[i:i+batch], training=False).numpy().reshape(-1)
                           for i in range(0, len(x), batch)])


def bce(y, score):
    p = np.clip(np.asarray(score, np.float64), 1e-7, 1-1e-7)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def save_curve(path, history):
    # Standalone SVG: no plotting package or network dependency.
    series = [('train_bce', '#2563eb'), ('valid_bce', '#dc2626')]
    ymax = max(0.1, max(h[key] for h in history for key, _ in series)) * 1.1
    tags = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="440" viewBox="0 0 800 440">',
            '<rect width="800" height="440" fill="white"/>',
            '<text x="65" y="30" font-family="sans-serif" font-size="20">Development loss (not independent test accuracy)</text>',
            '<path d="M65 55V365H760" fill="none" stroke="#334155"/>',
            '<text x="10" y="60">BCE</text><text x="700" y="400">Epoch</text>']
    for key, color in series:
        points = ' '.join(f'{65+695*i/max(1,len(history)-1):.2f},{365-310*h[key]/ymax:.2f}' for i,h in enumerate(history))
        tags.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    for i in range(5):
        tags.append(f'<text x="10" y="{369-310*i/4}" font-size="12">{ymax*i/4:.2f}</text>')
    tags += [f'<text x="65" y="390">1</text><text x="750" y="390">{len(history)}</text>',
             '<text x="90" y="425" fill="#2563eb">train BCE</text><text x="250" y="425" fill="#dc2626">validation BCE</text>', '</svg>']
    path.write_text('\n'.join(tags), encoding='utf-8')


def train_one(cfg, data, rows, seed, directory):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    x, y = data['train']
    vx, vy = data['valid']
    model = build_head(cfg, x)
    optimizer = tf.keras.optimizers.Adam(cfg['learning_rate'])
    objective = tf.keras.losses.BinaryCrossentropy()
    batch = cfg['batch_size']
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=False)

    @tf.function(reduce_retracing=True)
    def step(xx, yy):
        with tf.GradientTape() as tape:
            loss = objective(yy[:, None], model(xx, training=True)) + tf.add_n(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    best = float('inf')
    best_epoch = stale = 0
    history = []
    for epoch in range(1, cfg['epochs']+1):
        order = rng.permutation(len(x))
        objectives = []
        for start in range(0, len(x), batch):
            index = order[start:start+batch]
            objectives.append(float(step(x[index], y[index])))
        score = predict_batches(model, vx, batch)
        train_score = predict_batches(model, x, batch)
        value = bce(vy, score)
        entry = {'epoch': epoch, 'train_objective': float(np.mean(objectives)),
                 'train_bce': bce(y, train_score), 'valid_bce': value,
                 'train_accuracy': float(np.mean((train_score >= .5) == y)),
                 'valid_accuracy': float(np.mean((score >= .5) == vy))}
        history.append(entry)
        if value < best - cfg['min_delta']:
            best, best_epoch, stale = value, epoch, 0
            model.save(directory / 'best.keras')
        else:
            stale += 1
        dump(directory / 'history.json', history)
        print(f'SEED {seed} EPOCH {epoch}/{cfg["epochs"]} train_BCE={entry["train_bce"]:.4f} valid_BCE={value:.4f} valid_accuracy={entry["valid_accuracy"]:.3f} best={best_epoch}', flush=True)
        if stale >= cfg['patience']:
            break
    restored = tf.keras.models.load_model(directory / 'best.keras', compile=False)
    score = predict_batches(restored, vx, batch)
    result = {'seed': seed, 'epochs_run': len(history), 'best_epoch': best_epoch,
              'validation_bce': bce(vy, score), 'validation_metrics': baseline_common.metrics(vy, score),
              'checkpoint_sha256': file_sha256(directory / 'best.keras'),
              'test_evaluated': False, 'adopted': False}
    dump(directory / 'validation_predictions.json', [
        {'video_id': r['video_id'], 'label_id': int(r['label_id']), 'score_fake': float(p)}
        for r,p in zip(rows['valid'],score)])
    dump(directory / 'results.json', result)
    save_curve(directory / 'loss.svg', history)
    return result


def self_check(cfg):
    """Forward/serialization checks only. No optimizer and no weight updates."""
    import tempfile
    import tensorflow as tf
    from image_model_backbone import load_image_backbone
    from baseline_common import weight_hash
    tf.keras.utils.set_random_seed(42)
    synthetic = np.random.default_rng(42).normal(size=(4, cfg['frames_per_video'], 257)).astype(np.float32)
    model = build_head(cfg, synthetic)
    before = weight_hash(model)
    expected = model(synthetic, training=False).numpy()
    with tf.GradientTape() as tape:
        objective = tf.keras.losses.binary_crossentropy(
            tf.constant([[0.], [1.], [0.], [1.]]), model(synthetic, training=True))
        objective = tf.reduce_mean(objective) + tf.add_n(model.losses)
    gradients = tape.gradient(objective, model.trainable_variables)
    if any(g is None or not np.isfinite(g.numpy()).all() for g in gradients):
        raise ValueError('Missing or nonfinite GRU gradients')
    with tempfile.TemporaryDirectory(prefix='head_check_', dir=REPORTS) as temp:
        path = Path(temp) / 'untrained.keras'
        model.save(path)
        restored = tf.keras.models.load_model(path, compile=False)
        np.testing.assert_allclose(restored(synthetic, training=False), expected, atol=1e-6)
    if before != weight_hash(model):
        raise ValueError('Forward check changed weights')
    encoder, _ = load_image_backbone()
    features, score = encoder(tf.zeros((1,256,256,3), tf.float32))
    if tuple(features.shape) != (1,256) or tuple(score.shape) != (1,1):
        raise ValueError('Unexpected frozen image output')
    if not np.isfinite(features.numpy()).all() or not np.isfinite(score.numpy()).all():
        raise ValueError('Nonfinite frozen image output')
    # Test reader isolation without touching test videos.
    try:
        load_split('test')
    except ValueError:
        pass
    else:
        raise AssertionError('Test isolation failed')
    return {'passed': True, 'encoder_forward_shape': [1,256], 'head_roundtrip': True,
            'weight_updates': 0, 'gradients_finite_without_applying': True,
            'real_video_predictions': 0, 'test_reader_rejected': True,
            'head_parameter_count': model.count_params()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['preflight','self-check','prepare','train'], default='preflight')
    parser.add_argument('--runs', type=int, choices=[1,2,3], default=1)
    args = parser.parse_args()
    cfg = configuration()
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(cfg['cpu_threads'])
    tf.config.threading.set_inter_op_parallelism_threads(1)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows, info = preflight(cfg)
    dump(REPORTS / 'readiness.json', {**info, 'python': sys.executable,
         'tensorflow': tf.__version__, 'devices': [d.name for d in tf.config.list_physical_devices()]})
    print('READY: train=1000 (500/500), valid=200 (100/100); test sealed; existing model preserved.', flush=True)
    if args.action == 'preflight':
        print('Preflight only. No feature extraction or training started.', flush=True)
        return
    if args.action == 'self-check':
        check = self_check(cfg)
        protected_hashes()
        dump(REPORTS / 'self_check.json', check)
        print(json.dumps(check), flush=True)
        return
    data = prepare_features(rows, cfg, info['feature_contract'])
    # Preparation can be long; freeze again before any optimizer is created.
    _, current_info = preflight(cfg)
    if current_info != info or file_sha256(CONFIG) != info['config_sha256']:
        raise ValueError('Protocol, source model, or code changed during preparation')
    if args.action == 'prepare':
        print('Feature preparation complete; no GRU training started.', flush=True)
        return
    if args.runs > len(cfg['seeds']):
        raise ValueError('Not enough configured seeds')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = ROOT / 'checkpoints/development_gru_v1' / run_id
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'protocol.json', {**info, 'training_started': True, 'runs': args.runs,
         'normalization': 'train only', 'frozen_encoder': True,
         'limitations': ['Source-group independence is best effort', 'Landscape development differs from portrait-only test',
                        'Eight sampled frames can miss short events', 'No long-duration accuracy claim',
                        'No automatic deployment; combined existing model is preserved']})
    results = []
    try:
        for seed in cfg['seeds'][:args.runs]:
            results.append(train_one(cfg, data, rows, seed, output / f'seed_{seed}'))
        selected = min(results, key=lambda item: item['validation_bce'])
        protected_hashes()
        dump(output / 'summary.json', {'runs': results, 'selected_seed_by_validation_bce': selected['seed'],
             'status': 'research_training_complete_pending_independent_test', 'adopted': False,
             'test_predictions_computed': False, 'existing_model_replaced': False})
        print('COMPLETE:', output, '\nResearch candidate only. Test and current launcher are unchanged.', flush=True)
    except BaseException as error:
        dump(output / 'interrupted.json', {'error_type': type(error).__name__, 'error': str(error),
             'completed_runs': results, 'feature_cache_reusable': True,
             'partial_head_resume_supported': False})
        raise


if __name__ == '__main__':
    main()
