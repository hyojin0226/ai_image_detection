"""64-frame segmented inference and exact video-score Grad-CAM; no training."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import baseline_common
import numpy as np
from PIL import Image, ImageDraw
from video_pipeline import ROOT, file_sha256, imageio_ffmpeg
from development_video_data import sample_development_frames
from predict_final_research import RELEASE, verify
from improve_existing_features import pooled, predict
from video_explanation import positive_gradcam, overlay_heatmap, strongest_zone

MODE = 'segmented64_v1'


def progress(message):
    print(message, flush=True)


def is_black(frame):
    # Only an effectively solid black picture. Dark scenes/text are retained.
    return bool(np.max(frame) <= 3)


def decode_at(path, timestamp):
    vf = 'scale=256:256:force_original_aspect_ratio=decrease,pad=256:256:(ow-iw)/2:(oh-ih)/2:black,setsar=1'
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-nostdin', '-threads', '1',
               '-ss', f'{timestamp:.6f}', '-i', str(path), '-an', '-frames:v', '1', '-vf', vf,
               '-threads', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    output = subprocess.run(command, capture_output=True, timeout=60)
    if output.returncode or len(output.stdout) != 256 * 256 * 3:
        raise ValueError(f'{timestamp:.3f}초 프레임을 읽지 못했습니다.')
    return np.frombuffer(output.stdout, np.uint8).reshape(256, 256, 3)


def select_segments(frames, times, duration, decode):
    """Eight unique nonblack frames per retained segment; never pad duplicates."""
    if len(frames) != 64 or len(times) != 64:
        raise ValueError('Expected 64 planned sampling positions')
    groups, skipped, black = [], [], []
    for segment in range(8):
        start, end = segment * duration / 8, (segment + 1) * duration / 8
        candidates = []
        for i in range(segment * 8, (segment + 1) * 8):
            if is_black(frames[i]):
                black.append(float(times[i]))
            else:
                candidates.append((float(times[i]), frames[i]))
        if len(candidates) < 8:
            # Shifted positions, never identical to the original eight centers.
            for j in range(16):
                t = start + (j + .5) * (end - start) / 16
                frame = decode(t)
                if is_black(frame):
                    black.append(float(t))
                else:
                    candidates.append((float(t), frame))
        if len(candidates) < 8:
            skipped.append({'segment': segment + 1, 'start_sec': start, 'end_sec': end,
                            'reason': '검은 화면 제외 후 서로 다른 유효 시점 8개를 확보하지 못해 구간 점수를 계산하지 않음',
                            'nonblack_candidates': len(candidates)})
            continue
        candidates.sort(key=lambda x: x[0])
        indices = np.floor((np.arange(8) + .5) * len(candidates) / 8).astype(int)
        chosen = [candidates[int(i)] for i in indices]
        groups.append({'segment': segment + 1, 'start_sec': start, 'end_sec': end,
                       'times': np.asarray([x[0] for x in chosen]),
                       'frames': np.stack([x[1] for x in chosen])})
    return groups, skipped, black


def load_members():
    frozen = json.loads((RELEASE / 'frozen.json').read_text(encoding='utf-8'))
    members = []
    for member in frozen['selected']['members']:
        with np.load(RELEASE / member['file'], allow_pickle=False) as saved:
            model = {k: saved[k].copy() for k in saved.files}
        members.append((model, member['spec']))
    return members


def score_sequences(sequence, members):
    values = []
    for model, spec in members:
        x, _ = pooled(sequence, spec['representation'])
        values.append(predict(model, x, spec))
    return np.mean(values, axis=0)


def make_differentiable_head(members):
    """Same temporal statistics, clipping, kernels and sigmoid as the NumPy head."""
    import tensorflow as tf

    @tf.custom_gradient
    def std_time(x):
        centered = x - tf.reduce_mean(x, axis=0, keepdims=True)
        std = tf.sqrt(tf.reduce_mean(tf.square(centered), axis=0))
        def gradient(upstream):
            # Population std, with finite zero derivative at a constant feature.
            return upstream[None, :] * tf.math.divide_no_nan(centered, 8 * std[None, :])
        return std, gradient

    constants = [({k: tf.constant(v, tf.float64) for k, v in model.items()}, spec)
                 for model, spec in members]

    @tf.function(input_signature=[tf.TensorSpec((8, 257), tf.float64)])
    def head(sequence):
        embedding, scores = sequence[:, :256], sequence[:, 256:257]
        statistics = tf.stack([tf.reduce_mean(scores), std_time(scores)[0],
                               tf.reduce_min(scores), tf.reduce_max(scores)])
        x = tf.concat([tf.reduce_mean(embedding, axis=0), std_time(embedding),
                       tf.reduce_mean(tf.abs(embedding[1:] - embedding[:-1]), axis=0), statistics], axis=0)
        values = []
        for model, spec in constants:
            if spec['representation'] != 'temporal' or spec['kernel'] != 'rbf':
                raise ValueError('Unexpected frozen kernel head')
            z = tf.clip_by_value((x - model['mean']) / model['scale'], -10., 10.) * model['block']
            support = model['support']
            distance = tf.maximum(tf.reduce_sum(z * z) + tf.reduce_sum(support * support, axis=1)
                                  - 2 * tf.linalg.matvec(support, z), 0.)
            residual = tf.reduce_sum(tf.exp(-spec['gamma'] * distance) * model['coefficient'])
            values.append(tf.sigmoid(tf.clip_by_value(4 * residual, -60., 60.)))
        return tf.reduce_mean(tf.stack(values))
    return head


def replacement_records(sequences, segment_scores, members):
    """Rank by changes to the final mean score, not by frame heatmap brightness."""
    overall = float(np.mean(segment_scores))
    records = []
    for g, sequence in enumerate(sequences):
        changed = np.repeat(sequence[None], 8, axis=0)
        for i in range(8):
            changed[i, i] = sequence.mean(axis=0)
        substituted = score_sequences(changed, members)
        for score in substituted:
            alternative = overall + (float(score) - float(segment_scores[g])) / len(sequences)
            records.append({'replacement_video_score': alternative,
                            'video_score_delta': overall - alternative})
    return records


def rank_frames(records):
    ordered = sorted(records, key=lambda r: (-abs(r['video_score_delta']), r['timestamp_sec'], r['index']))
    main = [r['index'] for r in ordered[:8]]
    return main, [r['index'] for r in records if r['index'] not in set(main)]


def explain_frames(groups, sequences, segment_scores, members, bundle, folder, progress_fn=progress):
    import tensorflow as tf
    _, info, prefix, suffix = bundle
    if info['source_files_sha256'] != json.loads((RELEASE / 'frozen.json').read_text(encoding='utf-8'))['feature_contract']['frozen_image_files']:
        raise ValueError('Grad-CAM weights differ from frozen prediction weights')
    head = make_differentiable_head(members)
    overall = float(np.mean(segment_scores))
    target_fake = overall >= .5
    records = replacement_records(sequences, segment_scores, members)
    index = 0
    for g, group in enumerate(groups):
        np.testing.assert_allclose(float(head(tf.constant(sequences[g], tf.float64))),
                                   segment_scores[g], rtol=1e-5, atol=1e-6)
        for offset, (frame, timestamp) in enumerate(zip(group['frames'], group['times'], strict=True)):
            number = index + 1
            activation = prefix(tf.convert_to_tensor(frame[None], tf.float32))
            with tf.GradientTape() as tape:
                tape.watch(activation)
                embedding, real = suffix(activation)
                features = tf.concat([embedding[0], 1 - real[0]], axis=0)
                updated = tf.tensor_scatter_nd_update(tf.constant(sequences[g], tf.float64),
                                                      [[offset]], tf.cast(features[None], tf.float64))
                segment_score = head(updated)
                video_score = (segment_score + float(np.sum(segment_scores) - segment_scores[g])) / len(groups)
                target = video_score if target_fake else 1 - video_score
            np.testing.assert_allclose(features.numpy(), sequences[g, offset], rtol=2e-4, atol=2e-4)
            np.testing.assert_allclose(float(video_score), overall, rtol=1e-4, atol=1e-5)
            gradient = tape.gradient(target, activation)
            if gradient is None:
                raise ValueError('영상 점수와 공간 특징 사이의 Grad-CAM 연결을 확인하지 못했습니다.')
            heat = positive_gradcam(activation.numpy()[0], gradient.numpy()[0])
            record = records[index]
            original_path = folder / f'frame_{number:02d}.png'
            comparison_path = folder / f'comparison_{number:02d}.png'
            Image.fromarray(frame).save(original_path)
            if heat is not None:
                overlay, resized = overlay_heatmap(frame, heat)
                overlay_path = folder / f'gradcam_{number:02d}.png'
                overlay.save(overlay_path)
                zone = strongest_zone(resized, frame)
            else:
                overlay, zone = Image.fromarray(frame), None
                overlay_path = None
            comparison = Image.new('RGB', (512, 280), 'white')
            comparison.paste(Image.fromarray(frame), (0, 24))
            comparison.paste(overlay, (256, 24))
            draw = ImageDraw.Draw(comparison)
            draw.text((6, 5), 'Original', fill='black')
            draw.text((262, 5), 'Grad-CAM: ' + ('AI' if target_fake else 'REAL') if heat is not None else 'No positive signal', fill='black')
            comparison.save(comparison_path)
            record.update(index=number, timestamp_sec=float(timestamp), segment=group['segment'],
                          original_path=str(original_path.resolve()), comparison_path=str(comparison_path.resolve()),
                          gradcam_path=str(overlay_path.resolve()) if overlay_path else None,
                          gradcam_available=heat is not None, strongest_zone=zone,
                          image_fake_score=float(sequences[g, offset, 256]),
                          target_class='fake' if target_fake else 'real',
                          note=None if heat is not None else '양의 Grad-CAM 신호가 없어 색을 표시하지 않습니다.')
            index += 1
            progress_fn(f'Grad-CAM {index}/{len(records)}')
    return records


def analyze(source, output, progress_fn=progress):
    manifest = verify()
    import tensorflow as tf
    import image_model_backbone as backbone
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    backbone.SOURCE = RELEASE / 'image_model'
    started = time.monotonic()
    local = ROOT / '.local'
    local.mkdir(exist_ok=True)
    previews = local / 'analyzed_frames64'
    previews.mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='analysis_', dir=previews))
    with tempfile.TemporaryDirectory(prefix='input64_', dir=local) as temporary:
        if source.startswith(('https://', 'http://')):
            from shorts_worker import download
            path, provenance = download(source, Path(temporary), lambda stage, message: progress_fn(message), max_duration_sec=40)
        else:
            path = Path(source).expanduser().resolve()
            if path.is_relative_to((ROOT / 'data/sample/test').resolve()):
                raise ValueError('보존된 Test는 개별 분석 입력으로 사용하지 않습니다.')
            if not path.is_file() or not 0 < path.stat().st_size <= 90 * 1024 * 1024:
                raise ValueError('90 MB 이하의 영상 파일을 선택하세요.')
            provenance = {'kind': 'file', 'title': path.name}
            destination = Path(temporary) / ('input' + path.suffix)
            shutil.copyfile(path, destination)
            path = destination
        progress_fn('영상 전체에서 64개 시점을 추출하고 있습니다.')
        source_sha = file_sha256(path)
        frames, times, metadata = sample_development_frames(
            {'file_path': str(path.relative_to(ROOT)), 'sha256': source_sha}, 64, 256)
        duration = float(metadata['duration'])
        groups, skipped, black = select_segments(frames, times, duration, lambda t: decode_at(path, t))
        if not groups:
            raise ValueError('검은 화면 제외 후 분석 가능한 구간이 없습니다. 실제 영상으로 기본 분류하지 않습니다.')
        progress_fn(f'유효 프레임 {len(groups) * 8}/64 · 이미지 모델을 불러오고 있습니다.')
        encoder, _ = backbone.load_image_backbone()
        bundle = backbone.load_image_backbone(include_gradcam=True)
        sequences = []
        for group in groups:
            sequence = []
            for frame in group['frames']:
                embedding, real = encoder(tf.convert_to_tensor(frame[None], tf.float32))
                sequence.append(np.concatenate([embedding.numpy()[0], [1 - float(real.numpy()[0, 0])]]))
            sequences.append(np.asarray(sequence, np.float32))
            progress_fn(f"구간 {group['segment']}/8 · 프레임 특징 분석 완료")
        sequences = np.stack(sequences)
        members = load_members()
        segment_scores = score_sequences(sequences, members)
        score = float(np.mean(segment_scores))
        records = explain_frames(groups, sequences, segment_scores, members, bundle, folder, progress_fn)
    main, details = rank_frames(records)
    result = {
        'status': 'completed', 'model': manifest['release'], 'analysis_mode': MODE, 'threshold': .5,
        'source': provenance, 'source_sha256': source_sha, 'duration_sec': duration,
        'fake_score': score, 'prediction': 'fake' if score >= .5 else 'real',
        'prediction_ko': 'AI 생성 영상 추정' if score >= .5 else '실제 촬영 영상 추정',
        'planned_frames': 64, 'analyzed_frames': len(records), 'segments_used': len(groups),
        'black_seek_positions_sec': black, 'skipped_segments': skipped,
        'segments': [{'segment': group['segment'], 'start_sec': group['start_sec'], 'end_sec': group['end_sec'],
                      'fake_score': float(segment_score)} for group, segment_score in zip(groups, segment_scores, strict=True)],
        'frames': records, 'main_frame_indices': main, 'detail_frame_indices': details,
        'preview_directory': str(folder.resolve()), 'elapsed_sec': round(time.monotonic() - started, 1),
        'aggregation': 'Equal mean of retained segment scores; each segment uses the frozen 8-frame head',
        'ranking': 'Absolute final-video score change after replacing one feature row with its segment mean; sensitivity, not causal proof',
        'gradcam': {'method': 'Gradient of the selected overall video class through the exact frozen RBF video head and EfficientNet spatial layer',
                    'target': 'fake' if score >= .5 else 'real', 'spatial_tensor': bundle[1]['gradcam_tensor'],
                    'other_segments_held_fixed': True, 'forward_parity_checked_each_frame': True},
        'accuracy_validated_for_this_mode': False, 'score_is_calibrated_probability': False,
        'limitations_ko': '64프레임·구간 평균 방식은 정확도 재검증 전입니다. 기존 8프레임 정확도를 적용할 수 없습니다. '
                          'Grad-CAM은 선택한 분류를 지지하는 신호이며 합성 위치나 생성 사실의 증명이 아닙니다. '
                          '색은 각 프레임 안에서 정규화하므로 프레임끼리 진하기를 비교하지 마세요.'}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (folder / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    progress_fn('완료 · 주요 프레임 8개와 세부사항을 확인하세요.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / '.local/analysis64_result.json')
    args = parser.parse_args()
    try:
        analyze(args.input, args.output)
    except Exception as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'status': 'failed', 'error': str(error)}, ensure_ascii=False), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
