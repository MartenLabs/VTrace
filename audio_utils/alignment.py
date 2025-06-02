import numpy as np
from scipy.signal import correlate

def align_audio(a: np.ndarray, b: np.ndarray):
    """
    두 오디오 배열의 길이와 채널을 맞춤
    """
    # 차원 통일 (모노 → 스테레오 처리)
    def ensure_2d(x):
        return x[:, np.newaxis] if x.ndim == 1 else x

    a = ensure_2d(a)
    b = ensure_2d(b)

    # 채널 통일 (스테레오 ↔ 모노)
    if a.shape[1] != b.shape[1]:
        b = np.tile(b[:, 0:1], (1, a.shape[1])) if a.shape[1] > b.shape[1] else a[:, 0:1]

    # 길이 통일
    min_len = min(len(a), len(b))
    a = a[:min_len]
    b = b[:min_len]

    return a, b


def align_signals(reference: np.ndarray, target: np.ndarray, max_shift=2000):
    """
    reference (np.ndarray): 기준 신호
    target (np.ndarray): 정렬할 신호
    max_shift (int): 최대 오프셋 범위 (샘플 단위)
    
    Returns:
        aligned_target (np.ndarray): 정렬된 신호
        shift (int): 적용된 샘플 오프셋 (음수면 target이 앞으로 당겨짐)
    """
    ref_mono = reference[:, 0] if reference.ndim > 1 else reference
    tgt_mono = target[:, 0] if target.ndim > 1 else target
    
    corr = correlate(ref_mono, tgt_mono, mode='full')
    lag = np.argmax(corr) - len(tgt_mono) + 1

    # 안전 범위 클리핑
    lag = np.clip(lag, -max_shift, max_shift)
    
    aligned = np.roll(target, lag, axis=0)
    
    if lag > 0:
        aligned[:lag] = 0
    elif lag < 0:
        aligned[lag:] = 0
    
    print(f"🔧 Applied time shift: {lag} samples")
    return aligned, lag