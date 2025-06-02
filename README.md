# VTrace: Phase-Driven Vocal Tuner

**VTrace**는 AI 기반 보컬 제거 모델(Demucs)의 후처리 보조 도구로,  
보컬 음량 조절 및 자연스러운 보컬 감쇠 처리를 지원합니다.

<br/>

---

### 개발 배경

AI 보컬 제거 모델은 좋은 성능을 제공하지만

- **보컬을 완전히 제거하면** 배경 악기(저음역대, 리버브, 공간계 악기)까지 손실될 수 있습다.
    
- **보컬 추출 결과는** 너무 건조하여, 실제 음원에서 들리는 자연스러운 공간감이 사라집니다.
    

> **Phase-Driven Vocal Tuner는 이러한 한계를 보완합니다.**  
> AI 모델 출력에 부드러운 감쇠 처리를 추가해,  
> - 보컬 볼륨을 자연스럽게 줄이고,  
> - 배경 악기를 최대한 보존하며,  
> - 부드러운 보컬 추출까지 가능합니다.

<br/>

---

### 기능

- **보컬 감쇠 (Vocal Attenuation)**  
    보컬 볼륨을 완전히 제거하지 않고 사용자가 원하는 만큼 감쇠할 수 있습니다.
    
- **자연스러운 보컬 추출 (Smooth Extraction)**  
    AI 모델의 결과물보다 자연스럽게 약간의 공간감과 함께 보컬을 추출합니다.
    
- **악기 보존 (Instrumental Preservation)**  
    AI 모델에서 발생할 수 있는 배경 악기 손실을 최소화합니다.
    
- **학습 데이터 생성**  
    보컬 감쇠 처리 과정에서 생성된 **Residual Vocal**은 새로운 AI 학습 데이터로 재활용할 수 있습니다.
    
<br/>

---

### 핵심 특징

|기존 AI 모델|Phase-Driven Vocal Tuner 추가 시|
|---|---|
|보컬 완전 제거|보컬 볼륨 조절 (부드럽게 줄이기)|
|배경 손실 있음|배경 악기 보존|
|깨짐/뭉개짐 가능성 있음|깨지지 않고 자연스러운 보컬 추출|

<br/>
<br/>
<br/>

---

## 동작 원리

**Phase-Driven Vocal Tuner**는 AI 모델로 분리된 무보컬(instrumental) 음원과 원본(original) 음원을 혼합(blend)하여, 보컬을 완전히 제거하지 않고 볼륨을 조절(attenuation)하는 후처리 방식입니다.
또한, Residual Subtraction 및 Phase Cancel 기법을 통해 보컬 추출과 배경 복원까지 지원합니다.

<br/>

---

### 핵심 아이디어

* AI 분리 모델로 얻은 **Instrumental** 음원은 보컬이 제거되어 있으나, 일부 음질 손실(저음역대, 공간계 악기 등)이 발생할 수 있습니다.
* 따라서 원본 음원의 보컬 성분을 **감쇠(attenuate)** 하여 자연스러운 믹스를 만들어냅니다.
* 동시에 **Residual Subtraction**을 통해 보컬 성분만 추출하고, 이를 활용해 **Phase Cancel** 방식으로 배경 복원도 수행합니다.

<br/>

---

### 수식 표현

#### 보컬 감쇠 (Blend)

감쇠된 출력 음원 $B(t)$는 다음과 같이 계산됩니다:

$$B(t) = \alpha \cdot O(t) + (1 - \alpha) \cdot I(t)$$

* $O(t)$: 원본(original) 신호
* $I(t)$: AI 모델에서 추출된 무보컬(instrumental) 신호
* $\alpha$: 보컬 감쇠 비율 (0.0 \~ 1.0)

$$\text{즉, } \alpha = 0.0 \text{이면 보컬 제거, } \alpha = 1.0 \text{이면 원본 유지}$$

<br/>

#### Residual Subtraction (보컬 추출)

Residual Vocal은 다음과 같이 계산됩니다:

$$V_{\text{residual}}(t) = O(t) - I(t)$$

즉, 원본에서 무보컬을 빼면 보컬 성분만 남게 됩니다.

<br/>

#### Phase Cancel (배경 복원)

Residual Vocal을 원본에서 다시 빼면 배경(Instrumental)이 복원됩니다:

$$I_{\text{phase-cancel}}(t) = O(t) - V_{\text{residual}}(t)$$

<br/>

---

### Blend Mode (가중치 계산 방식)

* **Linear 모드**: $w = \alpha$
* **Exp 모드**: $w = e^{-\alpha}$
* **Log 모드**: $w = \log(1 + \alpha)$
* **Power 모드**: $w = \alpha^{\gamma}$ (기본 $\gamma = 2$)

다양한 가중치 계산 방식으로 감쇠 곡선을 유연하게 조절할 수 있습니다.

<br/>

---

### 처리 흐름

1. 원본과 무보컬 음원의 **샘플레이트** 및 **채널 수**를 일치시킴
2. 선택한 **Blend 모드**에 따라 가중치 계산
3. 두 음원을 합성:

$$blended = (original \times w_{\text{orig}}) + (instrumental \times w_{\text{inst}})$$

4. 클리핑 방지를 위해 **Amplitude Scaling**
5. Residual Vocal 추출:

$$residual = original - instrumental$$

6. Phase Cancel (배경 복원):

$$instrumental_{\text{phase-cancel}} = original - residual$$

7. 최종 출력은 `.wav`로 저장 (Normalize 처리 포함)

---

<br/>

### 처리 단계 요약

| 처리 단계                | 역할               | 출력 파일 예시                 |
| -------------------- | ---------------- | ------------------------ |
| Blend (보컬 감쇠)        | 보컬 볼륨 조절 및 믹스 생성 | `*_blend_알파.wav`         |
| Residual Subtraction | 보컬 성분 추출         | `*_vocal.wav`            |
| Phase Cancel         | 배경 복원 (보컬 제거)    | `*_phase_cancel_raw.wav` |

---

### 정리

**Phase-Driven Vocal Tuner**는

* AI 분리 모델의 한계(보컬 제거 시 배경 손실)를 보완하고,
* 보컬 음량 조절 및 추출 기능을 통해 자연스러운 믹스와 AI 학습용 데이터셋 생성을 지원합니다.


---

<br/>

---

### 사용 시나리오

- 보컬 볼륨을 줄여서 배경과 어우러지는 믹스 제작
    
- 보컬의 존재감은 살리되, 원곡의 배경 악기와 자연스럽게 섞이도록 처리
    
- 깨지지 않고 부드러운 보컬 추출 (감정, 발음 유지)
    
- Residual Vocal을 활용한 데이터셋 구축 및 AI 모델 학습

<br/>

---

### 샘플 결과 (Alpha별 보컬 감쇠 및 추출 예시)

VTrace 샘플 결과
각 곡에 대해 Alpha(보컬 감쇠 파라미터)를 다르게 설정하여,  보컬 음량과 보컬 추출 결과를 측정한 결과.


[VTrace samples](https://martenlabs.github.io/posts/VTrace/)


<br/>
<br/>
<br/>

---

## 설치 방법

VTrace는 Conda 환경으로 제공됩니다. [Anaconda](https://www.anaconda.com/download), [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

```bash
conda env create -f VTrace.yaml
```

설치가 완료되면, 다음 명령어로 환경을 활성화합니다:

```bash
conda activate vtrace
```

> **참고:**
> `VTrace.yaml`은 이 프로젝트의 최상위 디렉토리에 포함되어 있습니다.
> 필요한 Python 버전 및 라이브러리 (numpy, librosa, soundfile, demucs 등)도 함께 설치됩니다.


<br/>

---

## 실행 방법

### 기본 실행 명령어

```bash
python main.py -i <입력 폴더/파일 경로> [옵션들...]
```

또는 유튜브 링크를 입력하여 다운로드 + 처리:

```bash
python main.py -l <유튜브 링크> [옵션들...]
```

<br/>

---

### 주요 옵션

| 옵션                     | 설명                                         | 기본값/예시                                 |
| ---------------------- | ------------------------------------------ | -------------------------------------- |
| `-i`, `--input`        | 입력 파일 또는 폴더 경로                             | `input/` 또는 `song.wav`                 |
| `-l`, `--link`         | 유튜브 링크 (MP3 다운로드 후 처리)                     | `https://www.youtube.com/watch?v=xxxx` |
| `-o`, `--output`       | 출력 폴더 경로 (생략 시 입력 폴더 하위에 생성)               | `results/`                             |
| `-a`, `--alpha`        | 보컬 감쇠 비율 (0.0 \~ 1.0)                      | `0.5` (기본: `0.1`)                      |
| `--blend-mode`         | Blend 방식 (`linear`, `exp`, `log`, `power`) | `linear` (기본: `linear`)                |
| `--demucs-model`       | Demucs 모델명 (예: `htdemucs_ft`)              | `htdemucs_ft` (기본: `htdemucs_ft`)      |
| `--threshold`          | Noise Gate 임계값 (dB)                        | `-40` (기본: `-40`)                      |
| `-T`, `-t`, `--thread` | 동시 처리할 스레드 개수                              | `2` (기본: `2`)                          |
| `--cleanup`            | Demucs 분리 결과 폴더 삭제 여부                      | 지정 시 `True`                            |
| `--eval`               | SDR/SIR/dBFS 평가 실행 여부                      | 지정 시 `True`                            |

<br/>

---

### 실행 예시

폴더 전체 처리 (보컬 감쇠 0.3, 로그 블렌드 모드):

```bash
python main.py -i songs/ -a 0.3 --blend-mode log
```

유튜브 링크 다운로드 후 처리 (보컬 감쇠 0.5):

```bash
python main.py -l https://www.youtube.com/watch?v=xxxx -a 0.5
```

Noise Gate 임계값 조정 + 평가 실행:

```bash
python main.py -i songs/ --threshold -35 --eval
```

<br/>

---

### 기본 설정 (config.yaml)

별도의 `config.yaml`에서 기본값(알파, 블렌드 모드 등)을 설정 가능.
명령줄 옵션이 우선 적용됩니다.

<br/>

---

### 요구사항

* Python >= 3.10
* 필요한 패키지: numpy, librosa, soundfile, demucs, ffmpeg, yt-dlp 등
  (Conda 환경 파일 `VTrace.yaml` 제공)

<br/>

---
### 파일 구조
``` txt
VTrace/
├── main.py                       # VTrace 실행 메인 엔트리포인트 (CLI)
├── config_loader.py              # config.yaml 로드 및 파라미터 관리
├── logger.py                     # 로그 설정 및 출력 관리
├── VTrace.yaml                   # Conda 환경 설정 파일
├── config.yaml                   # VTrace 기본 설정 (alpha, blend_mode 등)
├── README.md                     # 프로젝트 설명
├── logs/                         # 로그 파일 저장 디렉토리
├── processors/                   # 오디오 처리 모듈
│   ├── blend.py                  # Blend 처리 함수 (원본 + 분리음 조합)
│   ├── residual_subtraction.py   # Residual Vocal 추출 (Original - Blend)
├── separated/                    # Demucs 분리 결과 저장 디렉토리
├── utils/                        # 유틸리티 함수 모음
│   ├── audio_utils.py            # 오디오 관련 유틸 (Normalize, Noise Gate 등)
│   ├── evaluate_metrics.py       # SDR, SIR, dBFS 평가 메트릭 계산
│   ├── evaluation.py             # 평가 모듈 (분석 파이프라인)
│   ├── file_utils.py             # 파일/경로 처리 유틸
│   ├── youtube.py                # 유튜브 다운로드 처리 모듈

```

<br/>

---
### 버전

VTrace 0.1 (Dev)

<br/>

---

### 결과물

* 보컬 감쇠 오디오 파일 (`*_blend_알파.wav`, `*_phase_cancel_raw`, `*_phase_cancel_norm`)
* 추출된 Residual Vocal (`*_vocal.wav`)
* (선택) 평가 결과: SDR/SIR/dBFS 로그

<br/>

---

### 향후 계획

- **Residual Vocal 품질 향상을 위한 후처리 기능 추가**  
    (Denoising, 타이밍 보정 등)
    
    
- **GUI 기반 툴킷 제공**  
    간단한 조작으로 보컬 감쇠 및 출력 파일 생성 기능 제공
    

- **Noise Gate 및 Smoothstep 기반 감쇠 모드 추가**  
    더욱 세밀한 감쇠 효과 구현
    

<br/>

---

### 🔒 라이선스

MIT License © 2025 JUNHEE LEE


| 라이브러리/모델      | 라이선스      | 출처                                                                                   |
| ------------- | --------- | ------------------------------------------------------------------------------------------ |
| **Demucs**    | MIT       | [https://github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs)   |

---