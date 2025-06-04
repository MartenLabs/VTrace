
# VTrace: Phase-Driven Vocal Tuner

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

VTrace 0.6.0 (Dev)

**VTrace is a post-processing tool specifically designed to work with the Demucs AI vocal separation model, offering vocal volume adjustment and natural vocal attenuation.**

This pipeline is built around the output structure of Demucs (i.e., `vocals.wav` and `no_vocals.wav` inside the `separated/demucs_model_name` folder), enabling fully automated, phase-consistent vocal extraction.

**Note:** VTrace is not directly compatible with other models like VR Arch, MDX-Net, or Spleeter. For these models, you must manually separate audio into `vocals` and `no_vocals` files and place them in the appropriate folder structure.

<br/>

---

### Background

While AI vocal separation models provide impressive results, they often suffer from the following issues:

- **Complete vocal removal** can also result in the loss of background instruments, such as low-frequency elements, reverbs, and spatial effects.
- **Isolated vocals** often sound too dry, lacking the natural spatial feel present in the original track.

<br/>

> **By applying smooth attenuation to the AI model's output, VTrace enables:**
> 
> - Natural reduction of vocal volume,
> - Maximum preservation of background instruments,
> - And smooth, natural-sounding vocal extraction.

<br/>


---

### Features

- **Vocal Attenuation**  
    Adjust the vocal volume without completely removing it, allowing users to control how much the vocals are reduced.

- **Smooth Extraction**  
    Extract vocals with a more natural sound and a subtle spatial feel, compared to raw AI model outputs.

- **Instrumental Preservation**  
    Minimize background instrument loss that can occur with AI models.

- **Training Data Generation**  
    The **Residual Vocal** generated during the attenuation process can be reused as new AI training data.

<br/>

---

### Key Highlights

|AI Model Only|With VTrace|
|---|---|
|Complete vocal removal|Adjustable vocal volume (smooth reduction)|
|Background loss|Background instruments preserved|
|Possible distortion/muddiness|Smooth, natural vocal extraction without distortion|

<br/>
<br/>
<br/>

---

## How It Works

**VTrace** is a post-processing method that blends the AI-separated instrumental track with the original track, allowing for controlled vocal attenuation instead of complete removal.  
It also supports vocal extraction and background restoration through **Residual Subtraction** and **Phase Cancel** techniques.

<br/>

---

### Core Concepts

- The **Instrumental** output from the AI separation model often lacks vocals but can suffer from quality loss in low frequencies and spatial instruments.
- VTrace attenuates the vocal components in the original track to create a more natural mix.
- It also performs **Residual Subtraction** to isolate vocal elements, and **Phase Cancel** techniques to restore the background audio.


<br/>

---

### Mathematical Expressions

#### Vocal Attenuation (Blend)

The attenuated output $B(t)$ is calculated as:

$$B(t) = \alpha \cdot O(t) + (1 - \alpha) \cdot I(t)$$

- $O(t)$: Original signal
- $I(t)$: AI-extracted instrumental signal
- $\alpha$: Vocal attenuation ratio (0.0 ~ 1.0)

$$\text{That is, } \alpha = 0.0 \text{ removes vocals completely, } \alpha = 1.0 \text{ retains the original.}$$

<br/>

#### Residual Subtraction (Vocal Extraction)

Residual Vocal is calculated as:

$$V_{\text{residual}}(t) = O(t) - I(t)$$

Subtracting the instrumental from the original leaves the vocal component.

<br/>

#### Phase Cancel (Background Restoration)

Subtracting the Residual Vocal from the original restores the background (Instrumental):

$$I_{\text{phase-cancel}}(t) = O(t) - V_{\text{residual}}(t)$$

<br/>

---

### Blend Mode (Weighting Functions)

- **Linear Mode**: $w = \alpha$
- **Exp Mode**: $w = e^{-\alpha}$
- **Log Mode**: $w = \log(1 + \alpha)$
- **Power Mode**: $w = \alpha^{\gamma}$ (default $\gamma = 2$)

These modes allow flexible control over the attenuation curve.

<br/>

---

### Processing Flow

1. Match the **sample rate** and **number of channels** between the original and instrumental tracks.
2. Compute weights based on the selected **Blend Mode**.
3. Mix the two audio tracks:

$$blended = (original \times w_{\text{orig}}) + (instrumental \times w_{\text{inst}})$$

4. Apply **Amplitude Scaling** to prevent clipping.
5. Extract Residual Vocal:

$$residual = original - instrumental$$

6. Phase Cancel (Background Restoration):

$$instrumental_{\text{phase-cancel}} = original - residual$$

7. Save the final output as `.wav` files (with normalization).

---

<br/>

### Processing Stages Summary

| Stage                   | Purpose                        | Example Output Files                |
| ----------------------- | ----------------------------- | ---------------------------------- |
| Blend (Vocal Attenuation)   | Adjust vocal volume, create mix | `*_blended.wav`                     |
| Residual Subtraction     | Extract vocal components       | `*_vocal_residual.wav`              |
| Phase Cancel             | Background restoration (vocal removal) | `*_phase_cancel.wav`      |


<br/>

---

### Use Cases

- Create mixes by reducing the vocal volume to blend naturally with the background.
- Maintain vocal presence while smoothly integrating with the original instrumental.
- Extract smooth, unbroken vocals (retain emotion and articulation).
- Use **Residual Vocal** data for dataset generation and AI model training.

<br/>

---

### Sample Results (Vocal Attenuation by Alpha)

VTrace sample results:
For each track, different Alpha (vocal attenuation parameter) settings were tested, comparing vocal volume and extracted results.

[VTrace samples](https://martenlabs.github.io/posts/VTrace/)

<br/>
<br/>
<br/>


---

## Installation

VTrace is provided as a Conda environment.  
Install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.

```bash
conda env create -f VTrace.yaml
```

Once installed, activate the environment:

```bash
conda activate vtrace
```

<br/>

>Note:
>VTrace.yaml is located in the root directory of this project.
>All required Python version and libraries (numpy, librosa, soundfile, demucs, etc.) will be >installed together.

<br/>

---

## Usage

### Basic Command

```bash
python main.py -i <input folder/file path> [options...]
```

Or process a YouTube link directly (download + process):

```bash
python main.py -l <YouTube link> [options...]
```

<br/>

---

### Main Options

| Option                  | Description                                                   | Default/Example                              |
| ----------------------- | ------------------------------------------------------------- | ------------------------------------------- |
| `-i`, `--input`         | Path to input file or folder                                  | `input/` or `song.wav`                       |
| `-l`, `--link`          | YouTube link (downloads MP3 and processes)                    | `https://www.youtube.com/watch?v=xxxx`       |
| `-o`, `--output`        | Output folder path (defaults to subfolder of input)           | `results/`                                   |
| `-a`, `--alpha`         | Vocal attenuation ratio (0.0 ~ 1.0)                           | `0.5` (default: `0.1`)                       |
| `--blend-mode`          | Blend mode (`linear`, `exp`, `log`, `power`)                  | `linear` (default: `linear`)                 |
| `--demucs-model`        | Demucs model name (e.g., `htdemucs_ft`)                       | `htdemucs_ft` (default: `htdemucs_ft`)       |
| `--device`              | Processing device (`cpu`, `cuda`, `mps`)                      | Auto-detected (default based on system)      |
| `-T`, `-t`, `--thread`  | Number of parallel **processes** to handle multiple files      | `1` (default: `1`) |
| `--cleanup`             | Whether to delete Demucs output folders                       | Set `True` to enable                         |
| `--eval`                | Run restoration evaluation (MSE, Cosine, STOI)                | Set `True` to enable                         |
| `--convert_to_mp3`      | Whether to convert output to MP3                              | Set `True` to enable                         |


<br/>

---

### Example Commands

Process entire folder (70% vocal attenuation, log blend mode):

```bash
python main.py -i songs/ -a 0.3 --blend-mode log
```

Download from YouTube and process (85% vocal attenuation):

```bash
python main.py -l https://www.youtube.com/watch?v=xxxx -a 0.15
```

Process entire folder with MP3 conversion and evaluation enabled:

```bash
python main.py -i songs/ --convert_to_mp3 --eval
```

Process all songs in the entire folder using 8 parallel workers, with custom blend settings, evaluation metrics enabled, MP3 conversion, and GPU acceleration (MPS for Apple Silicon):

``` bash
python main.py -i songs/ -o output/ -t 8 -ba 0.1 -va 1 --cleanup --convert_mp3 --eval --device mps
```
<br/>

---

### Default Settings (config.yaml)

You can define default values such as `alpha` and `blend_mode` in the `config.yaml` file.  
Command-line options will override these values.

<br/>

---

### Requirements

* Python >= 3.10
* Required packages: numpy, librosa, soundfile, demucs, ffmpeg, yt-dlp, etc.
  (Conda environment file `VTrace.yaml` provided)

<br/>

---

### File Structure

``` txt
VTrace/
├── main.py                       # Main entry point for VTrace (CLI)
├── config_loader.py              # Load and manage config.yaml parameters
├── logger.py                     # Logging setup and management
├── VTrace.yaml                   # Conda environment configuration (dependency management)
├── config.yaml                   # VTrace default configuration (alpha, blend_mode, model, etc.)
├── README.md                     # Project documentation
├── logs/                         # Directory for log files
│
├── processors/                   # Core VTrace processing modules
│   ├── vtrace_core.py            # VTrace core functions (Residual Vocal, Phase Cancel, Blend)
│
├── separated/                    # Directory for Demucs separation outputs (vocals.wav, no_vocals.wav)
│
├── audio_utils/                  # Audio utility functions
│   ├── alignment.py              # Audio phase and channel alignment (align_audio, align_signals)
│   ├── audio_conversion.py       # WAV to MP3 conversion (convert_wav_to_mp3)
│   ├── experimental.py           # Experimental functions (under development)
│   ├── filters.py                # Noise gate and filter processing (includes test code)
│   ├── loudness.py               # Gain adjustment functions (includes test code)
│
├── utils/                        # General utilities and evaluation modules
│   ├── evaluate_metrics.py       # MSE, Cosine, STOI evaluation metrics
│   ├── evaluation.py             # Evaluation pipeline management (results comparison, analysis)
│   ├── file_utils.py             # File/path utilities (file names, paths, etc.)
│   ├── youtube.py                # YouTube download module (yt_dlp integration)
```

<br/>

---
### Version

VTrace 0.6.0 (Dev)

<br/>

---

### Output Files

- Vocal attenuation audio files (`*_blended.wav`, `*_instrumental_phase_cancel.wav`)
- Extracted Residual Vocal (`*_vocal_residual.wav`)
- (Optional) Evaluation results: MSE, Cosine, STOI

<br/>

---

### Future Plans

- [x] **Add post-processing features to improve Residual Vocal quality**  
      (e.g., timing alignment, etc.)

- [ ] **Optional Web-based GUI for Non-technical Users (Experimental)**
      Provide a lightweight web interface using Gradio or Streamlit

- [ ] **Add Noise Gate and Smoothstep-based attenuation modes**  
      Implement more precise attenuation effects

<br/>

---

### 🔒 License

MIT License © 2025 JUNHEE LEE

| Library/Model    | License    | Source                                                                                  |
| ---------------- | ---------- | --------------------------------------------------------------------------------------- |
| **Demucs**       | MIT        | [https://github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs) |

---
