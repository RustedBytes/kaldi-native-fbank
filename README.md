# kaldi-native-fbank (Rust)

Rust port of `kaldi-native-fbank`, providing FBANK, MFCC, Whisper-style mel features, STFT/ISTFT, and simple online/streaming wrappers built on top of `realfft`/`rustfft`.

## Features
- FBANK and MFCC extraction with configurable frame/mel options.
- Whisper mel frontend (80-bin, Slaney scale) compatible with the C reference.
- STFT/ISTFT utilities and raw frame copies for debugging.
- Online feature wrapper for streaming use cases.

## Quick start
Add the crate to your project (path or git as needed), then compute FBANKs:

```rust
use kaldi_native_fbank::{
    fbank::{FbankComputer, FbankOptions},
    window::{extract_window, Window},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut opts = FbankOptions::default();
    opts.frame_opts.dither = 0.0;

    let mut comp = FbankComputer::new(opts.clone())?;
    let win = Window::new(&opts.frame_opts).unwrap();
    let padded = opts.frame_opts.padded_window_size();

    // Single dummy frame (sine)
    let mut wave = (0..padded)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / opts.frame_opts.samp_freq).sin())
        .collect::<Vec<_>>();

    let mut window_buf = vec![0.0; padded];
    let raw_log_energy =
        extract_window(0, &wave, 0, &opts.frame_opts, win.as_ref(), &mut window_buf)?;

    let mut feat = vec![0.0; comp.dim()];
    comp.compute(raw_log_energy, 1.0, &mut window_buf, &mut feat);
    println!("{feat:?}");
    Ok(())
}
```

For streaming input, wrap a `FeatureComputer` in `online::OnlineFeature` and feed audio via `accept_waveform`.

## Running tests
```
cargo test --tests -- --nocapture
```
