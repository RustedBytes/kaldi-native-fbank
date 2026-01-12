use kaldi_native_fbank::fbank::{FbankComputer, FbankOptions};
use kaldi_native_fbank::istft_compute;
use kaldi_native_fbank::mel::{MelBanks, MelOptions};
use kaldi_native_fbank::online::{FeatureComputer, OnlineFeature};
use kaldi_native_fbank::rfft::Rfft;
use kaldi_native_fbank::whisper::{WhisperComputer, WhisperOptions};
use kaldi_native_fbank::window::{extract_window, FrameOptions, Window};
use kaldi_native_fbank::MfccComputer;
use kaldi_native_fbank::MfccOptions;
use kaldi_native_fbank::IstftOptions;
use kaldi_native_fbank::StftOptions;
use rand::Rng;
use std::f32::consts::PI;
use kaldi_native_fbank::stft_compute;

#[test]
fn test_feature_demo() {
    let sample_rate = 16000.0;
    let num_seconds = 1;
    let num_samples = (sample_rate * num_seconds as f32) as usize;
    let frames_to_check = 3;

    // Use a fixed seed behavior if possible, or just random
    let mut rng = rand::thread_rng();
    let wave: Vec<f32> = (0..num_samples)
        .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
        .collect();

    let mut opts = FbankOptions::default();
    opts.frame_opts.dither = 0.0;
    opts.frame_opts.window_type = "hann".to_string();
    opts.mel_opts.num_bins = 23;
    opts.use_energy = false;
    opts.raw_energy = false;

    // Offline Compute
    let mut comp = FbankComputer::new(opts.clone()).unwrap();
    let win = Window::new(&opts.frame_opts);
    let padded = opts.frame_opts.padded_window_size();

    let mut offline = vec![vec![0.0; 23]; frames_to_check];
    let mut window_buf = vec![0.0; padded];

    for frame in 0..frames_to_check {
        let raw_log_energy = extract_window(
            0,
            &wave,
            frame,
            &opts.frame_opts,
            win.as_ref(),
            &mut window_buf,
        )
        .unwrap();
        comp.compute(raw_log_energy, 1.0, &mut window_buf, &mut offline[frame]);
    }

    // Online Compute
    let comp_online = FbankComputer::new(opts.clone()).unwrap();
    let mut online = OnlineFeature::new(FeatureComputer::Fbank(comp_online));
    online.accept_waveform(sample_rate, &wave);
    online.input_finished();

    assert!(online.num_frames_ready() >= frames_to_check);

    let _tol = 1e-3;
    for frame in 0..frames_to_check {
        let on = online.get_frame(frame).unwrap();
        for i in 0..opts.mel_opts.num_bins {
            let diff = (on[i] - offline[frame][i]).abs();
            assert!(
                diff < 5e-3,
                "Mismatch frame {} bin {}: online {} offline {} diff {}",
                frame,
                i,
                on[i],
                offline[frame][i],
                diff
            );
        }
    }
}

#[test]
fn test_stft_istft() {
    let n = 640;
    let mut wave = vec![0.0; n];
    for i in 0..n {
        wave[i] = (2.0 * PI * 440.0 * (i as f32 / 16000.0)).sin();
    }

    let stft_cfg = StftOptions::default();
    let res = stft_compute(&stft_cfg, &wave).expect("STFT failed");

    let mut istft_cfg = IstftOptions::default();
    istft_cfg.n_fft = stft_cfg.n_fft;
    istft_cfg.hop_length = stft_cfg.hop_length;
    istft_cfg.win_length = stft_cfg.win_length;
    istft_cfg.center = stft_cfg.center;
    istft_cfg.normalized = stft_cfg.normalized;

    let recon = istft_compute(&istft_cfg, &res).expect("ISTFT failed");

    assert_eq!(recon.len(), n);

    let mut max_err = 0.0f32;
    for i in 0..n {
        let err = (recon[i] - wave[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    println!("Max reconstruction error: {}", max_err);
    assert!(max_err < 1e-2);
}

#[test]
fn test_online() {
    let mut fopts = FbankOptions::default();
    fopts.frame_opts.dither = 0.0;

    let fbank = FbankComputer::new(fopts.clone()).unwrap();
    let mut feat = OnlineFeature::new(FeatureComputer::Fbank(fbank));

    let n = 3200;
    let freq = 1000.0;
    let samp_freq = fopts.frame_opts.samp_freq;
    let mut wave = vec![0.0; n];
    for i in 0..n {
        wave[i] = (2.0 * PI * freq * (i as f32 / samp_freq)).sin();
    }

    feat.accept_waveform(samp_freq, &wave);
    feat.input_finished();

    let ready = feat.num_frames_ready();
    assert!(ready > 0, "No frames ready");

    let frame = feat.get_frame(0).expect("Failed to get frame 0");
    for x in frame {
        assert!(x.is_finite());
    }
}

#[test]
fn test_whisper() {
    let opts = WhisperOptions::default();
    let mut comp = WhisperComputer::new(opts.clone()).expect("Failed");

    let n = opts.frame_opts.window_size();
    let mut wave = vec![0.0; n];
    // Generate sine
    for i in 0..n {
        wave[i] = (2.0 * PI * 300.0 * i as f32 / 16000.0).sin();
    }

    // Whisper needs padded buffer for FFT usually, check internal implementation.
    // In src/whisper.rs: `rfft.compute(signal_frame)`. rfft requires size=n_fft.
    // The C test calls it with `n` (window size), but wait, the C `knf_whisper_compute`
    // takes `signal_frame` which is presumably windowed?
    // In C test: `float *wave = calloc(n)...; knf_whisper_compute(..., wave, ...)`
    // The C implementation of `knf_whisper_compute` does:
    // `int32_t n_fft = knf_window_size...` -> `knf_rfft_compute(..., signal_frame)`
    // Wait, `knf_window_size` returns 400. `knf_rfft_create` uses `knf_window_size` (400) too.
    // So buffer size is 400.

    // In Rust, we need to make sure the buffer passed to compute matches rfft size.
    let mut feat = vec![0.0; opts.dim];

    // In strict C translation, we modify wave in place.
    comp.compute(0.0, 1.0, &mut wave, &mut feat);

    for (i, val) in feat.iter().enumerate() {
        assert!(val.is_finite(), "Whisper bin {} is not finite", i);
    }
}

#[test]

fn test_mfcc() {
    let mut opts = MfccOptions::default();
    opts.frame_opts.dither = 0.0;
    opts.frame_opts.preemph_coeff = 0.0;

    let mut comp = MfccComputer::new(opts.clone()).expect("Failed to create MfccComputer");

    let padded = opts.frame_opts.padded_window_size();
    let mut wave = vec![0.0; padded];
    for i in 0..padded {
        wave[i] = (2.0 * PI * 220.0 * (i as f32 / opts.frame_opts.samp_freq)).cos();
    }

    let win = Window::new(&opts.frame_opts);
    let mut window_buf = wave.clone();
    let raw_log_energy =
        extract_window(0, &wave, 0, &opts.frame_opts, win.as_ref(), &mut window_buf).unwrap();

    let mut feat = vec![0.0; comp.dim()];
    comp.compute(raw_log_energy, 1.0, &mut window_buf, &mut feat);

    for (i, val) in feat.iter().enumerate() {
        assert!(val.is_finite(), "MFCC bin {} is not finite", i);
    }
}

#[test]
fn test_fbank() {
    let mut opts = FbankOptions::default();
    opts.frame_opts.dither = 0.0;
    opts.frame_opts.preemph_coeff = 0.0;

    let mut comp = FbankComputer::new(opts.clone()).expect("Failed to create FbankComputer");

    let padded = opts.frame_opts.padded_window_size();
    let mut wave = vec![0.0; padded];
    for i in 0..padded {
        wave[i] = (2.0 * PI * 440.0 * (i as f32 / opts.frame_opts.samp_freq)).sin();
    }

    // Manual windowing check for energy
    let win = Window::new(&opts.frame_opts);
    let mut window_buf = wave.clone();
    let raw_log_energy =
        extract_window(0, &wave, 0, &opts.frame_opts, win.as_ref(), &mut window_buf).unwrap();

    let mut feat = vec![0.0; comp.dim()];
    comp.compute(raw_log_energy, 1.0, &mut window_buf, &mut feat);

    for (i, val) in feat.iter().enumerate() {
        assert!(val.is_finite(), "Feature bin {} is not finite", i);
    }
}

#[test]
fn test_mel_banks() {
    let mut fopts = FrameOptions::default();
    fopts.frame_length_ms = 25.0;
    fopts.frame_shift_ms = 10.0;

    let mut mopts = MelOptions::default();
    mopts.num_bins = 10;

    let banks = MelBanks::new(&mopts, &fopts, 1.0).expect("Failed to create mel banks");
    let cols = banks.num_fft_bins;

    // Create dummy FFT power spectrum
    let fft: Vec<f32> = (0..=cols).map(|i| i as f32).collect(); // +1 for N/2+1

    let mut out = vec![0.0; mopts.num_bins];
    banks.compute(&fft, &mut out);

    for (i, val) in out.iter().enumerate() {
        assert!(val.is_finite(), "Mel bin {} is not finite", i);
    }
}

#[test]
fn test_feature_window() {
    let opts = FrameOptions::default();
    assert_eq!(opts.window_size(), 400);
    assert_eq!(opts.padded_window_size(), 512);

    let window = Window::new(&opts).expect("Window creation failed");
    assert_eq!(window.data.len(), 400);

    let mut sample = vec![1.0; 400];
    window.apply(&mut sample);
    assert!(sample[0] <= 1.0 && sample[0] >= 0.0);

    let wave: Vec<f32> = (0..512).map(|i| i as f32).collect();
    let mut window_buf = vec![0.0; 512];

    let res = extract_window(0, &wave, 0, &opts, None, &mut window_buf);
    assert!(res.is_ok());
}

#[test]
fn test_rfft() {
    let mut signal = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let original = signal.clone();
    let n = signal.len();

    let mut fft = Rfft::new(n, false);
    fft.compute(&mut signal);

    let mut ifft = Rfft::new(n, true);
    ifft.compute(&mut signal);

    // FFTW inverse (and realfft) is unnormalized; expect n * original.
    for i in 0..n {
        let expected = original[i] * n as f32;
        let diff = (signal[i] - expected).abs();
        assert!(
            diff < 1e-3,
            "Mismatch at {}: got {}, expected {}",
            i,
            signal[i],
            expected
        );
    }
}
