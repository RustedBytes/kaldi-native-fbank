use crate::rfft::Rfft;
use crate::window::{FrameOptions, Window};

#[derive(Clone, Debug)]
pub struct StftOptions {
    pub frame_opts: FrameOptions,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub center: bool,
    pub pad_mode: String, // "reflect", "replicate", "constant"
    pub normalized: bool,
}

impl Default for StftOptions {
    fn default() -> Self {
        let mut f = FrameOptions::default();
        f.window_type = "povey".to_string();
        Self {
            frame_opts: f, // Used mainly for window generation helper
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            center: true,
            pad_mode: "reflect".to_string(),
            normalized: false,
        }
    }
}

pub struct StftResult {
    pub real: Vec<f32>, // [frame_idx * bins + bin_idx]
    pub imag: Vec<f32>,
    pub num_frames: usize,
    pub n_fft: usize,
}

pub fn stft_compute(opts: &StftOptions, waveform: &[f32]) -> Result<StftResult, String> {
    if opts.n_fft == 0 || opts.hop_length == 0 || opts.win_length == 0 {
        return Err("Invalid STFT parameters".to_string());
    }

    // Prepare window
    let mut win_opts = opts.frame_opts.clone();
    win_opts.frame_length_ms = 0.0; // Not used directly if we override size
                                    // We manually construct window of win_length
                                    // The Window struct uses frame_opts.window_size(). Let's force it.
                                    // Hack: adjust samp_freq/length to match win_length samples?
                                    // Easier: Reuse Window::new but we need to ensure size is win_length.
                                    // FrameOptions calculates size from ms. Let's make a temporary options struct that yields win_length.
    win_opts.samp_freq = 1000.0;
    win_opts.frame_length_ms = opts.win_length as f32; // 1000 * 0.001 * X = X
    let window = Window::new(&win_opts).ok_or("Failed to create window")?;

    // Padding
    let pad = if opts.center { opts.n_fft / 2 } else { 0 };
    let mut padded_wave = Vec::with_capacity(waveform.len() + 2 * pad);

    if opts.center {
        match opts.pad_mode.as_str() {
            "reflect" => {
                // Left pad
                for i in 0..pad {
                    let idx = pad - i;
                    let src = if idx >= waveform.len() {
                        waveform.len().saturating_sub(1)
                    } else {
                        idx
                    };
                    padded_wave.push(waveform[src]);
                }
                padded_wave.extend_from_slice(waveform);
                // Right pad
                for i in 0..pad {
                    let idx = waveform.len().saturating_sub(2).saturating_sub(i);
                    // Handle edge case of short waveform
                    let src = if idx >= waveform.len() { 0 } else { idx };
                    padded_wave.push(waveform[src]);
                }
            }
            "replicate" => {
                for _ in 0..pad {
                    padded_wave.push(waveform.first().copied().unwrap_or(0.0));
                }
                padded_wave.extend_from_slice(waveform);
                for _ in 0..pad {
                    padded_wave.push(waveform.last().copied().unwrap_or(0.0));
                }
            }
            _ => {
                // constant 0
                padded_wave.resize(pad, 0.0);
                padded_wave.extend_from_slice(waveform);
                padded_wave.resize(padded_wave.len() + pad, 0.0);
            }
        }
    } else {
        padded_wave.extend_from_slice(waveform);
    }

    let num_samples = padded_wave.len();
    if num_samples < opts.n_fft {
        return Ok(StftResult {
            real: vec![],
            imag: vec![],
            num_frames: 0,
            n_fft: opts.n_fft,
        });
    }

    let num_frames = 1 + (num_samples - opts.n_fft) / opts.hop_length;
    let bins = opts.n_fft / 2 + 1;

    let mut real = vec![0.0; num_frames * bins];
    let mut imag = vec![0.0; num_frames * bins];

    let mut rfft = Rfft::new(opts.n_fft, false);
    let mut frame_buf = vec![0.0; opts.n_fft];

    for i in 0..num_frames {
        let start = i * opts.hop_length;
        let end = start + opts.n_fft;
        frame_buf.copy_from_slice(&padded_wave[start..end]);

        // Apply window
        window.apply(&mut frame_buf[..opts.win_length.min(opts.n_fft)]);

        // FFT
        rfft.compute(&mut frame_buf);

        // Unpack RFFT result [Re0, ReN/2, Re1, Im1, ...]
        // Bin 0
        real[i * bins] = frame_buf[0];
        imag[i * bins] = 0.0;

        // Bin N/2
        if bins > 1 {
            // If n_fft is even, frame_buf[1] is Re(N/2).
            // If odd, packing is different, but assume even for standard audio STFT.
            real[i * bins + (bins - 1)] = frame_buf[1];
            imag[i * bins + (bins - 1)] = 0.0;
        }

        for k in 1..(bins - 1) {
            real[i * bins + k] = frame_buf[2 * k];
            imag[i * bins + k] = frame_buf[2 * k + 1];
        }
    }

    if opts.normalized {
        let scale = 1.0 / (opts.n_fft as f32).sqrt();
        for x in real.iter_mut() {
            *x *= scale;
        }
        for x in imag.iter_mut() {
            *x *= scale;
        }
    }

    Ok(StftResult {
        real,
        imag,
        num_frames,
        n_fft: opts.n_fft,
    })
}
