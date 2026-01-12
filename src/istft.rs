use crate::rfft::Rfft;
use crate::stft::{StftOptions, StftResult};
use crate::window::{FrameOptions, Window};

pub struct IstftOptions {
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub window_type: String,
    pub center: bool,
    pub normalized: bool,
}

impl Default for IstftOptions {
    fn default() -> Self {
        Self {
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            window_type: "povey".to_string(),
            center: true,
            normalized: false,
        }
    }
}

impl From<&StftOptions> for IstftOptions {
    fn from(s: &StftOptions) -> Self {
        Self {
            n_fft: s.n_fft,
            hop_length: s.hop_length,
            win_length: s.win_length,
            window_type: s.frame_opts.window_type.clone(),
            center: s.center,
            normalized: s.normalized,
        }
    }
}

pub fn istft_compute(opts: &IstftOptions, stft: &StftResult) -> Result<Vec<f32>, String> {
    if stft.num_frames == 0 || opts.n_fft == 0 {
        return Ok(Vec::new());
    }

    let n_fft = opts.n_fft;
    let hop = opts.hop_length;
    let total_len = n_fft + (stft.num_frames - 1) * hop;

    let mut samples = vec![0.0; total_len];
    let mut denom = vec![0.0; total_len];

    // Prepare window
    let mut win_opts = FrameOptions::default();
    win_opts.window_type = opts.window_type.clone();
    win_opts.samp_freq = 1000.0;
    win_opts.frame_length_ms = opts.win_length as f32;
    // For ISTFT we usually use a Hanning or Povey window same as analysis
    let window = Window::new(&win_opts).ok_or("Failed to create window")?;

    let mut ifft = Rfft::new(n_fft, true);
    let mut frame_buf = vec![0.0; n_fft];

    let bins = n_fft / 2 + 1;
    let inv_n = 1.0 / n_fft as f32;
    let pre_scale = if opts.normalized {
        (n_fft as f32).sqrt()
    } else {
        1.0
    };

    for i in 0..stft.num_frames {
        // Pack into RFFT format [Re0, ReN/2, Re1, Im1...]
        let r_ptr = &stft.real[i * bins..(i + 1) * bins];
        let i_ptr = &stft.imag[i * bins..(i + 1) * bins];

        frame_buf[0] = r_ptr[0] * pre_scale;
        if n_fft % 2 == 0 {
            frame_buf[1] = r_ptr[bins - 1] * pre_scale;
        }

        for k in 1..(bins - 1) {
            // Up to bins-1 because bins-1 is N/2
            frame_buf[2 * k] = r_ptr[k] * pre_scale;
            frame_buf[2 * k + 1] = i_ptr[k] * pre_scale;
        }

        ifft.compute(&mut frame_buf);

        // realfft inverse is unnormalized, multiply by 1/N
        for x in frame_buf.iter_mut() {
            *x *= inv_n;
        }

        // Apply window
        window.apply(&mut frame_buf[..opts.win_length.min(n_fft)]);

        // Overlap Add
        let start = i * hop;
        for k in 0..n_fft {
            samples[start + k] += frame_buf[k];
        }

        // Accumulate Window Squares for normalization
        for k in 0..opts.win_length.min(n_fft) {
            denom[start + k] += window.data[k] * window.data[k];
        }
    }

    // Normalize by window overlap
    for i in 0..total_len {
        if denom[i] > 1e-10 {
            samples[i] /= denom[i];
        }
    }

    // Trim centering padding
    if opts.center {
        let cut = n_fft / 2;
        if total_len > 2 * cut {
            let trimmed = samples[cut..total_len - cut].to_vec();
            return Ok(trimmed);
        }
        return Ok(vec![]);
    }

    Ok(samples)
}
