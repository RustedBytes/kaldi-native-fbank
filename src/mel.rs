use crate::window::FrameOptions;

#[derive(Clone, Debug)]
pub struct MelOptions {
    pub num_bins: usize,
    pub low_freq: f32,
    pub high_freq: f32,
    pub vtln_low: f32,
    pub vtln_high: f32,
    pub htk_mode: bool,
    pub is_librosa: bool,
    pub use_slaney_mel_scale: bool,
    pub norm: String,
    pub floor_to_int_bin: bool,
    pub debug_mel: bool,
}

impl Default for MelOptions {
    fn default() -> Self {
        Self {
            num_bins: 25,
            low_freq: 20.0,
            high_freq: 0.0,
            vtln_low: 100.0,
            vtln_high: -500.0,
            htk_mode: false,
            is_librosa: false,
            use_slaney_mel_scale: true,
            norm: "slaney".to_string(),
            floor_to_int_bin: false,
            debug_mel: false,
        }
    }
}

pub struct MelBanks {
    pub num_bins: usize,
    pub num_fft_bins: usize,
    // Flattened weights: [bin_idx * num_fft_bins + fft_bin_idx]
    pub weights: Vec<f32>,
}

impl MelBanks {
    pub fn new(
        opts: &MelOptions,
        frame_opts: &FrameOptions,
        vtln_warp: f32,
    ) -> Result<Self, String> {
        let window_length_padded = frame_opts.padded_window_size();
        let num_fft_bins = window_length_padded / 2;
        let sample_freq = frame_opts.samp_freq;
        let nyquist = 0.5 * sample_freq;

        let high_freq = if opts.high_freq > 0.0 {
            opts.high_freq
        } else {
            nyquist + opts.high_freq
        };

        if opts.low_freq < 0.0
            || opts.low_freq >= nyquist
            || high_freq <= 0.0
            || high_freq > nyquist
            || high_freq <= opts.low_freq
        {
            return Err("Invalid frequency range for Mel banks".to_string());
        }

        let fft_bin_width = sample_freq / window_length_padded as f32;
        let mel_low = Self::mel_scale(opts.low_freq);
        let mel_high = Self::mel_scale(high_freq);
        let mel_delta = (mel_high - mel_low) / (opts.num_bins as f32 + 1.0);

        // VTLN setup (simplified for brevity, matching C logic)
        let vtln_low = opts.vtln_low;
        let vtln_high = if opts.vtln_high < 0.0 {
            opts.vtln_high + nyquist
        } else {
            opts.vtln_high
        };

        let mut weights = vec![0.0; opts.num_bins * num_fft_bins];

        for bin in 0..opts.num_bins {
            let mut left_mel = mel_low + bin as f32 * mel_delta;
            let mut center_mel = mel_low + (bin + 1) as f32 * mel_delta;
            let mut right_mel = mel_low + (bin + 2) as f32 * mel_delta;

            if (vtln_warp - 1.0).abs() > 1e-5 {
                left_mel = Self::vtln_warp_mel(
                    vtln_low,
                    vtln_high,
                    opts.low_freq,
                    high_freq,
                    vtln_warp,
                    left_mel,
                );
                center_mel = Self::vtln_warp_mel(
                    vtln_low,
                    vtln_high,
                    opts.low_freq,
                    high_freq,
                    vtln_warp,
                    center_mel,
                );
                right_mel = Self::vtln_warp_mel(
                    vtln_low,
                    vtln_high,
                    opts.low_freq,
                    high_freq,
                    vtln_warp,
                    right_mel,
                );
            }

            for i in 0..num_fft_bins {
                let freq = fft_bin_width * i as f32;
                let mel = Self::mel_scale(freq);
                let weight = if mel > left_mel && mel < right_mel {
                    if mel <= center_mel {
                        (mel - left_mel) / (center_mel - left_mel)
                    } else {
                        (right_mel - mel) / (right_mel - center_mel)
                    }
                } else {
                    0.0
                };

                if weight > 0.0 {
                    weights[bin * num_fft_bins + i] = weight;
                }
            }
        }

        Ok(Self {
            num_bins: opts.num_bins,
            num_fft_bins,
            weights,
        })
    }

    pub fn compute(&self, fft_energies: &[f32], mel_energies_out: &mut [f32]) {
        assert_eq!(fft_energies.len(), self.num_fft_bins + 1); // +1 because FFT result usually has DC and Nyquist packed
        assert_eq!(mel_energies_out.len(), self.num_bins);

        // The FFT energies passed here are expected to be the power spectrum (N/2 + 1) elements.
        // But `weights` was built for `num_fft_bins` (N/2).
        // Typically, we ignore the Nyquist bin or handle it if `weights` supports it.
        // The C code loop `for (int32_t c = 0; c < cols; ++c)` iterates `num_fft_bins`.

        for r in 0..self.num_bins {
            let mut sum = 0.0;
            let row_offset = r * self.num_fft_bins;
            for c in 0..self.num_fft_bins {
                sum += self.weights[row_offset + c] * fft_energies[c];
            }
            mel_energies_out[r] = sum;
        }
    }

    fn mel_scale(freq: f32) -> f32 {
        1127.0 * (1.0 + freq / 700.0).ln()
    }

    fn inverse_mel_scale(mel: f32) -> f32 {
        700.0 * ((mel / 1127.0).exp() - 1.0)
    }

    fn vtln_warp_mel(
        vtln_low: f32,
        vtln_high: f32,
        low_freq: f32,
        high_freq: f32,
        vtln_warp: f32,
        mel: f32,
    ) -> f32 {
        let freq = Self::inverse_mel_scale(mel);
        let warped =
            Self::vtln_warp_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp, freq);
        Self::mel_scale(warped)
    }

    fn vtln_warp_freq(
        vtln_low: f32,
        vtln_high: f32,
        low_freq: f32,
        high_freq: f32,
        vtln_warp: f32,
        freq: f32,
    ) -> f32 {
        if freq < low_freq || freq > high_freq {
            return freq;
        }
        let l = vtln_low * 1.0f32.max(vtln_warp);
        let h = vtln_high * 1.0f32.min(vtln_warp);
        let scale = 1.0 / vtln_warp;
        let fl = scale * l;
        let fh = scale * h;

        if freq < l {
            let scale_left = (fl - low_freq) / (l - low_freq);
            low_freq + scale_left * (freq - low_freq)
        } else if freq < h {
            scale * freq
        } else {
            let scale_right = (high_freq - fh) / (high_freq - h);
            high_freq + scale_right * (freq - high_freq)
        }
    }
}
