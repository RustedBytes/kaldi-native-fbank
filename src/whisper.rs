use crate::mel::{MelBanks, MelOptions};
use crate::rfft::Rfft;
use crate::utils::compute_power_spectrum_inplace;
use crate::window::FrameOptions;

#[derive(Clone, Debug)]
pub struct WhisperOptions {
    pub frame_opts: FrameOptions,
    pub dim: usize,
}

impl Default for WhisperOptions {
    fn default() -> Self {
        let mut f = FrameOptions::default();
        f.samp_freq = 16000.0;
        f.frame_shift_ms = 10.0;
        f.frame_length_ms = 25.0;
        f.dither = 0.0;
        f.preemph_coeff = 0.0;
        f.remove_dc_offset = false;
        f.window_type = "hann".to_string();
        f.round_to_power_of_two = false;
        f.snip_edges = false;

        Self {
            frame_opts: f,
            dim: 80,
        }
    }
}

pub struct WhisperComputer {
    pub opts: WhisperOptions,
    rfft: Rfft,
    mel_banks: MelBanks,
}

impl WhisperComputer {
    pub fn new(opts: WhisperOptions) -> Result<Self, String> {
        let mut mel_opts = MelOptions::default();
        mel_opts.num_bins = opts.dim;
        mel_opts.low_freq = 0.0;
        mel_opts.is_librosa = true;
        mel_opts.use_slaney_mel_scale = true;
        mel_opts.norm = "slaney".to_string();

        let n_fft = opts.frame_opts.padded_window_size(); // likely 400 since round=false
        let rfft = Rfft::new(n_fft, false);

        let mel_banks = MelBanks::new(&mel_opts, &opts.frame_opts, 1.0)?;

        Ok(Self {
            opts,
            rfft,
            mel_banks,
        })
    }

    pub fn dim(&self) -> usize {
        self.opts.dim
    }

    pub fn compute(
        &mut self,
        _raw_log_energy: f32,
        _vtln_warp: f32,
        signal_frame: &mut [f32],
        feature: &mut [f32],
    ) {
        // Whisper doesn't use raw_log_energy or vtln inside the compute loop usually
        self.rfft.compute(signal_frame);
        compute_power_spectrum_inplace(signal_frame);
        let fft_bins = self.mel_banks.num_fft_bins + 1; // N/2 + 1 power spectrum bins
        self.mel_banks.compute(&signal_frame[..fft_bins], feature);

        // Note: Whisper usually logs the features afterwards (log10 or natural log depending on implementation step),
        // but the C code `knf_whisper_compute` -> `knf_mel_compute` -> done.
        // Checking C `whisper-feature.c`:
        // It calls `knf_rfft_compute`, `knf_compute_power_spectrum`, `knf_mel_compute`.
        // It does NOT apply log in the C implementation provided in the prompt.
        // So we leave it linear here to match C.
    }
}
