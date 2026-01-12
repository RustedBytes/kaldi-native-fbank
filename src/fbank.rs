use crate::mel::{MelBanks, MelOptions};
use crate::rfft::Rfft;
use crate::utils::{compute_power_spectrum_inplace, inner_product, log_energy};
use crate::window::FrameOptions;

#[derive(Clone, Debug)]
pub struct FbankOptions {
    pub frame_opts: FrameOptions,
    pub mel_opts: MelOptions,
    pub use_energy: bool,
    pub raw_energy: bool,
    pub htk_compat: bool,
    pub energy_floor: f32,
    pub use_log_fbank: bool,
    pub use_power: bool,
}

impl Default for FbankOptions {
    fn default() -> Self {
        Self {
            frame_opts: FrameOptions::default(),
            mel_opts: MelOptions::default(),
            use_energy: true,
            raw_energy: true,
            htk_compat: false,
            energy_floor: 0.0,
            use_log_fbank: true,
            use_power: true,
        }
    }
}

pub struct FbankComputer {
    pub opts: FbankOptions,
    rfft: Rfft,
    mel_banks: MelBanks,
    log_energy_floor: f32,
}

impl FbankComputer {
    pub fn new(opts: FbankOptions) -> Result<Self, String> {
        let n_fft = opts.frame_opts.padded_window_size();
        let rfft = Rfft::new(n_fft, false);
        let mel_banks = MelBanks::new(&opts.mel_opts, &opts.frame_opts, 1.0)?;

        let log_energy_floor = if opts.energy_floor > 0.0 {
            opts.energy_floor.ln()
        } else {
            -1e10
        };

        Ok(Self {
            opts,
            rfft,
            mel_banks,
            log_energy_floor,
        })
    }

    pub fn dim(&self) -> usize {
        self.opts.mel_opts.num_bins
            + if self.opts.use_energy && !self.opts.htk_compat {
                1
            } else {
                0
            }
    }

    pub fn compute(
        &mut self,
        mut signal_raw_log_energy: f32,
        _vtln_warp: f32,
        signal_frame: &mut [f32],
        feature: &mut [f32],
    ) {
        // vtln_warp handled in mel creation usually, but dynamic warp would require regenerating banks or more complex logic.
        // This port assumes static banks for now as per the simplified C code structure for `mel_banks_create` call in `new`.

        // 1. Calculate energy if needed and not raw
        if self.opts.use_energy && !self.opts.raw_energy {
            let energy = inner_product(signal_frame, signal_frame);
            signal_raw_log_energy = log_energy(energy);
        }

        // 2. FFT
        self.rfft.compute(signal_frame);

        // 3. Power Spectrum
        // signal_frame now contains complex FFT coefficients packed.
        compute_power_spectrum_inplace(signal_frame);

        // 4. Magnitude if not power
        if !self.opts.use_power {
            for x in signal_frame.iter_mut() {
                *x = x.sqrt();
            }
        }

        // 5. Mel integration
        let mel_offset = if self.opts.use_energy && !self.opts.htk_compat {
            1
        } else {
            0
        };
        let fft_bins = self.mel_banks.num_fft_bins + 1; // use only the computed power bins (N/2 + 1)
        self.mel_banks
            .compute(&signal_frame[..fft_bins], &mut feature[mel_offset..]);

        // 6. Log
        if self.opts.use_log_fbank {
            for x in feature[mel_offset..].iter_mut() {
                *x = log_energy(*x);
            }
        }

        // 7. Energy appending
        if self.opts.use_energy {
            if self.opts.energy_floor > 0.0 && signal_raw_log_energy < self.log_energy_floor {
                signal_raw_log_energy = self.log_energy_floor;
            }
            let energy_index = if self.opts.htk_compat {
                self.opts.mel_opts.num_bins
            } else {
                0
            };
            feature[energy_index] = signal_raw_log_energy;
        }
    }
}
