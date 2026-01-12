// Reuse fbank options structure or components if preferred
use crate::mel::{MelBanks, MelOptions};
use crate::rfft::Rfft;
use crate::utils::{compute_power_spectrum_inplace, inner_product, log_energy, PI, SQRT2};
use crate::window::FrameOptions;

#[derive(Clone, Debug)]
pub struct MfccOptions {
    pub frame_opts: FrameOptions,
    pub mel_opts: MelOptions,
    pub num_ceps: usize,
    pub cepstral_lifter: f32,
    pub use_energy: bool,
    pub raw_energy: bool,
    pub htk_compat: bool,
    pub energy_floor: f32,
}

impl Default for MfccOptions {
    fn default() -> Self {
        Self {
            frame_opts: FrameOptions::default(),
            mel_opts: MelOptions::default(),
            num_ceps: 13,
            cepstral_lifter: 22.0,
            use_energy: true,
            raw_energy: true,
            htk_compat: false,
            energy_floor: 0.0,
        }
    }
}

pub struct MfccComputer {
    pub opts: MfccOptions,
    rfft: Rfft,
    mel_banks: MelBanks,
    mel_energies: Vec<f32>,
    dct_matrix: Vec<f32>, // flattened [num_ceps * num_bins]
    lifter_coeffs: Vec<f32>,
    log_energy_floor: f32,
}

impl MfccComputer {
    pub fn new(opts: MfccOptions) -> Result<Self, String> {
        let n_fft = opts.frame_opts.padded_window_size();
        let rfft = Rfft::new(n_fft, false);
        let mel_banks = MelBanks::new(&opts.mel_opts, &opts.frame_opts, 1.0)?;

        let mel_energies = vec![0.0; opts.mel_opts.num_bins];

        // Compute DCT Matrix
        let mut dct_matrix = vec![0.0; opts.num_ceps * opts.mel_opts.num_bins];
        let k_factor = (2.0 / opts.mel_opts.num_bins as f32).sqrt();
        let k0_factor = (1.0 / opts.mel_opts.num_bins as f32).sqrt();

        for i in 0..opts.num_ceps {
            for j in 0..opts.mel_opts.num_bins {
                let val = if i == 0 {
                    k0_factor
                } else {
                    k_factor
                        * (PI / opts.mel_opts.num_bins as f32 * (j as f32 + 0.5) * i as f32).cos()
                };
                dct_matrix[i * opts.mel_opts.num_bins + j] = val;
            }
        }

        // Compute Lifter coeffs
        let mut lifter_coeffs = vec![1.0; opts.num_ceps];
        if opts.cepstral_lifter != 0.0 {
            for i in 0..opts.num_ceps {
                lifter_coeffs[i] =
                    1.0 + 0.5 * opts.cepstral_lifter * (PI * i as f32 / opts.cepstral_lifter).sin();
            }
        }

        let log_energy_floor = if opts.energy_floor > 0.0 {
            opts.energy_floor.ln()
        } else {
            -1e10
        };

        Ok(Self {
            opts,
            rfft,
            mel_banks,
            mel_energies,
            dct_matrix,
            lifter_coeffs,
            log_energy_floor,
        })
    }

    pub fn dim(&self) -> usize {
        self.opts.num_ceps
    }

    pub fn compute(
        &mut self,
        mut signal_raw_log_energy: f32,
        _vtln_warp: f32,
        signal_frame: &mut [f32],
        feature: &mut [f32],
    ) {
        if self.opts.use_energy && !self.opts.raw_energy {
            let energy = inner_product(signal_frame, signal_frame);
            signal_raw_log_energy = log_energy(energy);
        }

        self.rfft.compute(signal_frame);
        compute_power_spectrum_inplace(signal_frame);

        let fft_bins = self.mel_banks.num_fft_bins + 1; // use only power spectrum bins
        self.mel_banks
            .compute(&signal_frame[..fft_bins], &mut self.mel_energies);

        // Log Mel
        for x in self.mel_energies.iter_mut() {
            *x = log_energy(*x);
        }

        // DCT
        for i in 0..self.opts.num_ceps {
            let row_offset = i * self.opts.mel_opts.num_bins;
            feature[i] = inner_product(
                &self.dct_matrix[row_offset..row_offset + self.opts.mel_opts.num_bins],
                &self.mel_energies,
            );
        }

        // Lifter
        if self.opts.cepstral_lifter != 0.0 {
            for (i, val) in feature.iter_mut().enumerate().take(self.opts.num_ceps) {
                *val *= self.lifter_coeffs[i];
            }
        }

        // Energy replace
        if self.opts.use_energy {
            if self.opts.energy_floor > 0.0 && signal_raw_log_energy < self.log_energy_floor {
                signal_raw_log_energy = self.log_energy_floor;
            }
            feature[0] = signal_raw_log_energy;
        }

        // HTK Compat
        if self.opts.htk_compat {
            let mut energy = feature[0];
            if !self.opts.use_energy {
                energy *= SQRT2; // Scale if not true energy
            }
            // Rotate: C0 becomes last (stored as energy usually in HTK)
            for i in 0..self.opts.num_ceps - 1 {
                feature[i] = feature[i + 1];
            }
            feature[self.opts.num_ceps - 1] = energy;
        }
    }
}
