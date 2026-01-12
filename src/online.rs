use crate::fbank::FbankComputer;
use crate::mfcc::MfccComputer;
use crate::window::{extract_window, first_sample_of_frame, num_frames, FrameOptions, Window};

pub enum FeatureComputer {
    Fbank(FbankComputer),
    Mfcc(MfccComputer),
}

impl FeatureComputer {
    pub fn frame_opts(&self) -> &FrameOptions {
        match self {
            Self::Fbank(c) => &c.opts.frame_opts,
            Self::Mfcc(c) => &c.opts.frame_opts,
        }
    }

    pub fn dim(&self) -> usize {
        match self {
            Self::Fbank(c) => c.dim(),
            Self::Mfcc(c) => c.dim(),
        }
    }

    pub fn compute(
        &mut self,
        raw_log_energy: f32,
        vtln_warp: f32,
        window: &mut [f32],
        feature: &mut [f32],
    ) {
        match self {
            Self::Fbank(c) => c.compute(raw_log_energy, vtln_warp, window, feature),
            Self::Mfcc(c) => c.compute(raw_log_energy, vtln_warp, window, feature),
        }
    }

    pub fn need_raw_energy(&self) -> bool {
        match self {
            Self::Fbank(c) => c.opts.use_energy && c.opts.raw_energy,
            Self::Mfcc(c) => c.opts.use_energy && c.opts.raw_energy,
        }
    }
}

pub struct OnlineFeature {
    computer: FeatureComputer,
    window_function: Option<Window>,
    waveform: Vec<f32>,
    waveform_offset: usize,
    input_finished: bool,
    pub features: Vec<Vec<f32>>,
}

impl OnlineFeature {
    pub fn new(computer: FeatureComputer) -> Self {
        let opts = computer.frame_opts();
        let window_function = Window::new(opts);
        Self {
            computer,
            window_function,
            waveform: Vec::new(),
            waveform_offset: 0,
            input_finished: false,
            features: Vec::new(),
        }
    }

    pub fn accept_waveform(&mut self, sampling_rate: f32, waveform: &[f32]) {
        let opts = self.computer.frame_opts();
        if (sampling_rate - opts.samp_freq).abs() > 1.0 {
            panic!(
                "Sampling rate mismatch: expected {}, got {}",
                opts.samp_freq, sampling_rate
            );
        }
        self.waveform.extend_from_slice(waveform);
        self.compute_new();
    }

    pub fn input_finished(&mut self) {
        self.input_finished = true;
        self.compute_new();
    }

    pub fn num_frames_ready(&self) -> usize {
        self.features.len()
    }

    pub fn get_frame(&self, frame: usize) -> Option<&[f32]> {
        self.features.get(frame).map(|v| v.as_slice())
    }

    fn compute_new(&mut self) {
        let opts = self.computer.frame_opts().clone(); // clone to avoid borrow conflict
        let total_samples = self.waveform_offset + self.waveform.len();
        let prev_frames = self.features.len();
        let new_frames = num_frames(total_samples, &opts, self.input_finished);

        if new_frames <= prev_frames {
            return;
        }

        let padded_size = opts.padded_window_size();
        let mut window_buf = vec![0.0; padded_size];
        let dim = self.computer.dim();

        for frame in prev_frames..new_frames {
            let raw_log_energy = extract_window(
                self.waveform_offset,
                &self.waveform,
                frame,
                &opts,
                self.window_function.as_ref(),
                &mut window_buf,
            )
            .expect("Failed to extract window");

            // If we don't need raw energy, pass 0.0 (it will be recomputed inside computer if use_energy is true)
            // But wait, the extract_window returns log_energy computed on the extracted frame BEFORE windowing.
            // This is exactly what "raw_log_energy" implies.

            let mut feature_vec = vec![0.0; dim];
            self.computer
                .compute(raw_log_energy, 1.0, &mut window_buf, &mut feature_vec);
            self.features.push(feature_vec);
        }

        // Garbage collect waveform
        let first_sample_next = first_sample_of_frame(new_frames, &opts);
        let discard = first_sample_next - self.waveform_offset as isize;

        if discard > 0 {
            let discard = discard as usize;
            if discard <= self.waveform.len() {
                self.waveform.drain(0..discard);
                self.waveform_offset += discard;
            }
        }
    }
}
