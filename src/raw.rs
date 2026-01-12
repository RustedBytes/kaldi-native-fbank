use crate::window::FrameOptions;

#[derive(Clone, Debug)]
pub struct RawAudioOptions {
    pub frame_opts: FrameOptions,
}

impl Default for RawAudioOptions {
    fn default() -> Self {
        Self {
            frame_opts: FrameOptions::default(),
        }
    }
}

pub struct RawAudioComputer {
    pub opts: RawAudioOptions,
}

impl RawAudioComputer {
    pub fn new(opts: RawAudioOptions) -> Self {
        Self { opts }
    }
    pub fn dim(&self) -> usize {
        self.opts.frame_opts.padded_window_size()
    }
    pub fn compute(&mut self, _e: f32, _v: f32, signal: &mut [f32], feature: &mut [f32]) {
        let dim = self.dim();
        if feature.len() >= dim && signal.len() >= dim {
            feature[..dim].copy_from_slice(&signal[..dim]);
        }
    }
}
