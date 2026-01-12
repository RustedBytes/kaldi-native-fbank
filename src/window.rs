use crate::utils::TWO_PI;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct FrameOptions {
    pub samp_freq: f32,
    pub frame_shift_ms: f32,
    pub frame_length_ms: f32,
    pub dither: f32,
    pub preemph_coeff: f32,
    pub remove_dc_offset: bool,
    pub window_type: String,
    pub round_to_power_of_two: bool,
    pub blackman_coeff: f32,
    pub snip_edges: bool,
}

impl Default for FrameOptions {
    fn default() -> Self {
        Self {
            samp_freq: 16000.0,
            frame_shift_ms: 10.0,
            frame_length_ms: 25.0,
            dither: 0.00003,
            preemph_coeff: 0.97,
            remove_dc_offset: true,
            window_type: "povey".to_string(),
            round_to_power_of_two: true,
            blackman_coeff: 0.42,
            snip_edges: true,
        }
    }
}

impl FrameOptions {
    pub fn window_shift(&self) -> usize {
        (self.samp_freq * 0.001 * self.frame_shift_ms) as usize
    }

    pub fn window_size(&self) -> usize {
        (self.samp_freq * 0.001 * self.frame_length_ms) as usize
    }

    pub fn padded_window_size(&self) -> usize {
        let size = self.window_size();
        if self.round_to_power_of_two {
            size.next_power_of_two()
        } else {
            size
        }
    }
}

pub struct Window {
    pub data: Vec<f32>,
}

impl Window {
    pub fn new(opts: &FrameOptions) -> Option<Self> {
        let size = opts.window_size();
        if size == 0 {
            return None;
        }
        let mut data = vec![0.0; size];
        let a = TWO_PI / (size as f32 - 1.0);
        let a_hann = TWO_PI / (size as f32); // used for hann specifically

        for i in 0..size {
            let x = i as f32;
            data[i] = match opts.window_type.as_str() {
                "hanning" => 0.5 - 0.5 * (a * x).cos(),
                "sine" => (0.5 * a * x).sin(),
                "hamming" => 0.54 - 0.46 * (a * x).cos(),
                "hann" => 0.50 - 0.50 * (a_hann * x).cos(),
                "povey" => (0.5 - 0.5 * (a * x).cos()).powf(0.85),
                "rectangular" => 1.0,
                "blackman" => {
                    opts.blackman_coeff - 0.5 * (a * x).cos()
                        + (0.5 - opts.blackman_coeff) * (2.0 * a * x).cos()
                }
                _ => return None,
            };
        }
        Some(Self { data })
    }

    pub fn apply(&self, wave: &mut [f32]) {
        for (w, s) in wave.iter_mut().zip(self.data.iter()) {
            *w *= s;
        }
    }
}

pub fn num_frames(num_samples: usize, opts: &FrameOptions, flush: bool) -> usize {
    let frame_shift = opts.window_shift();
    let frame_length = opts.window_size();

    if opts.snip_edges {
        if num_samples < frame_length {
            return 0;
        }
        return 1 + (num_samples - frame_length) / frame_shift;
    }

    let num_frames = (num_samples as f32 + (frame_shift as f32 / 2.0)) / frame_shift as f32;
    let mut num_frames = num_frames as usize;

    if flush {
        return num_frames;
    }

    // Adjustment for center/non-snip
    // Logic mirrored from C:
    // int64_t end_sample = knf_first_sample_of_frame(num_frames - 1, opts) + frame_length;
    // while (num_frames > 0 && end_sample > num_samples) ...

    // We'll calculate end sample iteratively just to be safe with the translation
    while num_frames > 0 {
        let first_sample = first_sample_of_frame(num_frames - 1, opts);
        let end_sample = first_sample + frame_length as isize;
        if end_sample > num_samples as isize {
            num_frames -= 1;
        } else {
            break;
        }
    }
    num_frames
}

pub fn first_sample_of_frame(frame: usize, opts: &FrameOptions) -> isize {
    let frame_shift = opts.window_shift() as isize;
    if opts.snip_edges {
        return (frame as isize) * frame_shift;
    }
    let midpoint = frame_shift * (frame as isize) + frame_shift / 2;
    midpoint - (opts.window_size() as isize) / 2
}

pub fn extract_window(
    sample_offset: usize,
    wave: &[f32],
    frame_index: usize,
    opts: &FrameOptions,
    window_function: Option<&Window>,
    window_out: &mut [f32],
) -> Result<f32, ()> {
    let frame_length = opts.window_size();
    let num_samples = sample_offset + wave.len();
    let start_sample = first_sample_of_frame(frame_index, opts);
    let end_sample = start_sample + frame_length as isize;

    if opts.snip_edges {
        if start_sample < sample_offset as isize || end_sample > num_samples as isize {
            return Err(());
        }
    } else if !(sample_offset == 0 || start_sample >= sample_offset as isize) {
        return Err(());
    }

    // Zero out buffer
    window_out.fill(0.0);

    let wave_start = start_sample - sample_offset as isize;
    // let wave_end = wave_start + frame_length as isize;

    // Copy wave data
    for s in 0..frame_length {
        let s_in_wave = s as isize + wave_start;
        let mut idx = s_in_wave;

        // Reflective padding if needed
        while idx < 0 || idx >= wave.len() as isize {
            if idx < 0 {
                idx = -idx - 1;
            } else {
                idx = 2 * (wave.len() as isize) - 1 - idx;
            }
        }

        if idx >= 0 && (idx as usize) < wave.len() {
            window_out[s] = wave[idx as usize];
        }
    }

    // Dither
    if opts.dither != 0.0 {
        let mut rng = rand::thread_rng();
        for x in window_out.iter_mut().take(frame_length) {
            *x += opts.dither * (rng.gen::<f32>() - 0.5);
        }
    }

    // Remove DC
    if opts.remove_dc_offset {
        let sum: f32 = window_out.iter().take(frame_length).sum();
        let mean = sum / frame_length as f32;
        for x in window_out.iter_mut().take(frame_length) {
            *x -= mean;
        }
    }

    // Preemphasis
    if opts.preemph_coeff != 0.0 {
        let mut last = window_out[0];
        for i in (1..frame_length).rev() {
            let prev = window_out[i - 1];
            window_out[i] -= opts.preemph_coeff * prev;
            last = prev;
        }
        window_out[0] -= opts.preemph_coeff * last;
    }

    // Calculate raw log energy before windowing
    let energy: f32 = window_out.iter().take(frame_length).map(|x| x * x).sum();
    let log_energy = if energy < 1e-10 {
        1e-10f32.ln()
    } else {
        energy.ln()
    };

    // Apply window
    if let Some(win) = window_function {
        win.apply(&mut window_out[0..frame_length]);
    }

    Ok(log_energy)
}
