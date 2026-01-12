use std::f32;

pub const PI: f32 = f32::consts::PI;
pub const TWO_PI: f32 = 2.0 * PI;
pub const SQRT2: f32 = f32::consts::SQRT_2;

/// Computes the inner product (dot product) of two vectors.
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Computes the power spectrum from a complex FFT output.
///
/// `complex_fft` layout: [Re(0), Re(n/2), Re(1), Im(1), Re(2), Im(2), ...]
/// This matches the layout used in the C codebase after unpacking FFTW R2C.
///
/// Note: The specific packing in the Rust port depends on how we handle `realfft`.
/// `realfft` returns a `Complex<f32>` slice. We will adapt this function in `src/fbank.rs`
/// or `src/rfft.rs` specifically for the `rustfft` output format, but we keep a generic
/// logic here for reference if needed.
pub fn compute_power_spectrum_inplace(spectrum: &mut [f32]) {
    let dim = spectrum.len();
    let half_dim = dim / 2;

    // The C code assumes specific packing:
    // complex_fft[0] -> Energy at DC
    // complex_fft[1] -> Energy at Nyquist
    // complex_fft[2*i], complex_fft[2*i+1] -> Real, Imag at i

    let first = spectrum[0];
    let last = spectrum[1];
    let first_energy = first * first;
    let last_energy = last * last;

    for i in 1..half_dim {
        let real = spectrum[i * 2];
        let im = spectrum[i * 2 + 1];
        spectrum[i] = real * real + im * im;
    }

    spectrum[0] = first_energy;
    spectrum[half_dim] = last_energy;
}

pub fn log_energy(energy: f32) -> f32 {
    let v = if energy < 1e-20 { 1e-20 } else { energy };
    v.ln()
}
