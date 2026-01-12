use realfft::num_traits::Zero;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;

/// A wrapper around `realfft` to mimic the behavior of the C `knf_rfft`.
pub struct Rfft {
    n: usize,
    inverse: bool,
    r2c: Option<Arc<dyn RealToComplex<f32>>>,
    c2r: Option<Arc<dyn ComplexToReal<f32>>>,
    scratch: Vec<Complex<f32>>, // Scratch buffer for FFT computation
}

impl Rfft {
    pub fn new(n: usize, inverse: bool) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let (r2c, c2r) = if !inverse {
            (Some(planner.plan_fft_forward(n)), None)
        } else {
            (None, Some(planner.plan_fft_inverse(n)))
        };

        // Pre-allocate scratch space based on the planner requirements
        let scratch_len = if let Some(ref p) = r2c {
            p.get_scratch_len()
        } else {
            c2r.as_ref().unwrap().get_scratch_len()
        };

        Self {
            n,
            inverse,
            r2c,
            c2r,
            scratch: vec![Complex::zero(); scratch_len],
        }
    }

    /// Computes the RFFT or IRFFT.
    ///
    /// For Forward (Real->Complex):
    /// Input `data` is the real signal (length `n`).
    /// Output is written back into `data` mimicking the C-style packed format:
    /// [Re(0), Re(N/2), Re(1), Im(1), Re(2), Im(2), ...]
    ///
    /// For Inverse (Complex->Real):
    /// Input `data` is packed complex.
    /// Output is real signal.
    pub fn compute(&mut self, data: &mut [f32]) {
        if !self.inverse {
            self.compute_forward(data);
        } else {
            self.compute_inverse(data);
        }
    }

    fn compute_forward(&mut self, data: &mut [f32]) {
        let n = self.n;
        if data.len() < n {
            panic!("Data length {} too small for FFT size {}", data.len(), n);
        }

        // 1. Copy input to a temporary buffer because realfft process_dct modifies input
        // or requires specific types. realfft takes &[f32] input and &mut [Complex] output.
        let mut input_real = data[0..n].to_vec();

        // Output buffer for complex spectrum (N/2 + 1)
        let mut output_complex = vec![Complex::zero(); n / 2 + 1];

        // 2. Perform FFT
        self.r2c
            .as_ref()
            .unwrap()
            .process_with_scratch(&mut input_real, &mut output_complex, &mut self.scratch)
            .unwrap();

        // 3. Pack back into `data` to match the C implementation's expectation
        // Format: [Re(0), Re(N/2), Re(1), Im(1), ... ]

        data[0] = output_complex[0].re;
        if n % 2 == 0 {
            data[1] = output_complex[n / 2].re;
        } else {
            // Should usually be power of 2 in this library, but handle odd case
            data[1] = 0.0;
        }

        for i in 1..n / 2 {
            data[2 * i] = output_complex[i].re;
            data[2 * i + 1] = output_complex[i].im;
        }
    }

    fn compute_inverse(&mut self, data: &mut [f32]) {
        let n = self.n;
        // Unpack from [Re(0), Re(N/2), Re(1), Im(1)...] to [Complex]
        let mut input_complex = vec![Complex::zero(); n / 2 + 1];

        input_complex[0] = Complex::new(data[0], 0.0);
        if n % 2 == 0 {
            input_complex[n / 2] = Complex::new(data[1], 0.0);
        }

        for i in 1..n / 2 {
            input_complex[i] = Complex::new(data[2 * i], data[2 * i + 1]);
        }

        let mut output_real = vec![0.0; n];

        self.c2r
            .as_ref()
            .unwrap()
            .process_with_scratch(&mut input_complex, &mut output_real, &mut self.scratch)
            .unwrap();

        // Copy back
        data[0..n].copy_from_slice(&output_real);

        // Normalize (realfft usually doesn't normalize inverse)
        // The C code seems to handle scaling inside the wrapper?
        // Checking C source: "FFTW inverse is unnormalized".
        // Use existing C logic: The C wrapper has a `scale` field but `test_rfft.c` checks for unnormalized.
        // We will leave it unnormalized to match C behavior unless specified.
    }
}
