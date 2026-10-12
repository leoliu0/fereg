//! Deterministic SplitMix64 generator (plan D3: seeds never depend on threads).

#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Independent stream for `(seed, index)`, e.g. `(0x5EED, segment)`.
    pub fn stream(seed: u64, index: u64) -> Self {
        let mut s = SplitMix64::new(seed ^ index.wrapping_mul(0xD1B5_4A32_D192_ED03));
        s.next_u64();
        s
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}
