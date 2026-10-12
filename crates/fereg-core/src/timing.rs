//! Phase timings for `verbose(2)` (plan D2 roofline check).

use std::time::Instant;

pub struct PhaseTimer {
    start: Instant,
    last: Instant,
    phases: Vec<(String, f64, u64)>,
}

impl Default for PhaseTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl PhaseTimer {
    pub fn new() -> Self {
        let now = Instant::now();
        PhaseTimer { start: now, last: now, phases: Vec::new() }
    }

    /// Closes the current phase: records its wall time and the bytes it streamed.
    pub fn mark(&mut self, name: &str, bytes: u64) {
        let now = Instant::now();
        self.phases.push((name.to_string(), (now - self.last).as_secs_f64(), bytes));
        self.last = now;
    }

    pub fn total(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// One line per phase: name, seconds, GB streamed and GB/s.
    pub fn report(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .phases
            .iter()
            .map(|(name, secs, bytes)| {
                let gb = *bytes as f64 / 1e9;
                let rate = if *secs > 0.0 { gb / secs } else { 0.0 };
                format!("{name:<24} {secs:>9.4}s {gb:>9.3} GB {rate:>7.1} GB/s")
            })
            .collect();
        out.push(format!("{:<24} {:>9.4}s", "total", self.total()));
        out
    }
}
