//! `fereg._core`: the compiled part of the Python package (bindings of `fereg_core::fit`).

use pyo3::prelude::*;

#[pymodule]
mod _core {
    #[pymodule_export]
    #[allow(non_upper_case_globals)]
    const __version__: &str = fereg_core::VERSION;
}
