//! Autograd engine for Orca.

use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Sets whether gradient computation and tape recording are enabled.
///
/// If set to `false`, new operations will not record nodes on the autograd tape,
/// preventing memory accumulation and lock contention during evaluation/inference.
pub fn set_grad_enabled(enabled: bool) {
    GRAD_ENABLED.with(|state| state.set(enabled));
}

/// Returns whether gradient computation and tape recording are currently enabled.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

pub mod backend;
pub mod tape;
pub mod tensor;

pub use backend::{Autodiff, AutodiffStorage};
pub use tape::{NodeId, Tape};
pub use tensor::AutogradTensorExt;
