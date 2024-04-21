use std::sync::Arc;

use decision_tree::{Node, RootData};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use train::train;

pub mod decision_tree;
pub mod train;

#[pyfunction]
fn testing() {
    println!("AAAAA");
}

#[pyclass]
struct Tree {
    root: Node,
}

#[pymethods]
impl Tree {
    pub fn __repr__(&self) -> String {
        self.root.to_graphviz()
    }
}

#[pyfunction]
#[pyo3(name = "train")]
fn py_train(df: PyDataFrame) -> Tree {
    Tree {
        root: train(&df.into()),
    }
}

#[pymodule]
fn tp2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(testing, m)?)?;
    m.add_function(wrap_pyfunction!(py_train, m)?)?;
    Ok(())
}
