use std::{collections::HashMap, sync::Arc};

use decision_tree::{Node, RootData, Value};
use pyo3::{prelude::*, types::PyDict};
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
    pub fn to_graphviz(&self) -> String {
        self.root.to_graphviz()
    }

    pub fn classify(&self, record: HashMap<String, Value>) -> String {
        self.root.classify_with_names(
            &record
                .iter()
                .map(|(attr, &value)| (attr.as_str(), value))
                .collect(),
        )
    }
}

#[pyfunction]
#[pyo3(name = "train")]
fn py_train(
    df: PyDataFrame,
    output_col: String,
    value_mapping: HashMap<String, HashMap<Value, String>>,
    max_depth: Option<usize>,
) -> Tree {
    Tree {
        root: train(&df.into(), &output_col, value_mapping, max_depth),
    }
}

#[pymodule]
fn tp2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(testing, m)?)?;
    m.add_function(wrap_pyfunction!(py_train, m)?)?;
    Ok(())
}
