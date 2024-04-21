use pyo3::prelude::*;

pub mod decision_tree;
pub mod train;

#[pyfunction]
fn testing() {
    println!("AAAAA");
}

//struct Tree {
//root: Box<Node<'static>>
//}

//#[pyfunction]
//fn train() -> Tree {

//}

#[pymodule]
fn tp2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(testing, m)?)?;
    Ok(())
}
