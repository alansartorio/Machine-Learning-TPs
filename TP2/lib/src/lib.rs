use pyo3::prelude::*;

mod decision_tree;
mod train;


#[pyfunction]
fn testing() {
    println!("AAAAA");
}


#[pymodule]
fn tp2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(testing, m)?)?;
    Ok(())
}
