use pyo3::prelude::*;



#[pyfunction]
fn testing() {
    println!("AAAAA");
}


#[pymodule]
fn tp2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(testing, m)?)?;
    Ok(())
}
