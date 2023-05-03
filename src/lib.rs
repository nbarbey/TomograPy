use pyo3::prelude::*;

mod twoD;
mod threeD;
// #[pyfunction]
// fn axpy_py<'py>(
//     py: Python<'py>,
//     a: f64,
//     x: PyReadonlyArrayDyn<f64>,
//     y: PyReadonlyArrayDyn<f64>,
// ) -> &'py PyArrayDyn<f64> {
//     let x = x.as_array();
//     let y = y.as_array();
//     let z = axpy(a, x, y);
//     z.into_pyarray(py)
// }

#[pymodule]
fn tomograpy(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(axpy_py, m)?)?;
    Ok(())
}