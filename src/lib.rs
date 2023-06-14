use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn, PyArrayDyn};
use ndarray::prelude::*;
use pyo3::{pymodule, types::PyModule, PyResult, Python};

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
fn tomograpy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // example using immutable borrows producing a new array
    fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // example using a mutable borrow to modify an array in-place
    fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
        x *= a;
    }

    #[pyfn(m)]
    #[pyo3(name = "project_3d")]
    fn project_3d_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
        y: PyReadonlyArray2<f32>,
        z: PyReadonlyArray2<f32>,
        densities: PyReadonlyArray3<f32>,
        mask: PyReadonlyArray3<bool>,
        b: [f32; 3],
        delta: [f32; 3],
        unit_normal: [f32; 3],
        path_distance: f32
    ) -> &'py PyArray2<f32> {
        let x = x.as_array();
        let y = y.as_array();
        let z = z.as_array();
        let densities = densities.as_array();
        let mask = mask.as_array();

        let result = threeD::project_3d(
            &x.to_owned(),
            &y.to_owned(),
            &z.to_owned(),
            &densities.to_owned(),
            &mask.to_owned(),
            b,
            delta,
            unit_normal,
            path_distance);

        result.into_pyarray(py)
    }
    // wrapper of `axpy`
    #[pyfn(m)]
    #[pyo3(name = "axpy")]
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<f64>,
        y: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArrayDyn<f64> {
        let x = x.as_array();
        let y = y.as_array();
        let z = axpy(a, x, y);
        z.into_pyarray(py)
    }

    // wrapper of `mult`
    #[pyfn(m)]
    #[pyo3(name = "mult")]
    fn mult_py(_py: Python<'_>, a: f64, x: &PyArrayDyn<f64>) {
        let x = unsafe { x.as_array_mut() };
        mult(a, x);
    }

    Ok(())
}