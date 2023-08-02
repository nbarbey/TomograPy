use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArray2, PyArray3,
            PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn, PyArrayDyn};
use ndarray::prelude::*;
use pyo3::{pymodule, types::PyModule, PyResult, Python};

mod threeD;

#[pymodule]
fn tomograpy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
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

    #[pyfn(m)]
    #[pyo3(name = "backproject_3d")]
    fn backproject_3d_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
        y: PyReadonlyArray2<f32>,
        z: PyReadonlyArray2<f32>,
        image: PyReadonlyArray2<f32>,
        cube: &PyArray3<f32>,
        mask: PyReadonlyArray3<bool>,
        b: [f32; 3],
        delta: [f32; 3],
        unit_normal: [f32; 3],
        path_distance: f32,
        use_precise_method: bool
    ) -> &'py PyArray3<f32> {
        let x = x.as_array().to_owned();
        let y = y.as_array().to_owned();
        let z = z.as_array().to_owned();
        let image = image.as_array().to_owned();
        let n = cube.shape();
        // println!("{:?}", n);
        let mut cube = unsafe { cube.as_array_mut().to_owned()};
        let mask = mask.as_array().to_owned();

        let result = threeD::backproject_3d(
            &x,
            &y,
            &z,
            &image,
            &mut cube,
            n,
            &mask,
            b,
            delta,
            unit_normal,
            path_distance,
            use_precise_method);

        result.clone().into_pyarray(py)
    }

    Ok(())
}