use pyo3::prelude::*;
use rayon::prelude::*;
// use std::time::SystemTime;
use numpy::ndarray::{ArrayD, ArrayViewD, Array3};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::{Array, Ix3};
use std::cmp;

fn alpha_d(i: usize, p1: [f32; 2], p2: [f32; 2], b:[f32; 2], delta: [f32; 2], d: usize) -> f32 {
    ((b[d] + (i as f32) * delta[d]) - p1[d]) / (p2[d] - p1[d])
}

fn p_d(alpha: f32, p1: [f32; 2], p2: [f32; 2], d: usize) -> f32 {
    p1[d] + alpha * (p2[d] - p1[d])
}

fn phi_d(alpha: f32, p1: [f32; 2], p2: [f32; 2], b: [f32; 2], delta: [f32; 2], d: usize) -> f32 {
    (p_d(alpha, p1, p2, d) - b[d]) / delta[d]
}

fn get_bounds(alpha_min: f32, alpha_max: f32, 
    alpha_dmin: [f32; 2], alpha_dmax: [f32; 2], 
    p1: [f32; 2], p2: [f32; 2],
    b: [f32; 2], delta: [f32; 2], n: &[usize], d: usize) -> [usize; 2] {
    let d_min: usize;
    let d_max: usize;
    if p1[d] < p2[d] {
        if alpha_min == alpha_dmin[d] {
            d_min = 1;
        } else {
            d_min = phi_d(alpha_min, p1, p2, b, delta, d).ceil() as usize;
        }

        if alpha_max == alpha_dmax[d] {
            d_max = n[d] - 1;
        } else {
            d_max = phi_d(alpha_max, p1, p2, b, delta, d).floor() as usize;
        }
    } else {
        if alpha_min == alpha_dmin[d] {
            d_max = n[d] - 2;
        } else {
            d_max = phi_d(alpha_min, p1, p2, b, delta, d).floor() as usize;
        }

        if alpha_max == alpha_dmax[d] {
            d_min = 0;
        } else {
            d_min = phi_d(alpha_max, p1, p2, b, delta, d).ceil() as usize;
        }
    }

    [d_min, d_max]
}

fn alpha_arr(d_min: [usize; 2], d_max: [usize; 2], p1: [f32; 2], p2: [f32; 2],  b: [f32; 2], delta: [f32; 2], d: usize) -> Vec<f32>{
    if p1[d] < p2[d] {
        (d_min[d]..d_max[d]+2).collect::<Vec<usize>>()
                                .into_iter()
                                .map(|alpha| alpha_d(alpha.try_into().unwrap(), p1, p2, b, delta, d))
                                .collect::<Vec<f32>>()
    } else {
        (d_min[d]..d_max[d]+1).rev()
                                .collect::<Vec<usize>>()
                                .into_iter()
                                .map(|alpha| alpha_d(alpha.try_into().unwrap(), p1, p2, b, delta, d))
                                .collect::<Vec<f32>>()
    }
}

fn get_path(p1: [f32; 2], p2: [f32; 2], b: [f32; 2], delta: [f32; 2], densities: &Array<f32, Ix2>, mask: &Array<bool, Ix2>) -> f32 {
    let n = densities.shape();
    let alpha_xmin = alpha_d(0, p1, p2, b, delta, 0).min(alpha_d(n[0]-1, p1, p2, b, delta, 0));
    let alpha_ymin = alpha_d(0, p1, p2, b, delta, 1).min(alpha_d(n[1]-1, p1, p2, b, delta, 1));
    let alpha_dmin = [alpha_xmin, alpha_ymin];
    let alpha_min = alpha_xmin.max(alpha_ymin);

    let alpha_xmax = alpha_d(0, p1, p2, b, delta, 0).max(alpha_d(n[0]-1, p1, p2, b, delta, 0));
    let alpha_ymax = alpha_d(0, p1, p2, b, delta, 1).max(alpha_d(n[1]-1, p1, p2, b, delta, 1));
    let alpha_dmax = [alpha_xmax, alpha_ymax];
    let alpha_max = alpha_xmax.min(alpha_ymax);
    
    //println!("{} {}", alpha_min, alpha_max);

    let [i_min, i_max] = get_bounds(alpha_min, alpha_max, alpha_dmin, alpha_dmax, p1, p2, b, delta, n, 0); 
    let [j_min, j_max] = get_bounds(alpha_min, alpha_max, alpha_dmin, alpha_dmax, p1, p2, b, delta, n, 1); 
    let d_min = [i_min, j_min];
    let d_max = [i_max, j_max];
    //println!("{} {} {} {}", i_min, j_min, i_max, j_max);
    
    let mut alpha_x = alpha_arr(d_min, d_max, p1, p2, b, delta, 0);
    let mut alpha_y = alpha_arr(d_min, d_max, p1, p2, b, delta, 1);
    let mut alpha_xy = vec![alpha_min];
    alpha_xy.append(&mut alpha_x); 
    alpha_xy.append(&mut alpha_y);
    alpha_xy.sort_by(|a, b| a.partial_cmp(b).unwrap());
    alpha_xy.dedup();
    // for v in alpha_xy.iter() {
    //     print!("{}, ", *v);
    // }
    // println!("");

    let d_conv = ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)).sqrt();

    let i_m =(1..alpha_xy.len()-1).collect::<Vec<usize>>()
                                                             .iter()
                                                             .map(|m| phi_d((alpha_xy[*m] + alpha_xy[*m-1])/2., p1, p2, b, delta, 0).floor() as usize)
                                                             .collect::<Vec<usize>>();
    let j_m =(1..alpha_xy.len()-1).collect::<Vec<usize>>()
                                                             .iter()
                                                             .map(|m| phi_d((alpha_xy[*m] + alpha_xy[*m-1])/2., p1, p2, b, delta, 1).floor() as usize)
                                                             .collect::<Vec<usize>>();
                                                                                                           

    let l = (1..alpha_xy.len()-1).collect::<Vec<usize>>()
                                 .iter()
                                 .map(|m| (alpha_xy[*m] - alpha_xy[*m-1]) * d_conv)
                                 .collect::<Vec<f32>>();
    // for v in i_m.iter() {
    //     print!("{}, ", *v);
    // }
    // println!("");
    // for v in j_m.iter() {
    //     print!("{}, ", *v);
    // }
    // println!("");

    if l.len() == 0 {
        0.
    } else {
        let coords = (1..i_m.len()).collect::<Vec<usize>>()
            .into_iter()
            .filter(|&m| (i_m[m] < n[0]) && (j_m[m] < n[1]))
            .map(|m| (m, mask[[i_m[m], j_m[m]]]));
        let mut kept_coords = vec![];
        for (m, keep) in coords {
            if keep {
                kept_coords.push(m);
            } else {
                break;
            }
        }
        let total = kept_coords.into_iter()
                                    .map(|m| densities[[i_m[m], j_m[m]]] * l[m])
                                    .sum();
        total
    }
}


fn project(start: [f32; 2], end: [f32; 2], num_pixels: usize, densities: &Array<f32, Ix2>, mask: &Array<bool, Ix2>) -> Vec<f32> {
    let b = [-0.5, -0.5];
    let delta = [1., 1.];
    (0..num_pixels).collect::<Vec<usize>>() 
        .into_par_iter()
        .map(|pixel| (pixel as f32) / (num_pixels as f32))
        .map(|fraction| [start[0] + fraction * (end[0] - start[0]), start[1] + fraction * (end[1] - start[1])])
        .map(|origin| get_path(origin, [origin[0]+1.0, origin[1] + 12.0], b, delta, densities, mask))
        .collect::<Vec<f32>>()
}


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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_get_path_test() {
        let p1 = [10.4, 2.];
        let p2 = [-1.1, 3.];
        let b = [-0.5, -0.5];
        let delta = [1., 1.];
        let mut densities = Array::<f32, Ix2>::zeros((10, 10).f());
        let mask = Array::<u8, Ix2>::ones((10, 10)).mapv(|m| m == 1);
        densities[[1, 3]] = 0.25;
        densities[[2, 2]] = 0.5;
        densities[[2, 3]] = 1.0;
        densities[[5, 5]] = 2.0;
        let path = get_path(p1, p2, b, delta, &densities, &mask);
        println!("path is {}", path);
        assert_eq!((path - 1.25).abs() < 0.1, true);
    }

    #[test]
    fn simple_get_path_test2() {
        let p1 = [4.6, -1.56];
        let p2 = [5.6, 10.44];
        let b = [-0.5, -0.5];
        let delta = [1., 1.];
        let mut densities = Array::<f32, Ix2>::zeros((10, 10).f());
        let mask = Array::<u8, Ix2>::ones((10, 10)).mapv(|m| m == 1);
        densities[[1, 3]] = 0.25;
        densities[[2, 2]] = 0.5;
        densities[[2, 3]] = 1.0;
        densities[[5, 5]] = 2.0;
        let path = get_path(p1, p2, b, delta, &densities, &mask);
        println!("path is {}", path);
        assert_eq!((path - 2.0069).abs() < 0.1, true);
    }

    #[test]
    fn simple_project_test() {
        let start = [-2.0, -2.0];
        let end = [13.0, -1.0];
        // let start = [-2.0, -2.0];
        // let end = [-1.0, 13.0];
        let num_pixels: usize = 25;
        let mut densities = Array::<f32, Ix2>::zeros((10, 10).f());
        let mask = Array::<u8, Ix2>::ones((10, 10)).mapv(|m| m == 1);
        densities[[1, 3]] = 0.25;
        densities[[2, 2]] = 0.5;
        densities[[2, 3]] = 1.0;
        densities[[5, 5]] = 2.0;

        // densities[[3, 1]] = 0.25;
        // densities[[2, 2]] = 0.5;
        // densities[[3, 2]] = 1.0;
        // densities[[5, 5]] = 2.0;
        let outcome = project(start, end, num_pixels, &densities, &mask); 
        // outcome.into_iter().map(|e| print!("{} ", e));
        println!("length {}", outcome.len());
        for e in outcome.iter() {
            print!("{}, ", e);
        }
        println!("");
        println!("success!");
        assert_ne!(1.0, 0.0);

    }

}   

