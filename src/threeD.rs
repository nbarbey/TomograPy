use pyo3::prelude::*;
use rayon::prelude::*;
use ndarray::prelude::*;
//use ndarray::{Array, Ix3};
use std::cmp::Ordering;

fn alpha_3d(i: usize, p1: [f32; 3], p2: [f32; 3], b:[f32; 3], delta: [f32; 3], d: usize) -> f32 {
    ((b[d] + (i as f32) * delta[d]) - p1[d]) / (p2[d] - p1[d])
}

fn p_3d(alpha: f32, p1: [f32; 3], p2: [f32; 3], d: usize) -> f32 {
    p1[d] + alpha * (p2[d] - p1[d])
}

fn phi_3d(alpha: f32, p1: [f32; 3], p2: [f32; 3], b: [f32; 3], delta: [f32; 3], d: usize) -> f32 {
    (p_3d(alpha, p1, p2, d) - b[d]) / delta[d]
}

fn get_bounds_3d(alpha_min: f32, alpha_max: f32, 
    alpha_dmin: [f32; 3], alpha_dmax: [f32; 3], 
    p1: [f32; 3], p2: [f32; 3],
    b: [f32; 3], delta: [f32; 3], n: &[usize], d: usize) -> [usize; 2] {
    let d_min: usize;
    let d_max: usize;
    if p1[d] < p2[d] {
        if alpha_min == alpha_dmin[d] {
            d_min = 1;
        } else {
            d_min = phi_3d(alpha_min, p1, p2, b, delta, d).ceil() as usize;
        }

        if alpha_max == alpha_dmax[d] {
            d_max = n[d] - 1;
        } else {
            d_max = phi_3d(alpha_max, p1, p2, b, delta, d).floor() as usize;
        }
    } else {
        if alpha_min == alpha_dmin[d] {
            d_max = n[d] - 2;
        } else {
            d_max = phi_3d(alpha_min, p1, p2, b, delta, d).floor() as usize;
        }

        if alpha_max == alpha_dmax[d] {
            d_min = 0;
        } else {
            d_min = phi_3d(alpha_max, p1, p2, b, delta, d).ceil() as usize;
        }
    }

    [d_min, d_max]
}

fn alpha_arr_3d(d_min: [usize; 3], d_max: [usize; 3], p1: [f32; 3], p2: [f32; 3],  b: [f32; 3], delta: [f32; 3], d: usize) -> Vec<f32>{
    if p1[d] < p2[d] {
        (d_min[d]..d_max[d]+1).collect::<Vec<usize>>()
                                .into_iter()
                                .map(|alpha| alpha_3d(alpha.try_into().unwrap(), p1, p2, b, delta, d))
                                .collect::<Vec<f32>>()
    } else {
        (d_min[d]..d_max[d]+1).rev()
                                .collect::<Vec<usize>>()
                                .into_iter()
                                .map(|alpha| alpha_3d(alpha.try_into().unwrap(), p1, p2, b, delta, d))
                                .collect::<Vec<f32>>()
    }
}

fn calculate_path_lengths(p1: [f32; 3], p2: [f32; 3],
                          b: [f32; 3], delta: [f32; 3],
                          densities: &Array<f32, Ix3>, mask: &Array<bool, Ix3>)
    -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<f32>)  {
    let n = densities.shape();
    let alpha_xmin = alpha_3d(0, p1, p2, b, delta, 0).min(alpha_3d(n[0]-1, p1, p2, b, delta, 0));
    let alpha_ymin = alpha_3d(0, p1, p2, b, delta, 1).min(alpha_3d(n[1]-1, p1, p2, b, delta, 1));
    let alpha_zmin = alpha_3d(0, p1, p2, b, delta, 2).min(alpha_3d(n[2]-1, p1, p2, b, delta, 2));
    let alpha_dmin = [alpha_xmin, alpha_ymin, alpha_zmin];
    let alpha_min = alpha_xmin.max(alpha_ymin).max(alpha_zmin);

    let alpha_xmax = alpha_3d(0, p1, p2, b, delta, 0).max(alpha_3d(n[0]-1, p1, p2, b, delta, 0));
    let alpha_ymax = alpha_3d(0, p1, p2, b, delta, 1).max(alpha_3d(n[1]-1, p1, p2, b, delta, 1));
    let alpha_zmax = alpha_3d(0, p1, p2, b, delta, 2).max(alpha_3d(n[2]-1, p1, p2, b, delta, 2));
    let alpha_dmax = [alpha_xmax, alpha_ymax, alpha_zmax];
    let alpha_max = alpha_xmax.min(alpha_ymax).min(alpha_zmax);

    let [i_min, i_max] = get_bounds_3d(alpha_min, alpha_max, alpha_dmin, alpha_dmax, p1, p2, b, delta, n, 0);
    let [j_min, j_max] = get_bounds_3d(alpha_min, alpha_max, alpha_dmin, alpha_dmax, p1, p2, b, delta, n, 1);
    let [k_min, k_max] = get_bounds_3d(alpha_min, alpha_max, alpha_dmin, alpha_dmax, p1, p2, b, delta, n, 2);
    let d_min = [i_min, j_min, k_min];
    let d_max = [i_max, j_max, k_max];

    let mut alpha_x = alpha_arr_3d(d_min, d_max, p1, p2, b, delta, 0);
    let mut alpha_y = alpha_arr_3d(d_min, d_max, p1, p2, b, delta, 1);
    let mut alpha_z = alpha_arr_3d(d_min, d_max, p1, p2, b, delta, 2);
    let mut alpha_xyz = vec![alpha_min];
    alpha_xyz.append(&mut alpha_x);
    alpha_xyz.append(&mut alpha_y);
    alpha_xyz.append(&mut alpha_z);
    alpha_xyz.insert(alpha_xyz.len(), alpha_max);
    alpha_xyz.iter()
        .filter(|&n| n.is_finite())
        .collect::<Vec<&f32>>();
    alpha_xyz.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        //.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // println!("{:?}", alpha_xyz);
    // println!("{}", alpha_xyz[alpha_xyz.len()-1]);
    // println!("{}", alpha_min);
    alpha_xyz.dedup();

    let d_conv = ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt();

    // let i_m =(1..alpha_xyz.len()-1).collect::<Vec<usize>>()
    //                                 .par_iter()
    //                                 .map(|m| phi_3d((alpha_xyz[*m] + alpha_xyz[*m-1])/2., p1, p2, b, delta, 0).floor() as usize)
    //                                 .collect::<Vec<usize>>();
    // let j_m =(1..alpha_xyz.len()-1).collect::<Vec<usize>>()
    //                                 .par_iter()
    //                                 .map(|m| phi_3d((alpha_xyz[*m] + alpha_xyz[*m-1])/2., p1, p2, b, delta, 1).floor() as usize)
    //                                 .collect::<Vec<usize>>();
    // let k_m =(1..alpha_xyz.len()-1).collect::<Vec<usize>>()
    //                                 .par_iter()
    //                                 .map(|m| phi_3d((alpha_xyz[*m] + alpha_xyz[*m-1])/2., p1, p2, b, delta, 2).floor() as usize)
    //                                 .collect::<Vec<usize>>();

     let i_m =(1..alpha_xyz.len()).collect::<Vec<usize>>()
                                    .iter()
                                    .map(|m| phi_3d((alpha_xyz[*m] + alpha_xyz[*m-1])/2., p1, p2, b, delta, 0).floor() as usize)
                                    .collect::<Vec<usize>>();
    let j_m =(1..alpha_xyz.len()).collect::<Vec<usize>>()
                                    .iter()
                                    .map(|m| phi_3d((alpha_xyz[*m] + alpha_xyz[*m-1])/2., p1, p2, b, delta, 1).floor() as usize)
                                    .collect::<Vec<usize>>();
    let k_m =(1..alpha_xyz.len()).collect::<Vec<usize>>()
                                    .iter()
                                    .map(|m| phi_3d((alpha_xyz[*m] + alpha_xyz[*m-1])/2., p1, p2, b, delta, 2).floor() as usize)
                                    .collect::<Vec<usize>>();


    // let l = (1..alpha_xyz.len()-1).collect::<Vec<usize>>()
    //                              .par_iter()
    //                              .map(|m| (alpha_xyz[*m] - alpha_xyz[*m-1]) * d_conv)
    //                              .collect::<Vec<f32>>();

    let l = (1..alpha_xyz.len()).collect::<Vec<usize>>()
                                 .iter()
                                 .map(|m| (alpha_xyz[*m] - alpha_xyz[*m-1]) * d_conv)
                                 .collect::<Vec<f32>>();

    // for v in alpha_xyz.iter() {
    //     println!("{}", v);
    // }
    // println!("{:?}", alpha_xyz);
    //
    // println!("len of alpha_xyz {}", alpha_xyz.len());
    (i_m, j_m, k_m, l)
}

fn get_path_3d(p1: [f32; 3], p2: [f32; 3], b: [f32; 3], delta: [f32; 3], densities: &Array<f32, Ix3>, mask: &Array<bool, Ix3>) -> f32 {
    let n = densities.shape();
    if p1[0] != p2[0] && p1[1] != p2[1] && p1[2] != p2[2] {
        let (i_m, j_m, k_m, l) = calculate_path_lengths(p1, p2, b, delta, densities, mask);

        if l.len() == 0 {
            0.
        } else {
            let coords: Vec<(usize, bool)> = (0..l.len()).collect::<Vec<usize>>()
                .into_par_iter()
                .filter(|&m| (i_m[m] < n[0]) && (j_m[m] < n[1]) && (k_m[m] < n[2]))
                .map(|m| (m, mask[[i_m[m], j_m[m], k_m[m]]])).collect();
            let mut kept_coords = vec![];
            for (m, keep) in coords {
                if keep {
                    kept_coords.push(m);
                } else {
                    break;
                }
            }
            // println!("{:?}", l);

            let total = kept_coords.into_par_iter()
                .map(|m| densities[[i_m[m], j_m[m], k_m[m]]] * l[m])
                .sum();
            total
        }
    } else {
        -999.0
    }
}

pub fn project_3d(x: &Array<f32, Ix2>,
              y: &Array<f32, Ix2>,
              z: &Array<f32, Ix2>,
              densities: &Array<f32, Ix3>,
              mask: &Array<bool, Ix3>,
              b: [f32; 3],
              delta: [f32; 3],
              unit_normal: [f32; 3],
              path_distance: f32) -> Array<f32, Ix2> {
    // Create coordinate array to iterate over
    let mut coords = Vec::<(usize, usize)>::new();
    for j in 0..x.shape()[0] {
        for i in 0..x.shape()[1] {
            coords.push((i, j));
        }
    }

    // for (i, j) in coords.iter() {
    //     println!("({}, {}) ({}, {}, {}) ({}, {}, {})", i, j, x[[*i, *j]], y[[*i, *j]], z[[*i, *j]],
    //              x[[*i, *j]] + unit_normal[0]*path_distance,
    //                                          y[[*i, *j]] + unit_normal[1]*path_distance,
    //                                          z[[*i, *j]] + unit_normal[2]*path_distance);
    //
    // }

    // Iterate in parallel to calculate the contributions
    let calculation = coords.into_par_iter()
        .map(|(i, j)| get_path_3d([x[[i, j]], y[[i, j]], z[[i, j]]], 
                                            [x[[i, j]] + unit_normal[0]*path_distance, 
                                             y[[i, j]] + unit_normal[1]*path_distance,
                                             z[[i, j]] + unit_normal[2]*path_distance], b, delta, densities, mask))
        .collect::<Vec<f32>>();

    // Map back to an image
    let new_shape = x.shape();
    let (row, col) = (new_shape[0], new_shape[1]);
    let result = Array::from_shape_vec((row, col), calculation).unwrap(); // todo: more elegantly handle this unwrap
    result
}

pub fn backproject_3d<'a>(x: &'a Array<f32, Ix2>,
                      y: &'a Array<f32, Ix2>,
                      z: &'a Array<f32, Ix2>,
                      image: &'a Array<f32, Ix2>,
                      cube: &'a mut Array<f32, Ix3>,
                      n: &[usize],
                      mask: &'a Array<bool, Ix3>,
                      b: [f32; 3],
                      delta: [f32; 3],
                      unit_normal: [f32; 3],
                      path_distance: f32,
                      use_precise_method: bool) -> &'a Array<f32, Ix3> {  // todo: deal with use_precise_method

    // Create coordinate array to iterate over
    let mut coords = Vec::<(usize, usize)>::new();
    for i in 0..x.shape()[0] {
        for j in 0..x.shape()[1] {
            coords.push((i, j));
        }
    }

    let results = coords.into_par_iter().map(|(i,j)|
        {
            let p1 = [x[[i, j]], y[[i, j]], z[[i, j]]];
            let p2 = [x[[i, j]] + unit_normal[0] * path_distance,
                y[[i, j]] + unit_normal[1] * path_distance,
                z[[i, j]] + unit_normal[2] * path_distance];

            let (i_m, j_m, k_m, l) = calculate_path_lengths(p1, p2, b, delta, cube, mask);
            (image[[i,j]], i_m, j_m, k_m, l)
        }
    ).collect::<Vec<(f32, Vec<usize>, Vec<usize>, Vec<usize>, Vec<f32>)>>();

    // println!("res {:?}", results[0]);
    // println!("res {:?}", results[1]);
    // println!("res {:?}", results[24]);

    for (value, i_m, j_m, k_m, l) in results {
        for m in 0..i_m.len() {
            // if mask[[i_m[m], j_m[m], k_m[m]]] {
            if (i_m[m] < n[0]) && (j_m[m] < n[1]) && (k_m[m] < n[2]) && mask[[i_m[m], j_m[m], k_m[m]]] {
                cube[[i_m[m], j_m[m], k_m[m]]] += value * l[m];
            } else {
                break;
            }
        }
    }
    // coords.into_par_iter().for_each(|(i,j)|
    //     {
    //         let p1 = [x[[i, j]], y[[i, j]], z[[i, j]]];
    //     let p2 =  [x[[i, j]] + unit_normal[0]*path_distance,
    //                      y[[i, j]] + unit_normal[1]*path_distance,
    //                      z[[i, j]] + unit_normal[2]*path_distance];
    //
    //     if use_precise_method {
    //         let (i_m, j_m, k_m, l) = calculate_path_lengths(p1, p2, b, delta, cube, mask);
    //
    //         // TODO: worry about if there's an obstacle
    //
    //         for m in 0..i_m.len() {
    //             if mask[[i_m[m], j_m[m], k_m[m]]] {
    //                 //cube[[i_m[m], j_m[m], k_m[m]]] += image[[i, j]] * l[m];
    //             } else {
    //                 break;
    //             }
    //         }
    //     } else
    //     {  // TODO: implement an imprecise method?
    //         cube[[0, 0, 0]] = 0.;
    //     }
    //     });

    // for (i, j) in coords.into_iter() {
    //     let p1 = [x[[i, j]], y[[i, j]], z[[i, j]]];
    //     let p2 =  [x[[i, j]] + unit_normal[0]*path_distance,
    //                      y[[i, j]] + unit_normal[1]*path_distance,
    //                      z[[i, j]] + unit_normal[2]*path_distance];
    //
    //     if use_precise_method {
    //         let (i_m, j_m, k_m, l) = calculate_path_lengths(p1, p2, b, delta, cube, mask);
    //
    //         for m in 0..i_m.len() {
    //             if mask[[i_m[m], j_m[m], k_m[m]]] {
    //                 cube[[i_m[m], j_m[m], k_m[m]]] += image[[i, j]] * l[m];
    //             } else {
    //                 break;
    //             }
    //         }
    //     } else
    //     {  // TODO: implement an imprecise method?
    //         cube[[0, 0, 0]] = 0.;
    //     }
    // }
    cube
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[ignore]
    #[test]
    fn simple_run() {
        let b = [-0.5, -0.5, -0.5];
        let delta = [1., 1., 1.];

        let image_size = 128;
        let x = Array::from_elem((image_size, image_size), 1.);
        let y = Array::from_elem((image_size, image_size), 1.);
        let z = Array::from_elem((image_size, image_size), 1.);
        let densities = Array::<f32, Ix3>::zeros((128, 128, 128).f());
        let mask = Array::<u8, Ix3>::ones((128, 128, 128)).mapv(|m| m == 1);

        // Define unit normal and path distance
        let unit_normal = [0.0, 1.0, 0.0];
        let path_distance = 5.0;

        let result = project_3d(&x, &y, &z, &densities, &mask, b, delta, unit_normal, path_distance);
        assert_eq!(result[[0, 0]], 0.0);
    }

    #[ignore]
    #[test]
    fn simple_backproject() {
        let b = [-0.5, -0.5, -0.5];
        let delta = [1., 1., 1.];

        let image_size = 512;
        let x = Array::from_elem((image_size, image_size), 1.);
        let y = Array::from_elem((image_size, image_size), 1.);
        let z = Array::from_elem((image_size, image_size), 1.);
        let image = Array::<f32, Ix2>::zeros((image_size, image_size).f());

        let cube_size = 128;
        let mut cube = Array::<f32, Ix3>::zeros((cube_size, cube_size, cube_size).f());
        let mask = Array::<u8, Ix3>::ones((cube_size, cube_size, cube_size)).mapv(|m| m == 100);

        // Define unit normal and path distance
        let unit_normal = [0.0, 1.0, 0.0];
        let path_distance = 5.0;

        let result = backproject_3d(&x, &y, &z, &image, &mut cube, &mask, b, delta, unit_normal, path_distance, true);
        //assert_eq!(result, 0);
        assert_eq!(result[[0, 0, 0]], 0.0);
    }

    #[test]
    fn alpha_3d_with_zero_values() {
        let result = alpha_3d(0,
                              [0.0, 0.0, 0.0],
                              [1.0, 1.0, 1.0],
                              [0.0, 0.0, 0.0],
                              [1.0, 1.0, 1.0], 0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn alpha_3d_with_small_values() {
        let result = alpha_3d(10,
                              [1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [1E-3, 1E-3, 1E-3],
                              [1.0, 1.0, 1.0], 0);
        assert_approx_eq!(result, 3.0, 0.05);
    }

    #[ignore]
    #[test]
    fn project_through_cube_of_ones_with_almost_axis_aligned_slice() {
        let (p1, p2) = ([6.0, 1.0, 1.0], [-4.0, 0.9999999, 0.9999999]);

        let b = [0.0, 0.0, 0.0];
        let delta = [1., 1., 1.];
        let cube_size = 5;
        let mut cube = Array::<f32, Ix3>::ones((cube_size, cube_size, cube_size).f());
        let mask = Array::<u8, Ix3>::ones((cube_size, cube_size, cube_size)).mapv(|m| m == 1);

        let result = get_path_3d(p1, p2, b, delta, &cube, &mask);
        assert_approx_eq!(result, 0.0, 1E-6)
    }

    #[ignore]
    #[test]
    fn calculate_path_lengths_through_cube_of_ones_with_almost_axis_aligned_slice() {
        let (p1, p2) = ([6.0, 1.0, 1.0], [-4.0, 0.9999999, 0.9999999]);
        let b = [0.0, 0.0, 0.0];
        let delta = [1., 1., 1.];
        let cube_size = 5;
        let mut cube = Array::<f32, Ix3>::ones((cube_size, cube_size, cube_size).f());
        let mask = Array::<u8, Ix3>::ones((cube_size, cube_size, cube_size)).mapv(|m| m == 1);

        let (i_m, j_m, k_m, l) = calculate_path_lengths(p1, p2, b, delta, &cube, &mask);
        // for index in 0..i_m.len() {
        //     println!("{} {} {} {}", i_m[index], j_m[index], k_m[index], l[index]);
        // }
        assert_eq!(false, true);
        //assert_approx_eq!(-1.0, 0.0, 1E-6)
    }

    #[test]
    fn test_sort() {
        let mut v = [0.2, 0.3, 0.4, 0.5, 0.6, -0.0];
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(v == [-0.0, 0.2, 0.3, 0.4, 0.5, 0.6]);
    }
}
