#![feature(prelude_import)]
//! The crate `kde_ocl_sys` implements
//! [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) (KDE) using
//! OpenCL to accelerate the computation with GPUs. Currently, it only implements the (multivariate)
//! Gaussian kernel. This crate is an auxiliary library for the Python code using it along with this
//! code. For this reason, it receives Numpy arrays as input, and writes the results also in Numpy
//! arrays.
//!
//! The equation for the KDE with $`n`$ training points of dimensionality $`d`$ is the following
//! expression:
//!
//! ```math
//! \hat{f}_{n}(\mathbf{x}) = \frac{1}{n(2\pi)^{d/2}\sqrt{\lvert\mathbf{\Sigma}\rvert}}
//! \sum_{i=1}^{n} \exp\left(-\frac{1}{2}(\mathbf{x} - \mathbf{x}_{i})^{T}\Sigma^{-1}(\mathbf{x} -
//! \mathbf{x}_{i})\right)
//! ```
//! # Implementation
//!
//! The evaluation of each Gaussian is implemented using the Cholesky decomposition:
//!
//! ```math
//! \mathbf{\Sigma} = \mathbf{L}\mathbf{L}^{T}
//! ```
//!
//! such that:
//!
//! ```math
//! (\mathbf{x} - \mathbf{x}_{i})^{T}(\mathbf{L}\mathbf{L}^T)^{-1}(\mathbf{x} - \mathbf{x}_{i}) =
//! (\mathbf{x} - \mathbf{x}_{i})^{T}\mathbf{L}^{-T}\mathbf{L}^{-1}(\mathbf{x} - \mathbf{x}_{i}) =
//! (\mathbf{L}^{-1}(\mathbf{x} - \mathbf{x}_{i}))^T(\mathbf{L}^{-1}(\mathbf{x} - \mathbf{x}_{i}))
//! ```
//! Let $`\mathbf{y}^{i} = \mathbf{L}^{-1}(\mathbf{x} - \mathbf{x}_{i})`$, $`\mathbf{y}^{i}`$ can be
//! easily obtained by forward-solving $`\mathbf{L}\mathbf{y}^{i} = (\mathbf{x} - \mathbf{x}_{i})`$
//! (quite easy to solve as $`\mathbf{L}`$ is lower triangular). Once $`\mathbf{y}^{i}`$ is solved:
//!
//! ```math
//! \left(-\frac{1}{2}(\mathbf{x} - \mathbf{x}_{i})^{T}\Sigma^{-1}(\mathbf{x} - \mathbf{x}_{i})
//! \right) = -\frac{1}{2} \sum_{j=1}^{d} (y_{j}^{i})^{2}
//! ```
//!
//! Once this is computed, we only have to substract the constant (because it does not change for
//! every training instance in the KDE) quantity:
//! ```math
//!  \log n + \frac{d}{2}\log2\pi+ \log\lvert\mathbf{L}\rvert
//! ```
//! To obtain the log probability density function (logpdf) of a test point $`\mathbf{x}`$ with
//! respect to a training point $`\mathbf{x}_{i}`$, $`l_{i}`$:
//!
//! ```math
//! l_{i} = -\frac{1}{2} \sum_{j=1}^{d} (y_{j}^{i})^{2} - \log n - \frac{d}{2}\log2\pi -
//! \log\lvert\mathbf{L}\rvert
//! ```
//! The pdf of a test point $`\mathbf{x}`$ with respect to a training point
//! $`\mathbf{x}_{i}`$ is equal to the exponentiation of the previous logpdf:
//!
//! ```math
//! \exp\left(-\frac{1}{2} \sum_{j=1}^{d} (y_{j}^{i})^{2} - \log n - \frac{d}{2}\log2\pi -
//! \log\lvert\mathbf{L}\rvert\right)
//! ```
//!
//! To obtain the pdf of the KDE model, we have to sum over all the training points:
//!
//! ```math
//! \sum_{i=1}^{n}\exp\left(-\frac{1}{2} \sum_{j=1}^{d} (y_{j}^{i})^{2} - \log n -
//! \frac{d}{2}\log2\pi - \log\lvert\mathbf{L}\rvert\right)
//! ```
//! ## LogSumExp trick
//! For computing the logpdf of the KDE model, the
//! [LogSumExp trick](https://en.wikipedia.org/wiki/LogSumExp) is used for improved precision:
//!
//! ```math
//! \log\sum_{i}^{n} \exp x_{i} = \max_{i} x_{i} + \log\sum_{i}^{n} \exp\left(x_{i} -
//! \max_{i} x_{i}\right)
//! ```
//!
//! Then, the logpdf of the KDE model is:
//!
//! ```math
//! \max_{i} l_{i} + \log\sum_{i=1}^{n}\exp\left(l_{i} - \max_{i} l_{i}\right)
//! ```
#[prelude_import]
use std::prelude::v1::*;
#[macro_use]
extern crate std;

extern crate libc;
extern crate ndarray;
extern crate num;

extern crate ocl;

use libc::{c_double, size_t};
use ndarray::{Array2, ShapeBuilder};
use ocl::{enums::{DeviceInfo, DeviceInfoResult}, Buffer, ProQue};
use std::f64;
use std::mem;
use std::ptr;
use std::slice;

mod denominator {

    //pub use denominator_onlygaussian::{
    //    ckde_free, ckde_init, gaussian_regression_free, gaussian_regression_init, GaussianRegression,
    //};

    //pub use denominator_onlykde::logdenominator_dataset_onlykde;








    // We need to convert the strides from bytes to the number of elements.





    // Summing the sqrt of the covariance determinant using the cholesky matrix.

    // TODO: The OpenCL code should be included in the code to make easier distribute the library.

    //    println!("\t[RUST] New proque {:p}", ptr);








    //    println!("\t[RUST] Gaussian_kde_init {:p}", ptr_kde);


    //    println!("\t[RUST] Gaussian_kde_free {:p}", kde);

    //    println!("\t[RUST] Free Proque! {:p}", pro_que);



    //        if DEBUG_MODE {
    //            println!("Rust strides: ({}, {}), row major", row_stride, column_stride);
    //        }
    //        if DEBUG_MODE {
    //            println!("Rust strides: ({}, {}), column major", row_stride, column_stride);
    //        }


    //static mut DEBUG_MODE: bool = false;

    //#[no_mangle]
    //pub unsafe extern "C" fn change_debug(debug: bool) {
    //    println!("Change debug: {}", debug);
    //    DEBUG_MODE = debug;
    //}





    //    Iterates over training or testing data?






















    //    println!("\t[RUST] kde_logpdf KDE {:p}", kde);

    //        if DEBUG_MODE {
    //            println!("Iterate test");
    //        }
    //        if DEBUG_MODE {
    //            println!("Iterate train");
    //        }










    //    println!("train_rowmajor = {}, test_rowmajor = {}, train_leading_dimension = {}, \
    //    test_leading_dimension = {}", kde.rowmajor, test_rowmajor,
    //             kde.leading_dimension, test_leading_dimension);


    //    if DEBUG_MODE {
    //        println!("Train rowmajor: {}, n: {}, d: {}, leading_dimension: {}",
    //                 kde.rowmajor, kde.n, kde.d, kde.leading_dimension);
    //
    //        print_buffers!(pro_que, kde.training_data);
    //        println!("Debugging mode!");
    //    }







































    // TODO: If n < 2m, is it better to iterate over the training data?















    // Writes the max loglikelihoods in the max_buffer
    // TODO: Find max with euclidian distance is probably faster.

    // Computes the loglikelihood using the max_buffer.


















    // Writes the max loglikelihoods in the max_buffer












    use crate::{empty_buffers, copy_buffers, Error, GaussianKDE,
                buffer_fill_value};
    use std::slice;
    use std::ptr;
    use ocl::{Buffer, ProQue};
    use libc::{c_uint, c_double};
    mod denominator_gaussian {
        use crate::{empty_buffers, copy_buffers, Error, DoubleNumpyArray,
                    buffer_fill_value, get_max_work_size, is_rowmajor,
                    max_gpu_vec_copy, log_sum_gpu_vec, max_gpu_mat,
                    sum_gpu_mat};
        use crate::denominator::{CKDE, s2, s1_s3_coefficients};
        use std::slice;
        use std::f64;
        use ocl::{Buffer, ProQue};
        use libc::{c_double};
        #[no_mangle]
        pub unsafe extern "C" fn logdenominator_dataset_gaussian(ckde:
                                                                     *mut CKDE,
                                                                 pro_que:
                                                                     *mut ProQue,
                                                                 x:
                                                                     *const DoubleNumpyArray,
                                                                 result:
                                                                     *mut c_double,
                                                                 error:
                                                                     *mut Error) {
            *error = Error::NotFinished;
            let mut ckde = Box::from_raw(ckde);
            let mut pro_que = Box::from_raw(pro_que);
            let m = *(*x).shape;
            if (*ckde.kde).n >= m {
                logdenominator_iterate_test_gaussian(&mut ckde, &mut pro_que,
                                                     x, result, error);
            } else {
                logdenominator_iterate_train_gaussian(&mut ckde, &mut pro_que,
                                                      x, result, error);
            }
            Box::into_raw(ckde);
            Box::into_raw(pro_que);
        }
        fn onlygaussian_exponent_coefficients_iterate_test_name(rowmajor:
                                                                    bool)
         -> &'static str {
            if rowmajor {
                "onlygaussian_exponent_coefficients_iterate_test_rowmajor"
            } else {
                "onlygaussian_exponent_coefficients_iterate_test_columnmajor"
            }
        }
        fn onlygaussian_exponent_coefficients_iterate_train_high_memory_name(rowmajor:
                                                                                 bool)
         -> &'static str {
            if rowmajor {
                "onlygaussian_exponent_coefficients_iterate_train_high_memory_rowmajor"
            } else {
                "onlygaussian_exponent_coefficients_iterate_train_high_memory_columnmajor"
            }
        }
        fn onlygaussian_exponent_coefficients_iterate_train_low_memory_checkmax_name(rowmajor:
                                                                                         bool)
         -> &'static str {
            if rowmajor {
                "onlygaussian_exponent_coefficients_iterate_train_low_memory_checkmax_rowmajor"
            } else {
                "onlygaussian_exponent_coefficients_iterate_train_low_memory_checkmax_columnmajor"
            }
        }
        fn onlygaussian_exponent_coefficients_iterate_train_low_memory_compute_name(rowmajor:
                                                                                        bool)
         -> &'static str {
            if rowmajor {
                "onlygaussian_exponent_coefficients_iterate_train_low_memory_compute_rowmajor"
            } else {
                "onlygaussian_exponent_coefficients_iterate_train_low_memory_compute_columnmajor"
            }
        }
        unsafe fn logdenominator_iterate_test_gaussian(ckde: &Box<CKDE>,
                                                       pro_que:
                                                           &mut Box<ProQue>,
                                                       x:
                                                           *const DoubleNumpyArray,
                                                       result: *mut c_double,
                                                       error: *mut Error) {
            let kde = Box::from_raw(ckde.kde);
            let test_shape = slice::from_raw_parts((*x).shape, 2);
            let m = test_shape[0];
            let d_test = test_shape[1];
            let n = kde.n;
            let max_work_size = get_max_work_size(&pro_que);
            let local_work_size =
                if n < max_work_size { n } else { max_work_size };
            let num_groups =
                (n as f32 / local_work_size as f32).ceil() as usize;
            let test_slice = slice::from_raw_parts((*x).ptr, m * d_test);
            let (test_instances_buffer,) =
                {
                    (match Buffer::builder().context(pro_que.context()).len(test_slice.len()).copy_host_slice(&test_slice).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },)
                };
            let (s1, s3, final_result_buffer, tmp_train_coefficients,
                 max_coefficients) =
                {
                    (match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(n).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(num_groups).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     })
                };
            let a = 0.5 * (ckde.precision_variable + s2(ckde));
            let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);
            s1_s3_coefficients(ckde, pro_que, &test_instances_buffer,
                               test_leading_dimension as u32, test_rowmajor,
                               &s1, &s3, m);
            let kernel_onlygaussian_exponent_coefficients =
                pro_que.kernel_builder(onlygaussian_exponent_coefficients_iterate_test_name(kde.rowmajor)).global_work_size(n).arg(&kde.training_data).arg(kde.leading_dimension
                                                                                                                                                               as
                                                                                                                                                               u32).arg(&ckde.precision).arg(&s1).arg((4.0
                                                                                                                                                                                                           *
                                                                                                                                                                                                           a).recip()).arg(&s3).arg_named("test_index",
                                                                                                                                                                                                                                          &0u32).arg(&tmp_train_coefficients).build().expect("Kernel onlygaussian_exponent_coefficients build failed.");
            let kernel_log_sum_gpu =
                pro_que.kernel_builder("copy_logpdf_result").global_work_size(1).arg(&tmp_train_coefficients).arg(&max_coefficients).arg(&final_result_buffer).arg_named("offset",
                                                                                                                                                                         &0u32).build().expect("Kernel copy_logpdf_result build failed.");
            for i in 0..m {
                kernel_onlygaussian_exponent_coefficients.set_arg("test_index",
                                                                  i as
                                                                      u32).unwrap();
                kernel_onlygaussian_exponent_coefficients.enq().expect("Error while executing kernel_onlygaussian_exponent_coefficients kernel.");
                max_gpu_vec_copy(pro_que, &tmp_train_coefficients,
                                 &max_coefficients, n, max_work_size,
                                 local_work_size, num_groups);
                log_sum_gpu_vec(&pro_que, &tmp_train_coefficients,
                                &max_coefficients, n, max_work_size,
                                local_work_size, num_groups);
                kernel_log_sum_gpu.set_arg("offset", i as u32).unwrap();
                kernel_log_sum_gpu.enq().expect("Error while executing copy_logpdf_result kernel.");
            }
            let kernel_sum_constant =
                pro_que.kernel_builder("sum_constant").global_work_size(m).arg(&final_result_buffer).arg(ckde.lognorm_factor).build().expect("Kernel sum_constant build failed.");
            kernel_sum_constant.enq().expect("Error while executing sum_constant kernel.");
            let final_result = slice::from_raw_parts_mut(result, m);
            final_result_buffer.cmd().queue(pro_que.queue()).read(final_result).enq().expect("Error reading result data.");
            Box::into_raw(kde);
            *error = Error::NoError;
        }
        unsafe fn logdenominator_iterate_train_gaussian(ckde: &Box<CKDE>,
                                                        pro_que:
                                                            &mut Box<ProQue>,
                                                        x:
                                                            *const DoubleNumpyArray,
                                                        result: *mut c_double,
                                                        error: *mut Error) {
            let m = *(*x).shape;
            let tmp_vec_buffer =
                Buffer::<f64>::builder().context(pro_que.context()).len(m *
                                                                            (*ckde.kde).n).build();
            match tmp_vec_buffer {
                Ok(b) => {
                    logdenominator_iterate_train_gaussian_high_memory(ckde,
                                                                      pro_que,
                                                                      x,
                                                                      result,
                                                                      &b,
                                                                      error)
                }
                Err(_) => {
                    logdenominator_iterate_train_gaussian_low_memory(ckde,
                                                                     pro_que,
                                                                     x,
                                                                     result,
                                                                     error);
                }
            }
        }
        unsafe fn logdenominator_iterate_train_gaussian_low_memory(ckde:
                                                                       &Box<CKDE>,
                                                                   pro_que:
                                                                       &mut Box<ProQue>,
                                                                   x:
                                                                       *const DoubleNumpyArray,
                                                                   result:
                                                                       *mut c_double,
                                                                   error:
                                                                       *mut Error) {
            let kde = Box::from_raw(ckde.kde);
            let test_shape = slice::from_raw_parts((*x).shape, 2);
            let m = test_shape[0];
            let d_test = test_shape[1];
            let n = kde.n;
            let test_slice = slice::from_raw_parts((*x).ptr, m * d_test);
            let (test_instances_buffer,) =
                {
                    (match Buffer::builder().context(pro_que.context()).len(test_slice.len()).copy_host_slice(&test_slice).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },)
                };
            let (s1, s3, final_result_buffer, max_coefficients) =
                {
                    (match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     })
                };
            buffer_fill_value(&pro_que, &max_coefficients, m, f64::MIN);
            buffer_fill_value(&pro_que, &final_result_buffer, m, 0.0f64);
            let a = 0.5 * (ckde.precision_variable + s2(ckde));
            let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);
            s1_s3_coefficients(ckde, pro_que, &test_instances_buffer,
                               test_leading_dimension as u32, test_rowmajor,
                               &s1, &s3, m);
            let kernel_onlygaussian_exponent_coefficients_checkmax =
                pro_que.kernel_builder(onlygaussian_exponent_coefficients_iterate_train_low_memory_checkmax_name(kde.rowmajor)).global_work_size(m).arg(&kde.training_data).arg(kde.leading_dimension
                                                                                                                                                                                    as
                                                                                                                                                                                    u32).arg(&ckde.precision).arg(&s1).arg((4.0
                                                                                                                                                                                                                                *
                                                                                                                                                                                                                                a).recip()).arg(&s3).arg_named("train_index",
                                                                                                                                                                                                                                                               &0u32).arg(&max_coefficients).build().expect("Kernel kernel_onlygaussian_exponent_coefficients_checkmax build failed.");
            let kernel_onlygaussian_exponent_coefficients_compute =
                pro_que.kernel_builder(onlygaussian_exponent_coefficients_iterate_train_low_memory_compute_name(kde.rowmajor)).global_work_size(m).arg(&kde.training_data).arg(kde.leading_dimension
                                                                                                                                                                                   as
                                                                                                                                                                                   u32).arg(&ckde.precision).arg(&s1).arg((4.0
                                                                                                                                                                                                                               *
                                                                                                                                                                                                                               a).recip()).arg(&s3).arg_named("train_index",
                                                                                                                                                                                                                                                              &0u32).arg(&max_coefficients).arg(&final_result_buffer).build().expect("Kernel kernel_onlygaussian_exponent_coefficients_checkmax build failed.");
            let kernel_log_and_sum =
                pro_que.kernel_builder("log_and_sum").global_work_size(m).arg(&final_result_buffer).arg(&max_coefficients).build().expect("Kernel log_and_sum build failed.");
            for i in 0..n {
                kernel_onlygaussian_exponent_coefficients_checkmax.set_arg("train_index",
                                                                           i
                                                                               as
                                                                               u32).unwrap();
                kernel_onlygaussian_exponent_coefficients_checkmax.enq().expect("Error while executing kernel_onlygaussian_exponent_coefficients_checkmax kernel.");
            }
            for i in 0..n {
                kernel_onlygaussian_exponent_coefficients_compute.set_arg("train_index",
                                                                          i as
                                                                              u32).unwrap();
                kernel_onlygaussian_exponent_coefficients_compute.enq().expect("Error while executing kernel_onlygaussian_exponent_coefficients_compute kernel.");
            }
            kernel_log_and_sum.enq().expect("Error while executing kernel_log_and_sum kernel.");
            let kernel_sum_constant =
                pro_que.kernel_builder("sum_constant").global_work_size(m).arg(&final_result_buffer).arg(ckde.lognorm_factor).build().expect("Kernel sum_constant build failed.");
            kernel_sum_constant.enq().expect("Error while executing sum_constant kernel.");
            let final_result = slice::from_raw_parts_mut(result, m);
            final_result_buffer.cmd().queue(pro_que.queue()).read(final_result).enq().expect("Error reading result data.");
            Box::into_raw(kde);
            *error = Error::NoError;
        }
        unsafe fn logdenominator_iterate_train_gaussian_high_memory(ckde:
                                                                        &Box<CKDE>,
                                                                    pro_que:
                                                                        &mut Box<ProQue>,
                                                                    x:
                                                                        *const DoubleNumpyArray,
                                                                    result:
                                                                        *mut c_double,
                                                                    tmp_coefficients:
                                                                        &Buffer<f64>,
                                                                    error:
                                                                        *mut Error) {
            let kde = Box::from_raw(ckde.kde);
            let test_shape = slice::from_raw_parts((*x).shape, 2);
            let m = test_shape[0];
            let d_test = test_shape[1];
            let n = kde.n;
            let max_work_size = get_max_work_size(&pro_que);
            let local_work_size =
                if n < max_work_size { n } else { max_work_size };
            let num_groups =
                (n as f32 / local_work_size as f32).ceil() as usize;
            let test_slice = slice::from_raw_parts((*x).ptr, m * d_test);
            let (test_instances_buffer,) =
                {
                    (match Buffer::builder().context(pro_que.context()).len(test_slice.len()).copy_host_slice(&test_slice).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },)
                };
            let (s1, s3, final_result_buffer, max_coefficients) =
                {
                    (match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m
                                                                                       *
                                                                                       num_groups).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     })
                };
            let a = 0.5 * (ckde.precision_variable + s2(ckde));
            let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);
            s1_s3_coefficients(ckde, pro_que, &test_instances_buffer,
                               test_leading_dimension as u32, test_rowmajor,
                               &s1, &s3, m);
            let kernel_onlygaussian_exponent_coefficients =
                pro_que.kernel_builder(onlygaussian_exponent_coefficients_iterate_train_high_memory_name(kde.rowmajor)).global_work_size(m
                                                                                                                                             *
                                                                                                                                             n).arg(&kde.training_data).arg(kde.leading_dimension
                                                                                                                                                                                as
                                                                                                                                                                                u32).arg(&ckde.precision).arg(&s1).arg((4.0
                                                                                                                                                                                                                            *
                                                                                                                                                                                                                            a).recip()).arg(&s3).arg(tmp_coefficients).arg(n
                                                                                                                                                                                                                                                                               as
                                                                                                                                                                                                                                                                               u32).build().expect("Kernel onlygaussian_exponent_coefficients build failed.");
            kernel_onlygaussian_exponent_coefficients.enq().expect("Error while executing kernel_onlygaussian_exponent_coefficients kernel.");
            max_gpu_mat(&pro_que, tmp_coefficients, &max_coefficients, m, n,
                        max_work_size, local_work_size, num_groups);
            let kernel_exp_and_sum_mat =
                pro_que.kernel_builder("exp_and_sum_mat").global_work_size(m *
                                                                               n).arg(tmp_coefficients).arg(&max_coefficients).arg(n
                                                                                                                                       as
                                                                                                                                       u32).arg(num_groups
                                                                                                                                                    as
                                                                                                                                                    u32).build().expect("Kernel exp_and_sum_mat build failed.");
            kernel_exp_and_sum_mat.enq().expect("Error while executing exp_and_sum_mat kernel.");
            sum_gpu_mat(&pro_que, tmp_coefficients, m, n, max_work_size,
                        local_work_size, num_groups);
            let kernel_log_and_sum =
                pro_que.kernel_builder("log_and_sum_mat").global_work_size(m).arg(&final_result_buffer).arg(tmp_coefficients).arg(&max_coefficients).arg(n
                                                                                                                                                             as
                                                                                                                                                             u32).arg(num_groups
                                                                                                                                                                          as
                                                                                                                                                                          u32).build().expect("Kernel log_and_sum_mat build failed.");
            kernel_log_and_sum.enq().expect("Error while executing kernel_log_and_sum kernel.");
            let kernel_sum_constant =
                pro_que.kernel_builder("sum_constant").global_work_size(m).arg(&final_result_buffer).arg(ckde.lognorm_factor).build().expect("Kernel sum_constant build failed.");
            kernel_sum_constant.enq().expect("Error while executing sum_constant kernel.");
            let final_result = slice::from_raw_parts_mut(result, m);
            final_result_buffer.cmd().queue(pro_que.queue()).read(final_result).enq().expect("Error reading result data.");
            Box::into_raw(kde);
            *error = Error::NoError;
        }
    }
    pub use denominator_gaussian::logdenominator_dataset_gaussian;
    mod denominator_onlykde {
        use crate::{empty_buffers, copy_buffers, Error, DoubleNumpyArray,
                    buffer_fill_value, get_max_work_size, is_rowmajor,
                    max_gpu_vec_copy, log_sum_gpu_vec, max_gpu_mat,
                    sum_gpu_mat};
        use crate::denominator::{CKDE};
        use std::f64;
        use std::slice;
        use ocl::{Buffer, ProQue};
        use libc::{c_double};
        #[no_mangle]
        pub unsafe extern "C" fn logdenominator_dataset_onlykde(ckde:
                                                                    *mut CKDE,
                                                                pro_que:
                                                                    *mut ProQue,
                                                                x:
                                                                    *const DoubleNumpyArray,
                                                                result:
                                                                    *mut c_double,
                                                                error:
                                                                    *mut Error) {
            *error = Error::NotFinished;
            let mut ckde = Box::from_raw(ckde);
            let mut pro_que = Box::from_raw(pro_que);
            let m = *(*x).shape;
            *error = Error::NoError;
            if (*ckde.kde).n >= m {
                logdenominator_iterate_test_onlykde(&mut ckde, &mut pro_que,
                                                    x, result, error);
            } else {
                logdenominator_iterate_train_onlykde(&mut ckde, &mut pro_que,
                                                     x, result, error);
            }
            Box::into_raw(ckde);
            Box::into_raw(pro_que);
        }
        fn substract_without_origin_name(train_rowmajor: bool,
                                         test_rowmajor: bool)
         -> &'static str {
            if train_rowmajor {
                if test_rowmajor {
                    "substract_without_origin_rowmajor_rowmajor"
                } else { "substract_without_origin_rowmajor_columnmajor" }
            } else {
                if test_rowmajor {
                    "substract_without_origin_columnmajor_rowmajor"
                } else { "substract_without_origin_columnmajor_columnmajor" }
            }
        }
        unsafe fn logdenominator_iterate_test_onlykde(ckde: &mut Box<CKDE>,
                                                      pro_que:
                                                          &mut Box<ProQue>,
                                                      x:
                                                          *const DoubleNumpyArray,
                                                      result: *mut c_double,
                                                      error: *mut Error) {
            let kde = Box::from_raw(ckde.kde);
            let m = *(*x).shape;
            let d = kde.d;
            let nparents_kde = d - 1;
            let n = kde.n;
            let max_work_size = get_max_work_size(&pro_que);
            let local_work_size =
                if n < max_work_size { n } else { max_work_size };
            let num_groups =
                (n as f32 / local_work_size as f32).ceil() as usize;
            let test_slice = slice::from_raw_parts((*x).ptr, m * d);
            let (test_instances_buffer,) =
                {
                    (match Buffer::builder().context(pro_que.context()).len(test_slice.len()).copy_host_slice(&test_slice).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },)
                };
            let (ti_buffer, final_result_buffer, coeffs, max_coefficients) =
                {
                    (match Buffer::<f64>::builder().context(pro_que.context()).len(n
                                                                                       *
                                                                                       nparents_kde).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(n).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(num_groups).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     })
                };
            let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);
            let kernel_substract =
                pro_que.kernel_builder(substract_without_origin_name(kde.rowmajor,
                                                                     test_rowmajor)).global_work_size(n
                                                                                                          *
                                                                                                          nparents_kde).arg(&kde.training_data).arg(kde.leading_dimension
                                                                                                                                                        as
                                                                                                                                                        u32).arg(&test_instances_buffer).arg(test_leading_dimension
                                                                                                                                                                                                 as
                                                                                                                                                                                                 u32).arg(&ti_buffer).arg_named("row",
                                                                                                                                                                                                                                &0u32).arg(nparents_kde
                                                                                                                                                                                                                                               as
                                                                                                                                                                                                                                               u32).build().expect("Kernel substract_without_origin build failed.");
            let kernel_exponent_coefficients =
                pro_que.kernel_builder("onlykde_exponent_coefficients_iterate_test").global_work_size(n
                                                                                                          *
                                                                                                          nparents_kde
                                                                                                          *
                                                                                                          nparents_kde).local_work_size(nparents_kde
                                                                                                                                            *
                                                                                                                                            nparents_kde).arg(&ti_buffer).arg(nparents_kde
                                                                                                                                                                                  as
                                                                                                                                                                                  u32).arg(&ckde.marginal_precision).arg(&coeffs).arg_local::<f64>(nparents_kde
                                                                                                                                                                                                                                                       *
                                                                                                                                                                                                                                                       nparents_kde).build().expect("Kernel onlykde_exponent_coefficients_iterate_test build failed.");
            let kernel_log_sum_gpu =
                pro_que.kernel_builder("copy_logpdf_result").global_work_size(1).arg(&coeffs).arg(&max_coefficients).arg(&final_result_buffer).arg_named("offset",
                                                                                                                                                         &0u32).build().expect("Kernel copy_logpdf_result build failed.");
            for i in 0..m {
                kernel_substract.set_arg("row", i as u32).unwrap();
                kernel_substract.enq().expect("Error while executing substract_without_origin kernel.");
                kernel_exponent_coefficients.enq().expect("Error while executing onlykde_exponent_coefficients_iterate_test kernel.");
                max_gpu_vec_copy(pro_que, &coeffs, &max_coefficients, n,
                                 max_work_size, local_work_size, num_groups);
                log_sum_gpu_vec(&pro_que, &coeffs, &max_coefficients, n,
                                max_work_size, local_work_size, num_groups);
                kernel_log_sum_gpu.set_arg("offset", i as u32).unwrap();
                kernel_log_sum_gpu.enq().expect("Error while executing copy_logpdf_result kernel.");
            }
            let kernel_sum_constant =
                pro_que.kernel_builder("sum_constant").global_work_size(m).arg(&final_result_buffer).arg(ckde.lognorm_factor).build().expect("Kernel sum_constant build failed.");
            kernel_sum_constant.enq().expect("Error while executing sum_constant kernel.");
            let final_result = slice::from_raw_parts_mut(result, m);
            final_result_buffer.cmd().queue(pro_que.queue()).read(final_result).enq().expect("Error reading result data.");
            Box::into_raw(kde);
            *error = Error::NoError;
        }
        unsafe fn logdenominator_iterate_train_onlykde(ckde: &mut Box<CKDE>,
                                                       pro_que:
                                                           &mut Box<ProQue>,
                                                       x:
                                                           *const DoubleNumpyArray,
                                                       result: *mut c_double,
                                                       error: *mut Error) {
            let m = *(*x).shape;
            let tmp_vec_buffer =
                Buffer::<f64>::builder().context(pro_que.context()).len(m *
                                                                            (*ckde.kde).n).build();
            match tmp_vec_buffer {
                Ok(b) =>
                logdenominator_iterate_train_high_memory_onlykde(ckde,
                                                                 pro_que, x,
                                                                 result, &b,
                                                                 error),
                Err(_) => {
                    logdenominator_iterate_train_low_memory_onlykde(ckde,
                                                                    pro_que,
                                                                    x, result,
                                                                    error);
                }
            }
        }
        unsafe fn logdenominator_iterate_train_low_memory_onlykde(ckde:
                                                                      &mut Box<CKDE>,
                                                                  pro_que:
                                                                      &mut Box<ProQue>,
                                                                  x:
                                                                      *const DoubleNumpyArray,
                                                                  result:
                                                                      *mut c_double,
                                                                  error:
                                                                      *mut Error) {
            let kde = Box::from_raw(ckde.kde);
            let m = *(*x).shape;
            let d = kde.d;
            let nparents_kde = d - 1;
            let n = kde.n;
            let test_slice = slice::from_raw_parts((*x).ptr, m * d);
            let (test_instances_buffer,) =
                {
                    (match Buffer::builder().context(pro_que.context()).len(test_slice.len()).copy_host_slice(&test_slice).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },)
                };
            let (ti_buffer, final_result_buffer, max_buffer) =
                {
                    (match Buffer::<f64>::builder().context(pro_que.context()).len(m
                                                                                       *
                                                                                       nparents_kde).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     })
                };
            buffer_fill_value(&pro_que, &max_buffer, m, f64::MIN);
            buffer_fill_value(&pro_que, &final_result_buffer, m, 0.0f64);
            let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);
            let kernel_substract =
                pro_que.kernel_builder(substract_without_origin_name(kde.rowmajor,
                                                                     test_rowmajor)).global_work_size(m
                                                                                                          *
                                                                                                          nparents_kde).arg(&test_instances_buffer).arg(test_leading_dimension
                                                                                                                                                            as
                                                                                                                                                            u32).arg(&kde.training_data).arg(kde.leading_dimension
                                                                                                                                                                                                 as
                                                                                                                                                                                                 u32).arg(&ti_buffer).arg_named("row",
                                                                                                                                                                                                                                &0u32).arg(nparents_kde
                                                                                                                                                                                                                                               as
                                                                                                                                                                                                                                               u32).build().expect("Kernel substract_without_origin build failed.");
            let kernel_coefficients_checkmax =
                pro_que.kernel_builder("onlykde_exponent_coefficients_iterate_train_low_memory_checkmax").global_work_size(m
                                                                                                                               *
                                                                                                                               nparents_kde
                                                                                                                               *
                                                                                                                               nparents_kde).local_work_size(nparents_kde
                                                                                                                                                                 *
                                                                                                                                                                 nparents_kde).arg(&ti_buffer).arg(nparents_kde
                                                                                                                                                                                                       as
                                                                                                                                                                                                       u32).arg(&ckde.marginal_precision).arg(&max_buffer).arg_local::<f64>(nparents_kde
                                                                                                                                                                                                                                                                                *
                                                                                                                                                                                                                                                                                nparents_kde).arg(n
                                                                                                                                                                                                                                                                                                      as
                                                                                                                                                                                                                                                                                                      u32).build().expect("Kernel onlykde_exponent_coefficients_iterate_train_low_memory_checkmax build failed.");
            let kernel_coefficients_compute =
                pro_que.kernel_builder("onlykde_exponent_coefficients_iterate_train_low_memory_compute").global_work_size(m
                                                                                                                              *
                                                                                                                              nparents_kde
                                                                                                                              *
                                                                                                                              nparents_kde).local_work_size(nparents_kde
                                                                                                                                                                *
                                                                                                                                                                nparents_kde).arg(&ti_buffer).arg(nparents_kde
                                                                                                                                                                                                      as
                                                                                                                                                                                                      u32).arg(&ckde.marginal_precision).arg(&final_result_buffer).arg(&max_buffer).arg_local::<f64>(nparents_kde
                                                                                                                                                                                                                                                                                                         *
                                                                                                                                                                                                                                                                                                         nparents_kde).arg(n
                                                                                                                                                                                                                                                                                                                               as
                                                                                                                                                                                                                                                                                                                               u32).build().expect("Kernel onlykde_exponent_coefficients_iterate_train_low_memory_compute build failed.");
            let kernel_log_and_sum =
                pro_que.kernel_builder("log_and_sum").global_work_size(m).arg(&final_result_buffer).arg(&max_buffer).build().expect("Kernel log_and_sum build failed.");
            for i in 0..n {
                kernel_substract.set_arg("row", i as u32).unwrap();
                kernel_substract.enq().expect("Error while executing substract_without_origin kernel.");
                kernel_coefficients_checkmax.enq().expect("Error while executing onlykde_exponent_coefficients_iterate_train_low_memory_checkmax kernel.");
            }
            for i in 0..n {
                kernel_substract.set_arg("row", i as u32).unwrap();
                kernel_substract.enq().expect("Error while executing substract_without_origin kernel.");
                kernel_coefficients_compute.enq().expect("Error while executing onlykde_exponent_coefficients_iterate_train_low_memory_compute kernel.");
            }
            kernel_log_and_sum.enq().expect("Error while executing kernel_log_and_sum kernel.");
            let kernel_sum_constant =
                pro_que.kernel_builder("sum_constant").global_work_size(m).arg(&final_result_buffer).arg(ckde.lognorm_factor).build().expect("Kernel sum_constant build failed.");
            kernel_sum_constant.enq().expect("Error while executing sum_constant kernel.");
            let final_result = slice::from_raw_parts_mut(result, m);
            final_result_buffer.cmd().queue(pro_que.queue()).read(final_result).enq().expect("Error reading result data.");
            Box::into_raw(kde);
            *error = Error::NoError;
        }
        unsafe fn logdenominator_iterate_train_high_memory_onlykde(ckde:
                                                                       &mut Box<CKDE>,
                                                                   pro_que:
                                                                       &mut Box<ProQue>,
                                                                   x:
                                                                       *const DoubleNumpyArray,
                                                                   result:
                                                                       *mut c_double,
                                                                   coeffs:
                                                                       &Buffer<f64>,
                                                                   error:
                                                                       *mut Error) {
            let kde = Box::from_raw(ckde.kde);
            let m = *(*x).shape;
            let d = kde.d;
            let nparents_kde = d - 1;
            let n = kde.n;
            let max_work_size = get_max_work_size(&pro_que);
            let local_work_size =
                if n < max_work_size { n } else { max_work_size };
            let num_groups =
                (n as f32 / local_work_size as f32).ceil() as usize;
            let test_slice = slice::from_raw_parts((*x).ptr, m * d);
            let (test_instances_buffer,) =
                {
                    (match Buffer::builder().context(pro_que.context()).len(test_slice.len()).copy_host_slice(&test_slice).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },)
                };
            let (ti_buffer, final_result_buffer, max_buffer) =
                {
                    (match Buffer::<f64>::builder().context(pro_que.context()).len(m
                                                                                       *
                                                                                       nparents_kde).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m
                                                                                       *
                                                                                       num_groups).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     })
                };
            let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);
            let kernel_substract =
                pro_que.kernel_builder(substract_without_origin_name(test_rowmajor,
                                                                     kde.rowmajor)).global_work_size(m
                                                                                                         *
                                                                                                         nparents_kde).arg(&test_instances_buffer).arg(test_leading_dimension
                                                                                                                                                           as
                                                                                                                                                           u32).arg(&kde.training_data).arg(kde.leading_dimension
                                                                                                                                                                                                as
                                                                                                                                                                                                u32).arg(&ti_buffer).arg_named("row",
                                                                                                                                                                                                                               &0u32).arg(nparents_kde
                                                                                                                                                                                                                                              as
                                                                                                                                                                                                                                              u32).build().expect("Kernel substract_without_origin build failed.");
            let kernel_exponent_coefficients =
                pro_que.kernel_builder("onlykde_exponent_coefficients_iterate_train_high_memory").global_work_size(m
                                                                                                                       *
                                                                                                                       nparents_kde
                                                                                                                       *
                                                                                                                       nparents_kde).local_work_size(nparents_kde
                                                                                                                                                         *
                                                                                                                                                         nparents_kde).arg(&ti_buffer).arg(nparents_kde
                                                                                                                                                                                               as
                                                                                                                                                                                               u32).arg(&ckde.marginal_precision).arg(coeffs).arg_local::<f64>(nparents_kde
                                                                                                                                                                                                                                                                   *
                                                                                                                                                                                                                                                                   nparents_kde).arg_named("offset",
                                                                                                                                                                                                                                                                                           &0u32).arg(n
                                                                                                                                                                                                                                                                                                          as
                                                                                                                                                                                                                                                                                                          u32).build().expect("Kernel onlykde_exponent_coefficients_iterate_test build failed.");
            let kernel_exp_and_sum_mat =
                pro_que.kernel_builder("exp_and_sum_mat").global_work_size(m *
                                                                               n).arg(coeffs).arg(&max_buffer).arg(n
                                                                                                                       as
                                                                                                                       u32).arg(num_groups
                                                                                                                                    as
                                                                                                                                    u32).build().expect("Kernel exp_and_sum_mat build failed.");
            let kernel_log_and_sum =
                pro_que.kernel_builder("log_and_sum_mat").global_work_size(m).arg(&final_result_buffer).arg(coeffs).arg(&max_buffer).arg(n
                                                                                                                                             as
                                                                                                                                             u32).arg(num_groups
                                                                                                                                                          as
                                                                                                                                                          u32).build().expect("Kernel log_and_sum_mat build failed.");
            for i in 0..n {
                kernel_substract.set_arg("row", i as u32).unwrap();
                kernel_substract.enq().expect("Error while executing substract_without_origin kernel.");
                kernel_exponent_coefficients.set_arg("offset",
                                                     i as u32).unwrap();
                kernel_exponent_coefficients.enq().expect("Error while executing onlykde_exponent_coefficients_iterate_train_high_memory kernel.");
            }
            max_gpu_mat(&pro_que, coeffs, &max_buffer, m, n, max_work_size,
                        local_work_size, num_groups);
            kernel_exp_and_sum_mat.enq().expect("Error while executing exp_and_sum_mat kernel.");
            sum_gpu_mat(&pro_que, coeffs, m, n, max_work_size,
                        local_work_size, num_groups);
            kernel_log_and_sum.enq().expect("Error while executing kernel_log_and_sum kernel.");
            let kernel_sum_constant =
                pro_que.kernel_builder("sum_constant").global_work_size(m).arg(&final_result_buffer).arg(ckde.lognorm_factor).build().expect("Kernel sum_constant build failed.");
            kernel_sum_constant.enq().expect("Error while executing sum_constant kernel.");
            let final_result = slice::from_raw_parts_mut(result, m);
            final_result_buffer.cmd().queue(pro_que.queue()).read(final_result).enq().expect("Error reading result data.");
            Box::into_raw(kde);
            *error = Error::NoError;
        }
    }
    pub use denominator_onlykde::logdenominator_dataset_onlykde;
    mod denominator_mix {
        use crate::{empty_buffers, copy_buffers, Error, DoubleNumpyArray,
                    buffer_fill_value, get_max_work_size, is_rowmajor,
                    max_gpu_vec_copy, log_sum_gpu_vec, max_gpu_mat,
                    sum_gpu_mat};
        use crate::denominator::{CKDE, s1_s3_coefficients, s2};
        use std::slice;
        use std::f64;
        use ocl::{Buffer, ProQue};
        use libc::{c_double};
        fn substract_without_origin_from_indices_iterate_test_name(train_rowmajor:
                                                                       bool,
                                                                   test_rowmajor:
                                                                       bool)
         -> &'static str {
            if train_rowmajor {
                if test_rowmajor {
                    "substract_without_origin_from_indices_iterate_test_rowmajor_rowmajor"
                } else {
                    "substract_without_origin_from_indices_iterate_test_rowmajor_columnmajor"
                }
            } else {
                if test_rowmajor {
                    "substract_without_origin_from_indices_iterate_test_columnmajor_rowmajor"
                } else {
                    "substract_without_origin_from_indices_iterate_test_columnmajor_columnmajor"
                }
            }
        }
        fn substract_without_origin_from_indices_iterate_train_name(train_rowmajor:
                                                                        bool,
                                                                    test_rowmajor:
                                                                        bool)
         -> &'static str {
            if train_rowmajor {
                if test_rowmajor {
                    "substract_without_origin_from_indices_iterate_train_rowmajor_rowmajor"
                } else {
                    "substract_without_origin_from_indices_iterate_train_rowmajor_columnmajor"
                }
            } else {
                if test_rowmajor {
                    "substract_without_origin_from_indices_iterate_train_columnmajor_rowmajor"
                } else {
                    "substract_without_origin_from_indices_iterate_train_columnmajor_columnmajor"
                }
            }
        }
        fn exponent_coefficients_iterate_test_name(train_rowmajor: bool)
         -> &'static str {
            if train_rowmajor {
                "exponent_coefficients_iterate_test_rowmajor"
            } else { "exponent_coefficients_iterate_test_columnmajor" }
        }
        fn exponent_coefficients_iterate_train_high_memory_name(train_rowmajor:
                                                                    bool)
         -> &'static str {
            if train_rowmajor {
                "exponent_coefficients_iterate_train_high_memory_rowmajor"
            } else {
                "exponent_coefficients_iterate_train_high_memory_columnmajor"
            }
        }
        fn exponent_coefficients_iterate_train_low_memory_checkmax_name(train_rowmajor:
                                                                            bool)
         -> &'static str {
            if train_rowmajor {
                "exponent_coefficients_iterate_train_low_memory_checkmax_rowmajor"
            } else {
                "exponent_coefficients_iterate_train_low_memory_checkmax_columnmajor"
            }
        }
        fn exponent_coefficients_iterate_train_low_memory_compute_name(train_rowmajor:
                                                                           bool)
         -> &'static str {
            if train_rowmajor {
                "exponent_coefficients_iterate_train_low_memory_compute_rowmajor"
            } else {
                "exponent_coefficients_iterate_train_low_memory_compute_columnmajor"
            }
        }
        #[no_mangle]
        pub unsafe extern "C" fn logdenominator_dataset(ckde: *mut CKDE,
                                                        pro_que: *mut ProQue,
                                                        x:
                                                            *const DoubleNumpyArray,
                                                        result: *mut c_double,
                                                        error: *mut Error) {
            *error = Error::NotFinished;
            let mut ckde = Box::from_raw(ckde);
            let mut pro_que = Box::from_raw(pro_que);
            let m = *(*x).shape;
            *error = Error::NoError;
            if (*ckde.kde).n >= m {
                logdenominator_iterate_test(&mut ckde, &mut pro_que, x,
                                            result, error);
            } else {
                logdenominator_iterate_train(&mut ckde, &mut pro_que, x,
                                             result, error);
            }
            Box::into_raw(ckde);
            Box::into_raw(pro_que);
        }
        unsafe fn logdenominator_iterate_test(ckde: &Box<CKDE>,
                                              pro_que: &mut Box<ProQue>,
                                              x: *const DoubleNumpyArray,
                                              result: *mut c_double,
                                              error: *mut Error) {
            let kde = Box::from_raw(ckde.kde);
            let test_shape = slice::from_raw_parts((*x).shape, 2);
            let m = test_shape[0];
            let d_test = test_shape[1];
            let nparents_kde = kde.d - 1;
            let n = kde.n;
            let max_work_size = get_max_work_size(&pro_que);
            let local_work_size =
                if n < max_work_size { n } else { max_work_size };
            let num_groups =
                (n as f32 / local_work_size as f32).ceil() as usize;
            let test_slice = slice::from_raw_parts((*x).ptr, m * d_test);
            let (test_instances_buffer,) =
                {
                    (match Buffer::builder().context(pro_que.context()).len(test_slice.len()).copy_host_slice(&test_slice).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },)
                };
            let (s1, s3, final_result_buffer, mahalanobis_buffer,
                 dotproduct_buffer, ti_buffer, max_coefficients) =
                {
                    (match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(n).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(n).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(n
                                                                                       *
                                                                                       nparents_kde).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(num_groups).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     })
                };
            let a = 0.5 * (ckde.precision_variable + s2(ckde));
            let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);
            s1_s3_coefficients(ckde, pro_que, &test_instances_buffer,
                               test_leading_dimension as u32, test_rowmajor,
                               &s1, &s3, m);
            let kernel_substract =
                pro_que.kernel_builder(substract_without_origin_from_indices_iterate_test_name(kde.rowmajor,
                                                                                               test_rowmajor)).global_work_size(n
                                                                                                                                    *
                                                                                                                                    nparents_kde).arg(&kde.training_data).arg(kde.leading_dimension
                                                                                                                                                                                  as
                                                                                                                                                                                  u32).arg(&test_instances_buffer).arg(test_leading_dimension
                                                                                                                                                                                                                           as
                                                                                                                                                                                                                           u32).arg(&ti_buffer).arg_named("row",
                                                                                                                                                                                                                                                          &0u32).arg(nparents_kde
                                                                                                                                                                                                                                                                         as
                                                                                                                                                                                                                                                                         u32).arg(ckde.kde_indices.as_ref().unwrap()).build().expect("Kernel substract_without_origin_from_indices build failed.");
            let kernel_mahalanobis =
                pro_que.kernel_builder("mahalanobis").global_work_size(n *
                                                                           nparents_kde
                                                                           *
                                                                           nparents_kde).local_work_size(nparents_kde
                                                                                                             *
                                                                                                             nparents_kde).arg(&ti_buffer).arg(&ckde.precision).arg(&mahalanobis_buffer).arg_local::<f64>(nparents_kde
                                                                                                                                                                                                              *
                                                                                                                                                                                                              nparents_kde).arg(nparents_kde
                                                                                                                                                                                                                                    as
                                                                                                                                                                                                                                    u32).build().expect("Kernel mahalanobis build failed.");
            let kernel_dotproduct =
                pro_que.kernel_builder("dotproduct").global_work_size(n *
                                                                          nparents_kde).local_work_size(nparents_kde).arg(&ti_buffer).arg(&ckde.precision).arg(&dotproduct_buffer).arg_local::<f64>(nparents_kde).arg(nparents_kde
                                                                                                                                                                                                                          as
                                                                                                                                                                                                                          u32).build().expect("Kernel dotproduct build failed.");
            let kernel_coefficients =
                pro_que.kernel_builder(exponent_coefficients_iterate_test_name(kde.rowmajor)).global_work_size(n).arg(&kde.training_data).arg(kde.leading_dimension
                                                                                                                                                  as
                                                                                                                                                  u32).arg(&ckde.precision).arg(&mahalanobis_buffer).arg(&dotproduct_buffer).arg(&s1).arg((4.0f64
                                                                                                                                                                                                                                               *
                                                                                                                                                                                                                                               a).recip()).arg(&s3).arg_named("offset",
                                                                                                                                                                                                                                                                              &0u32).build().expect("Kernel exponent_coefficients_iterate_test build failed.");
            let kernel_log_sum_gpu =
                pro_que.kernel_builder("copy_logpdf_result").global_work_size(1).arg(&mahalanobis_buffer).arg(&max_coefficients).arg(&final_result_buffer).arg_named("offset",
                                                                                                                                                                     &0u32).build().expect("Kernel copy_logpdf_result build failed.");
            for i in 0..m {
                kernel_substract.set_arg("row", i as u32).unwrap();
                kernel_substract.enq().expect("Error while executing substract_without_origin_from_indices kernel.");
                kernel_mahalanobis.enq().expect("Error while executing mahalanobis kernel.");
                kernel_dotproduct.enq().expect("Error while executing dotproduct kernel.");
                kernel_coefficients.set_arg("offset", i as u32).unwrap();
                kernel_coefficients.enq().expect("Error while executing exponent_coefficients_iterate_test kernel.");
                max_gpu_vec_copy(pro_que, &mahalanobis_buffer,
                                 &max_coefficients, n, max_work_size,
                                 local_work_size, num_groups);
                log_sum_gpu_vec(&pro_que, &mahalanobis_buffer,
                                &max_coefficients, n, max_work_size,
                                local_work_size, num_groups);
                kernel_log_sum_gpu.set_arg("offset", i as u32).unwrap();
                kernel_log_sum_gpu.enq().expect("Error while executing copy_logpdf_result kernel.");
            }
            let kernel_sum_constant =
                pro_que.kernel_builder("sum_constant").global_work_size(m).arg(&final_result_buffer).arg(ckde.lognorm_factor).build().expect("Kernel sum_constant build failed.");
            kernel_sum_constant.enq().expect("Error while executing sum_constant kernel.");
            let final_result = slice::from_raw_parts_mut(result, m);
            final_result_buffer.cmd().queue(pro_que.queue()).read(final_result).enq().expect("Error reading result data.");
            Box::into_raw(kde);
            *error = Error::NoError;
        }
        unsafe fn logdenominator_iterate_train(ckde: &Box<CKDE>,
                                               pro_que: &mut Box<ProQue>,
                                               x: *const DoubleNumpyArray,
                                               result: *mut c_double,
                                               error: *mut Error) {
            let m = *(*x).shape;
            let tmp_vec_buffer =
                Buffer::<f64>::builder().context(pro_que.context()).len(m *
                                                                            (*ckde.kde).n).build();
            match tmp_vec_buffer {
                Ok(b) => {
                    logdenominator_iterate_train_high_memory(ckde, pro_que, x,
                                                             result, &b,
                                                             error)
                }
                Err(_) => {
                    logdenominator_iterate_train_low_memory(ckde, pro_que, x,
                                                            result, error);
                }
            }
        }
        unsafe fn logdenominator_iterate_train_low_memory(ckde: &Box<CKDE>,
                                                          pro_que:
                                                              &mut Box<ProQue>,
                                                          x:
                                                              *const DoubleNumpyArray,
                                                          result:
                                                              *mut c_double,
                                                          error: *mut Error) {
            let kde = Box::from_raw(ckde.kde);
            let test_shape = slice::from_raw_parts((*x).shape, 2);
            let m = test_shape[0];
            let d_test = test_shape[1];
            let nparents_kde = kde.d - 1;
            let n = kde.n;
            let test_slice = slice::from_raw_parts((*x).ptr, m * d_test);
            let (test_instances_buffer,) =
                {
                    (match Buffer::builder().context(pro_que.context()).len(test_slice.len()).copy_host_slice(&test_slice).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },)
                };
            let (s1, s3, final_result_buffer, ti_buffer, mahalanobis_buffer,
                 dotproduct_buffer, max_buffer) =
                {
                    (match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m
                                                                                       *
                                                                                       nparents_kde).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     })
                };
            buffer_fill_value(&pro_que, &max_buffer, m, f64::MIN);
            buffer_fill_value(&pro_que, &final_result_buffer, m, 0.0f64);
            let a = 0.5 * (ckde.precision_variable + s2(ckde));
            let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);
            s1_s3_coefficients(ckde, pro_que, &test_instances_buffer,
                               test_leading_dimension as u32, test_rowmajor,
                               &s1, &s3, m);
            let kernel_substract =
                pro_que.kernel_builder(substract_without_origin_from_indices_iterate_train_name(kde.rowmajor,
                                                                                                test_rowmajor)).global_work_size(m
                                                                                                                                     *
                                                                                                                                     nparents_kde).arg(&kde.training_data).arg(kde.leading_dimension
                                                                                                                                                                                   as
                                                                                                                                                                                   u32).arg(&test_instances_buffer).arg(test_leading_dimension
                                                                                                                                                                                                                            as
                                                                                                                                                                                                                            u32).arg(&ti_buffer).arg_named("row",
                                                                                                                                                                                                                                                           &0u32).arg(nparents_kde
                                                                                                                                                                                                                                                                          as
                                                                                                                                                                                                                                                                          u32).arg(ckde.kde_indices.as_ref().unwrap()).build().expect("Kernel substract_without_origin_from_indices build failed.");
            let kernel_mahalanobis =
                pro_que.kernel_builder("mahalanobis").global_work_size(m *
                                                                           nparents_kde
                                                                           *
                                                                           nparents_kde).local_work_size(nparents_kde
                                                                                                             *
                                                                                                             nparents_kde).arg(&ti_buffer).arg(&ckde.precision).arg(&mahalanobis_buffer).arg_local::<f64>(nparents_kde
                                                                                                                                                                                                              *
                                                                                                                                                                                                              nparents_kde).arg(nparents_kde
                                                                                                                                                                                                                                    as
                                                                                                                                                                                                                                    u32).build().expect("Kernel mahalanobis build failed.");
            let kernel_dotproduct =
                pro_que.kernel_builder("dotproduct").global_work_size(m *
                                                                          nparents_kde).local_work_size(nparents_kde).arg(&ti_buffer).arg(&ckde.precision).arg(&dotproduct_buffer).arg_local::<f64>(nparents_kde).arg(nparents_kde
                                                                                                                                                                                                                          as
                                                                                                                                                                                                                          u32).build().expect("Kernel dotproduct build failed.");
            let kernel_coefficients_checkmax =
                pro_que.kernel_builder(exponent_coefficients_iterate_train_low_memory_checkmax_name(kde.rowmajor)).global_work_size(m).arg(&kde.training_data).arg(kde.leading_dimension
                                                                                                                                                                       as
                                                                                                                                                                       u32).arg(&ckde.precision).arg(&mahalanobis_buffer).arg(&max_buffer).arg(&dotproduct_buffer).arg(&s1).arg((4.0
                                                                                                                                                                                                                                                                                     *
                                                                                                                                                                                                                                                                                     a).recip()).arg(&s3).arg_named("offset",
                                                                                                                                                                                                                                                                                                                    &0u32).build().expect("Kernel exponent_coefficients_iterate_train_low_memory_checkmax build failed.");
            let kernel_coefficients_compute =
                pro_que.kernel_builder(exponent_coefficients_iterate_train_low_memory_compute_name(kde.rowmajor)).global_work_size(m).arg(&kde.training_data).arg(kde.leading_dimension
                                                                                                                                                                      as
                                                                                                                                                                      u32).arg(&ckde.precision).arg(&mahalanobis_buffer).arg(&final_result_buffer).arg(&max_buffer).arg(&dotproduct_buffer).arg(&s1).arg((4.0
                                                                                                                                                                                                                                                                                                              *
                                                                                                                                                                                                                                                                                                              a).recip()).arg(&s3).arg_named("offset",
                                                                                                                                                                                                                                                                                                                                             &0u32).build().expect("Kernel exponent_coefficients_iterate_train_low_memory_compute build failed.");
            let kernel_log_and_sum =
                pro_que.kernel_builder("log_and_sum").global_work_size(m).arg(&final_result_buffer).arg(&max_buffer).build().expect("Kernel log_and_sum build failed.");
            for i in 0..n {
                kernel_substract.set_arg("row", i as u32).unwrap();
                kernel_substract.enq().expect("Error while executing substract_without_origin_from_indices kernel.");
                kernel_mahalanobis.enq().expect("Error while executing mahalanobis kernel.");
                kernel_dotproduct.enq().expect("Error while executing dotproduct kernel.");
                kernel_coefficients_checkmax.set_arg("offset",
                                                     i as u32).unwrap();
                kernel_coefficients_checkmax.enq().expect("Error while executing exponent_coefficients_iterate_train_low_memory_checkmax kernel.");
            }
            for i in 0..n {
                kernel_substract.set_arg("row", i as u32).unwrap();
                kernel_substract.enq().expect("Error while executing substract_without_origin_from_indices kernel.");
                kernel_mahalanobis.enq().expect("Error while executing mahalanobis kernel.");
                kernel_dotproduct.enq().expect("Error while executing dotproduct kernel.");
                kernel_coefficients_compute.set_arg("offset",
                                                    i as u32).unwrap();
                kernel_coefficients_compute.enq().expect("Error while executing exponent_coefficients_iterate_train_low_memory_compute kernel.");
            }
            kernel_log_and_sum.enq().expect("Error while executing kernel_log_and_sum kernel.");
            let kernel_sum_constant =
                pro_que.kernel_builder("sum_constant").global_work_size(m).arg(&final_result_buffer).arg(ckde.lognorm_factor).build().expect("Kernel sum_constant build failed.");
            kernel_sum_constant.enq().expect("Error while executing sum_constant kernel.");
            let final_result = slice::from_raw_parts_mut(result, m);
            final_result_buffer.cmd().queue(pro_que.queue()).read(final_result).enq().expect("Error reading result data.");
            Box::into_raw(kde);
            *error = Error::NoError;
        }
        unsafe fn logdenominator_iterate_train_high_memory(ckde: &Box<CKDE>,
                                                           pro_que:
                                                               &mut Box<ProQue>,
                                                           x:
                                                               *const DoubleNumpyArray,
                                                           result:
                                                               *mut c_double,
                                                           coeffs:
                                                               &Buffer<f64>,
                                                           error:
                                                               *mut Error) {
            let kde = Box::from_raw(ckde.kde);
            let test_shape = slice::from_raw_parts((*x).shape, 2);
            let m = test_shape[0];
            let d_test = test_shape[1];
            let nparents_kde = kde.d - 1;
            let n = kde.n;
            let max_work_size = get_max_work_size(&pro_que);
            let local_work_size =
                if n < max_work_size { n } else { max_work_size };
            let num_groups =
                (n as f32 / local_work_size as f32).ceil() as usize;
            let test_slice = slice::from_raw_parts((*x).ptr, m * d_test);
            let (test_instances_buffer,) =
                {
                    (match Buffer::builder().context(pro_que.context()).len(test_slice.len()).copy_host_slice(&test_slice).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },)
                };
            let (s1, s3, final_result_buffer, ti_buffer, dotproduct_buffer,
                 max_buffer) =
                {
                    (match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m
                                                                                       *
                                                                                       nparents_kde).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },
                     match Buffer::<f64>::builder().context(pro_que.context()).len(m
                                                                                       *
                                                                                       num_groups).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     })
                };
            let a = 0.5 * (ckde.precision_variable + s2(ckde));
            let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);
            s1_s3_coefficients(ckde, pro_que, &test_instances_buffer,
                               test_leading_dimension as u32, test_rowmajor,
                               &s1, &s3, m);
            let kernel_substract =
                pro_que.kernel_builder(substract_without_origin_from_indices_iterate_train_name(kde.rowmajor,
                                                                                                test_rowmajor)).global_work_size(m
                                                                                                                                     *
                                                                                                                                     nparents_kde).arg(&kde.training_data).arg(kde.leading_dimension
                                                                                                                                                                                   as
                                                                                                                                                                                   u32).arg(&test_instances_buffer).arg(test_leading_dimension
                                                                                                                                                                                                                            as
                                                                                                                                                                                                                            u32).arg(&ti_buffer).arg_named("row",
                                                                                                                                                                                                                                                           &0u32).arg(nparents_kde
                                                                                                                                                                                                                                                                          as
                                                                                                                                                                                                                                                                          u32).arg(ckde.kde_indices.as_ref().unwrap()).build().expect("Kernel substract_without_origin_from_indices build failed.");
            let kernel_mahalanobis =
                pro_que.kernel_builder("mahalanobis_mat").global_work_size(m *
                                                                               nparents_kde
                                                                               *
                                                                               nparents_kde).local_work_size(nparents_kde
                                                                                                                 *
                                                                                                                 nparents_kde).arg(&ti_buffer).arg(&ckde.precision).arg(coeffs).arg_local::<f64>(nparents_kde
                                                                                                                                                                                                     *
                                                                                                                                                                                                     nparents_kde).arg(nparents_kde
                                                                                                                                                                                                                           as
                                                                                                                                                                                                                           u32).arg_named("offset",
                                                                                                                                                                                                                                          &0u32).arg(n
                                                                                                                                                                                                                                                         as
                                                                                                                                                                                                                                                         u32).build().expect("Kernel mahalanobis_mat build failed.");
            let kernel_dotproduct =
                pro_que.kernel_builder("dotproduct").global_work_size(m *
                                                                          nparents_kde).local_work_size(nparents_kde).arg(&ti_buffer).arg(&ckde.precision).arg(&dotproduct_buffer).arg_local::<f64>(nparents_kde).arg(nparents_kde
                                                                                                                                                                                                                          as
                                                                                                                                                                                                                          u32).build().expect("Kernel dotproduct build failed.");
            let kernel_coefficients =
                pro_que.kernel_builder(exponent_coefficients_iterate_train_high_memory_name(kde.rowmajor)).global_work_size(m).arg(&kde.training_data).arg(kde.leading_dimension
                                                                                                                                                               as
                                                                                                                                                               u32).arg(&ckde.precision).arg(coeffs).arg(&dotproduct_buffer).arg(&s1).arg((4.0
                                                                                                                                                                                                                                               *
                                                                                                                                                                                                                                               a).recip()).arg(&s3).arg_named("offset",
                                                                                                                                                                                                                                                                              &0u32).arg(n
                                                                                                                                                                                                                                                                                             as
                                                                                                                                                                                                                                                                                             u32).build().expect("Kernel exponent_coefficients_iterate_train_high_memory build failed.");
            let kernel_exp_and_sum_mat =
                pro_que.kernel_builder("exp_and_sum_mat").global_work_size(m *
                                                                               n).arg(coeffs).arg(&max_buffer).arg(n
                                                                                                                       as
                                                                                                                       u32).arg(num_groups
                                                                                                                                    as
                                                                                                                                    u32).build().expect("Kernel exp_and_sum_mat build failed.");
            let kernel_log_and_sum =
                pro_que.kernel_builder("log_and_sum_mat").global_work_size(m).arg(&final_result_buffer).arg(coeffs).arg(&max_buffer).arg(n
                                                                                                                                             as
                                                                                                                                             u32).arg(num_groups
                                                                                                                                                          as
                                                                                                                                                          u32).build().expect("Kernel log_and_sum_mat build failed.");
            for i in 0..n {
                kernel_substract.set_arg("row", i as u32).unwrap();
                kernel_substract.enq().expect("Error while executing substract_without_origin_from_indices kernel.");
                kernel_mahalanobis.set_arg("offset", i as u32).unwrap();
                kernel_mahalanobis.enq().expect("Error while executing mahalanobis_mat kernel.");
                kernel_dotproduct.enq().expect("Error while executing dotproduct kernel.");
                kernel_coefficients.set_arg("offset", i as u32).unwrap();
                kernel_coefficients.enq().expect("Error while executing exponent_coefficients_iterate_train_high_memory kernel.");
            }
            max_gpu_mat(&pro_que, coeffs, &max_buffer, m, n, max_work_size,
                        local_work_size, num_groups);
            kernel_exp_and_sum_mat.enq().expect("Error while executing exp_and_sum_mat kernel.");
            sum_gpu_mat(&pro_que, coeffs, m, n, max_work_size,
                        local_work_size, num_groups);
            kernel_log_and_sum.enq().expect("Error while executing kernel_log_and_sum kernel.");
            let kernel_sum_constant =
                pro_que.kernel_builder("sum_constant").global_work_size(m).arg(&final_result_buffer).arg(ckde.lognorm_factor).build().expect("Kernel sum_constant build failed.");
            kernel_sum_constant.enq().expect("Error while executing sum_constant kernel.");
            let final_result = slice::from_raw_parts_mut(result, m);
            final_result_buffer.cmd().queue(pro_que.queue()).read(final_result).enq().expect("Error reading result data.");
            Box::into_raw(kde);
            *error = Error::NoError;
        }
    }
    pub use denominator_mix::logdenominator_dataset;
    pub struct GaussianRegression {
        variable_index: c_uint,
        beta: Buffer<f64>,
        variable_beta: c_double,
        evidence_index: Option<Buffer<u32>>,
        nparents: c_uint,
        variance: c_double,
    }
    #[no_mangle]
    pub unsafe extern "C" fn gaussian_regression_init(pro_que: *mut ProQue,
                                                      variable_index: c_uint,
                                                      beta: *mut c_double,
                                                      evidence_index:
                                                          *mut c_uint,
                                                      nparents: c_uint,
                                                      variance: c_double,
                                                      error: *mut Error)
     -> *mut GaussianRegression {
        *error = Error::NotFinished;
        let pro_que = Box::from_raw(pro_que);
        let beta_slice = slice::from_raw_parts(beta, (nparents + 1) as usize);
        let (beta_buffer,) =
            {
                (match Buffer::builder().context(pro_que.context()).len(beta_slice.len()).copy_host_slice(&beta_slice).build()
                     {
                     Ok(b) => b,
                     Err(_) => {
                         *error = Error::MemoryError;
                         return ptr::null_mut();
                     }
                 },)
            };
        let evidence_index_buffer =
            if nparents > 1 {
                let evidence_index_slice =
                    slice::from_raw_parts(evidence_index,
                                          (nparents - 1) as usize);
                let (tmp,) =
                    {
                        (match Buffer::builder().context(pro_que.context()).len(evidence_index_slice.len()).copy_host_slice(&evidence_index_slice).build()
                             {
                             Ok(b) => b,
                             Err(_) => {
                                 *error = Error::MemoryError;
                                 return ptr::null_mut();
                             }
                         },)
                    };
                Some(tmp)
            } else { None };
        let gr =
            Box::new(GaussianRegression{variable_index: variable_index,
                                        beta: beta_buffer,
                                        variable_beta: beta_slice[1],
                                        evidence_index: evidence_index_buffer,
                                        nparents: nparents,
                                        variance: variance,});
        let ptr_gr = Box::into_raw(gr);
        Box::into_raw(pro_que);
        *error = Error::NoError;
        ptr_gr
    }
    #[no_mangle]
    pub unsafe extern "C" fn gaussian_regression_free(gr:
                                                          *mut GaussianRegression) {
        if gr.is_null() { return; }
        Box::from_raw(gr);
    }
    pub struct CKDE {
        kde: *mut GaussianKDE,
        precision: Buffer<f64>,
        precision_variable: f64,
        marginal_precision: Buffer<f64>,
        kde_indices: Option<Buffer<u32>>,
        regressions: *mut *mut GaussianRegression,
        nregressions: c_uint,
        lognorm_factor: f64,
    }
    #[no_mangle]
    pub unsafe extern "C" fn ckde_init(pro_que: *mut ProQue,
                                       kde: *mut GaussianKDE,
                                       precision: *mut c_double,
                                       kde_indices: *mut c_uint,
                                       regressions:
                                           *mut *mut GaussianRegression,
                                       nregressions: c_uint,
                                       lognorm_factor: f64, error: *mut Error)
     -> *mut CKDE {
        *error = Error::NotFinished;
        let d = (*kde).d;
        let pro_que = Box::from_raw(pro_que);
        let precision_slice = slice::from_raw_parts(precision, d * d);
        let (precision_buffer,) =
            {
                (match Buffer::builder().context(pro_que.context()).len(precision_slice.len()).copy_host_slice(&precision_slice).build()
                     {
                     Ok(b) => b,
                     Err(_) => {
                         *error = Error::MemoryError;
                         return ptr::null_mut();
                     }
                 },)
            };
        let (onlykde_precision,) =
            if d > 1 {
                {
                    (match Buffer::<f64>::builder().context(pro_que.context()).len((d
                                                                                        -
                                                                                        1)
                                                                                       *
                                                                                       (d
                                                                                            -
                                                                                            1)).build()
                         {
                         Ok(b) => b,
                         Err(_) => {
                             *error = Error::MemoryError;
                             return ptr::null_mut();
                         }
                     },)
                }
            } else {
                {
                    (match Buffer::<f64>::builder().context(pro_que.context()).len(1).build()
                         {
                         Ok(b) => b,
                         Err(_) => {
                             *error = Error::MemoryError;
                             return ptr::null_mut();
                         }
                     },)
                }
            };
        let kde_indices_slice = slice::from_raw_parts(kde_indices, d);
        let kde_indices =
            if d > 1 {
                let (b,) =
                    {
                        (match Buffer::builder().context(pro_que.context()).len(kde_indices_slice.len()).copy_host_slice(&kde_indices_slice).build()
                             {
                             Ok(b) => b,
                             Err(_) => {
                                 *error = Error::MemoryError;
                                 return ptr::null_mut();
                             }
                         },)
                    };
                Some(b)
            } else { None };
        let kernel_precompute_onlykde_precision =
            pro_que.kernel_builder("precompute_marginal_precision").global_work_size((d
                                                                                          -
                                                                                          1)
                                                                                         *
                                                                                         (d
                                                                                              -
                                                                                              1)).arg(&precision_buffer).arg(precision_slice[0].recip()).arg(d
                                                                                                                                                                 as
                                                                                                                                                                 u32).arg(&onlykde_precision).build().expect("Kernel sum_constant build failed.");
        kernel_precompute_onlykde_precision.enq().expect("Error while executing substract_without_origin kernel.");
        let ckde =
            Box::new(CKDE{kde,
                          precision: precision_buffer,
                          precision_variable: precision_slice[0],
                          marginal_precision: onlykde_precision,
                          kde_indices,
                          regressions,
                          nregressions,
                          lognorm_factor,});
        Box::into_raw(pro_que);
        let ptr_ckde = Box::into_raw(ckde);
        *error = Error::NoError;
        ptr_ckde
    }
    #[no_mangle]
    pub unsafe extern "C" fn ckde_free(ckde: *mut CKDE) {
        if ckde.is_null() { return; }
        Box::from_raw(ckde);
    }
    unsafe fn s2(ckde: &Box<CKDE>) -> f64 {
        let mut s2 = 0.0f64;
        for i in 0..ckde.nregressions {
            let gr = &**(ckde.regressions.offset(i as isize));
            let coeff = gr.variable_beta;
            s2 += (coeff * coeff) / gr.variance;
        }
        s2
    }
    fn s1_and_s3_constant_name(rowmajor: bool) -> &'static str {
        if rowmajor {
            "s1_and_s3_sum_constant_rowmajor"
        } else { "s1_and_s3_sum_constant_columnmajor" }
    }
    fn s1_and_s3_parents_name(rowmajor: bool) -> &'static str {
        if rowmajor {
            "s1_and_s3_sum_parents_rowmajor"
        } else { "s1_and_s3_sum_parents_columnmajor" }
    }
    unsafe fn s1_s3_coefficients(ckde: &Box<CKDE>, pro_que: &mut Box<ProQue>,
                                 test_instances_buffer: &Buffer<f64>,
                                 test_leading_dimension: u32,
                                 test_rowmajor: bool, s1: &Buffer<f64>,
                                 s3: &Buffer<f64>, m: usize) {
        buffer_fill_value(pro_que, &s1, m, 0.0f64);
        buffer_fill_value(pro_que, &s3, m, 0.0f64);
        let kernel_s1_and_s3_sum_constant =
            pro_que.kernel_builder(s1_and_s3_constant_name(test_rowmajor)).global_work_size(m).arg(test_instances_buffer).arg(test_leading_dimension).arg_named("beta",
                                                                                                                                                                None::<&Buffer<f64>>).arg_named("variable_index",
                                                                                                                                                                                                &0u32).arg_named("inv_variance",
                                                                                                                                                                                                                 &0.0f64).arg(s1).arg(s3).build().expect("Kernel s1_and_s3_sum_constant build failed.");
        let kernel_s1_and_s3_sum_parents =
            pro_que.kernel_builder(s1_and_s3_parents_name(test_rowmajor)).global_work_size(m).arg(test_instances_buffer).arg(test_leading_dimension).arg_named("beta",
                                                                                                                                                               None::<&Buffer<f64>>).arg_named("variable_index",
                                                                                                                                                                                               &0u32).arg_named("evidence_index",
                                                                                                                                                                                                                None::<&Buffer<u32>>).arg_named("len_evidence",
                                                                                                                                                                                                                                                &0u32).arg_named("inv_variance",
                                                                                                                                                                                                                                                                 &0.0f64).arg(s1).arg(s3).build().expect("Kernel s1_and_s3_sum_parents build failed.");
        for i in 0..ckde.nregressions {
            let gr = &**ckde.regressions.offset(i as isize);
            if gr.nparents > 1 {
                kernel_s1_and_s3_sum_parents.set_arg("beta",
                                                     &gr.beta).unwrap();
                kernel_s1_and_s3_sum_parents.set_arg("variable_index",
                                                     gr.variable_index).unwrap();
                kernel_s1_and_s3_sum_parents.set_arg("evidence_index",
                                                     gr.evidence_index.as_ref().unwrap()).unwrap();
                kernel_s1_and_s3_sum_parents.set_arg("len_evidence",
                                                     gr.nparents -
                                                         1).unwrap();
                kernel_s1_and_s3_sum_parents.set_arg("inv_variance",
                                                     gr.variance.recip()).unwrap();
                kernel_s1_and_s3_sum_parents.enq().expect("Error while executing kernel_s1_and_s3_sum_parents kernel.");
            } else {
                kernel_s1_and_s3_sum_constant.set_arg("beta",
                                                      &gr.beta).unwrap();
                kernel_s1_and_s3_sum_constant.set_arg("variable_index",
                                                      gr.variable_index).unwrap();
                kernel_s1_and_s3_sum_constant.set_arg("inv_variance",
                                                      gr.variance.recip()).unwrap();
                kernel_s1_and_s3_sum_constant.enq().expect("Error while executing kernel_s1_and_s3_sum_constant kernel.");
            }
        }
    }
}
pub use denominator::logdenominator_dataset_gaussian;
pub use denominator::logdenominator_dataset_onlykde;
pub use denominator::logdenominator_dataset;
mod open_cl_code {
    pub const OPEN_CL_CODE: &str =
        r#"#line 1 "./pgmpy/rust/src/kde_gaussian.cl.src"

/*
 *****************************************************************************
 **       This file was autogenerated from a template  DO NOT EDIT!!!!      **
 **       Changes should be made to the original source (.src) file         **
 *****************************************************************************
 */

#line 1
// Row major indexing for row i column j and leading dimension for the columns
#define RM(i, j, leading) ((i)*(leading) + (j))
// Column major indexing for row i column j and leading dimension for the rows
#define CM(i, j, leading) ((j)*(leading) + (i))

#define BASE_RM(i, j, leading) ((i)*(leading))
#define BASE_CM(i, j, leading) (i)

#define ADD_BASE_RM(i, j, leading) (j)
#define ADD_BASE_CM(i, j, leading) ((j)*(leading))

/**
##########################################
################  MISC  ##################
##########################################
*/


__kernel void fill_value(__global double *vec, __private double value) {
    vec[get_global_id(0)] = value;
}

__kernel void fill_value_uint(__global uint *vec, __private uint value) {
    vec[get_global_id(0)] = value;
}

__kernel void sum_vectors(__global double *left, __constant double *right) {
    uint idx = get_global_id(0);
    left[idx] += right[idx];
}

__kernel void sum_constant(__global double *v, __private double c) {
    v[get_global_id(0)] += c;
}


/**
##########################################
###############  COMMON  #################
##########################################
 */

#line 49

#line 54

__kernel void substract_rowmajor_rowmajor(__constant double *train_data,
                                            __private uint train_cols,
                                            __constant double *vec,
                                            __global double *res,
                                            __private uint row,
                                            __private uint train_leading_dimension,
                                            __private uint test_leading_dimension)
{
    int i = get_global_id(0);

    int r = i / train_leading_dimension;
    int c = i % train_leading_dimension;

    res[RM(r, c, train_cols)] = train_data[i] - vec[RM(row, c, test_leading_dimension)];
}


#line 54

__kernel void substract_rowmajor_columnmajor(__constant double *train_data,
                                            __private uint train_cols,
                                            __constant double *vec,
                                            __global double *res,
                                            __private uint row,
                                            __private uint train_leading_dimension,
                                            __private uint test_leading_dimension)
{
    int i = get_global_id(0);

    int r = i / train_leading_dimension;
    int c = i % train_leading_dimension;

    res[RM(r, c, train_cols)] = train_data[i] - vec[CM(row, c, test_leading_dimension)];
}




#line 49

#line 54

__kernel void substract_columnmajor_rowmajor(__constant double *train_data,
                                            __private uint train_cols,
                                            __constant double *vec,
                                            __global double *res,
                                            __private uint row,
                                            __private uint train_leading_dimension,
                                            __private uint test_leading_dimension)
{
    int i = get_global_id(0);

    int r = i % train_leading_dimension;
    int c = i / train_leading_dimension;

    res[RM(r, c, train_cols)] = train_data[i] - vec[RM(row, c, test_leading_dimension)];
}


#line 54

__kernel void substract_columnmajor_columnmajor(__constant double *train_data,
                                            __private uint train_cols,
                                            __constant double *vec,
                                            __global double *res,
                                            __private uint row,
                                            __private uint train_leading_dimension,
                                            __private uint test_leading_dimension)
{
    int i = get_global_id(0);

    int r = i % train_leading_dimension;
    int c = i / train_leading_dimension;

    res[RM(r, c, train_cols)] = train_data[i] - vec[CM(row, c, test_leading_dimension)];
}





__kernel void solve(__global double *diff_data, __constant double *chol, __private uint n_col) {
    uint r = get_global_id(0);
    uint index_row = r * n_col;

    for (uint c = 0; c < n_col; c++) {
        for (uint i = 0; i < c; i++) {
            diff_data[index_row + c] -= chol[c * n_col + i] * diff_data[index_row + i];
        }
        diff_data[index_row + c] /= chol[c * n_col + c];
    }
}

__kernel void square(__global double *solve_data) {
    uint idx = get_global_id(0);
    double d = solve_data[idx];
    solve_data[idx] = d * d;
}

/**
##########################################
#################  PDF  ##################
##########################################
*/

__kernel void sumout(__constant double *square_data,
                    __global double *sol_vec,
                    __private uint n_col,
                    __private double lognorm_factor) {
    uint r = get_global_id(0);
    uint idx = r * n_col;

    sol_vec[r] = square_data[idx];
    for (uint i = 1; i < n_col; i++) {
        sol_vec[r] += square_data[idx + i];
    }

    sol_vec[r] = exp(-0.5 * sol_vec[r] - lognorm_factor);
}

__kernel void sum_gpu_vec(__global double *input,
                          __local double *localSums) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    uint group_id = get_group_id(0);
    uint num_groups = get_num_groups(0);

    if (group_id == num_groups) {
        group_size = get_global_size(0) - group_id*group_size;
    }

    localSums[local_id] = input[global_id];

    while (group_size > 1) {
        int stride = group_size / 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (group_size % 2 == 0) {
            if (local_id < stride) {
                localSums[local_id] += localSums[local_id + stride];
            }

            group_size = group_size / 2;
        }
        else {
            if (local_id < stride) {
                localSums[local_id+1] += localSums[local_id+1 + stride];
            }
            group_size = (group_size / 2) + 1;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        input[group_id] = localSums[0];
    }
}

/**
##########################################
########  logPDF - Iterate test  #########
##########################################
*/

__kernel void logsumout(__constant double *square_data,
                        __global double *sol_vec,
                        __private uint n_col,
                        __private double lognorm_factor) {
    uint r = get_global_id(0);
    uint idx = r * n_col;

    sol_vec[r] = square_data[idx];
    for (uint i = 1; i < n_col; i++) {
        sol_vec[r] += square_data[idx + i];
    }

    sol_vec[r] = (-0.5 * sol_vec[r]) - lognorm_factor;
}

__kernel void max_gpu_vec_copy(__constant double *input,
                               __global double *maxGroups,
                               __local double *localMaxs) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    uint group_id = get_group_id(0);
    uint num_groups = get_num_groups(0);

    if (group_id == num_groups) {
        group_size = get_global_size(0) - group_id*group_size;
    }

    localMaxs[local_id] = input[global_id];

    while (group_size > 1) {
        int stride = group_size / 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (group_size % 2 == 0) {
            if (local_id < stride) {
                localMaxs[local_id] = max(localMaxs[local_id], localMaxs[local_id + stride]);
            }

            group_size = group_size / 2;
        }
        else {
            if (local_id < stride) {
                localMaxs[local_id+1] = max(localMaxs[local_id+1 + stride], localMaxs[local_id+1]);
            }
            group_size = (group_size / 2) + 1;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    if (local_id == 0) {
        maxGroups[group_id] = localMaxs[0];
    }
}

__kernel void max_gpu_vec(__global double* maxGroups,
                     __local double *localMaxs) {

    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    uint group_id = get_group_id(0);
    uint num_groups = get_num_groups(0);

    if (group_id == num_groups) {
        group_size = get_global_size(0) - group_id*group_size;
    }

    localMaxs[local_id] = maxGroups[global_id];

    while (group_size > 1) {
        int stride = group_size / 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (group_size % 2 == 0) {
            if (local_id < stride) {
                localMaxs[local_id] = max(localMaxs[local_id], localMaxs[local_id + stride]);
            }

            group_size = group_size / 2;
        }
        else {
            if (local_id < stride) {
                localMaxs[local_id+1] = max(localMaxs[local_id+1 + stride], localMaxs[local_id+1]);
            }
            group_size = (group_size / 2) + 1;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    if (local_id == 0) {
        maxGroups[group_id] = localMaxs[0];
    }
}

__kernel void log_sum_gpu_vec(__global double *input,
                          __local double *localSums,
                          __constant double *maxexp) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    uint group_id = get_group_id(0);
    uint num_groups = get_num_groups(0);

    if (group_id == num_groups) {
        group_size = get_global_size(0) - group_id*group_size;
    }

    localSums[local_id] = exp(input[global_id]-maxexp[0]);

    while (group_size > 1) {
        int stride = group_size / 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (group_size % 2 == 0) {
            if (local_id < stride) {
                localSums[local_id] += localSums[local_id + stride];
            }

            group_size = group_size / 2;
        }
        else {
            if (local_id < stride) {
                localSums[local_id+1] += localSums[local_id+1 + stride];
            }
            group_size = (group_size / 2) + 1;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        input[group_id] = localSums[0];
    }
}

__kernel void copy_logpdf_result(__constant double *logsum,
                                 __constant double *maxexp,
                                 __global double *res,
                                 __private uint res_offset) {
    res[res_offset] = maxexp[0] + log(logsum[0]);
}


/**
##########################################
## logPDF - Iterate train (low memory) ###
##########################################
*/

__kernel void logsumout_checkmax(__constant double *square_data,
                                __global double *sol_vec,
                                __global double *max_vec,
                                __private uint n_col,
                                __private double lognorm_factor) {
    uint r = get_global_id(0);
    uint idx = r * n_col;

    sol_vec[r] = square_data[idx];
    for (uint i = 1; i < n_col; i++) {
        sol_vec[r] += square_data[idx + i];
    }

    sol_vec[r] = (-0.5 * sol_vec[r]) - lognorm_factor;
    max_vec[r] = max(max_vec[r], sol_vec[r]);
}

__kernel void exp_and_sum(__constant double* logsum, __constant double* maxexp, __global double *res) {
    uint idx = get_global_id(0);
    res[idx] += exp(logsum[idx] - maxexp[idx]);
}

__kernel void log_and_sum(__global double* res, __constant double* maxexp) {
    uint idx = get_global_id(0);
    res[idx] = log(res[idx]) + maxexp[idx];
}



/**
##########################################
## logPDF - Iterate train (high memory) ##
##########################################
*/

__kernel void logsumout_to_matrix(__constant double *square_data,
                                    __global double *sol_mat,
                                    __private uint n_col,
                                    __private uint sol_row,
                                    __private uint n_train_instances,
                                    __private double lognorm_factor) {
    uint r = n_train_instances*get_global_id(0) + sol_row;
    uint idx = get_global_id(0) * n_col;

    sol_mat[r] = square_data[idx];
    for (uint i = 1; i < n_col; i++) {
        sol_mat[r] += square_data[idx + i];
    }

    sol_mat[r] = (-0.5 * sol_mat[r]) - lognorm_factor;
}

__kernel void max_gpu_mat_copy(__constant double *input,
                               __global double* maxGroups,
                               __local double *localMaxs,
                               __private uint array_n_cols) {

    uint global_id_row = get_global_id(0);
    uint global_id_col = get_global_id(1);
    uint n_cols = get_global_size(1);
    uint local_id = get_local_id(1);
    uint group_size = get_local_size(1);
    uint group_id = get_group_id(1);
//   FIXME: This code returns num_groups = 3 for global_size = 1000 and local_size = 256, so it does not work as expected
//      when local_work_size does not evenly divide global_work_size.
//    uint num_groups = get_num_groups(1);

    //This is equal to ceil(n_cols/group_size): https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
    uint num_groups = (n_cols + group_size - 1) / group_size;

    if (group_id+1 == num_groups) {
        group_size = get_global_size(1) - group_id*group_size;
    }

    localMaxs[local_id] = input[global_id_row*array_n_cols + global_id_col];

    while (group_size > 1) {
        int stride = group_size / 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (group_size % 2 == 0) {
            if (local_id < stride) {
                localMaxs[local_id] = max(localMaxs[local_id], localMaxs[local_id+stride]);
            }
            group_size = group_size / 2;
        }
        else {
            if (local_id < stride) {
                localMaxs[local_id+1] = max(localMaxs[local_id+1], localMaxs[local_id+1+stride]);
            }
            group_size = (group_size / 2) + 1;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    if (local_id == 0) {
        maxGroups[global_id_row*num_groups+group_id] = localMaxs[0];
    }
}

__kernel void max_gpu_mat(__global double* maxGroups,
                          __local double *localMaxs,
                          __private uint array_n_cols) {

    uint global_id_row = get_global_id(0);
    uint global_id_col = get_global_id(1);
    uint n_cols = get_global_size(1);
    uint local_id = get_local_id(1);
    uint group_size = get_local_size(1);
    uint group_id = get_group_id(1);
//   FIXME: This code returns num_groups = 3 for global_size = 1000 and local_size = 256, so it does not work as expected
//      when local_work_size does not evenly divide global_work_size.
//    uint num_groups = get_num_groups(1);

    //This is equal to ceil(n_cols/group_size): https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
    uint num_groups = (n_cols + group_size - 1) / group_size;

    if (group_id+1 == num_groups) {
        group_size = get_global_size(1) - group_id*group_size;
    }

    localMaxs[local_id] = maxGroups[global_id_row*array_n_cols + global_id_col];

    while (group_size > 1) {
        int stride = group_size / 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (group_size % 2 == 0) {
            if (local_id < stride) {
                localMaxs[local_id] = max(localMaxs[local_id], localMaxs[local_id + stride]);
            }

            group_size = group_size / 2;
        }
        else {
            if (local_id < stride) {
                localMaxs[local_id+1] = max(localMaxs[local_id+1 + stride], localMaxs[local_id+1]);
            }
            group_size = (group_size / 2) + 1;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    if (local_id == 0) {
        maxGroups[global_id_row*array_n_cols + group_id] = localMaxs[0];
    }
}


__kernel void exp_and_sum_mat(__global double* res, __constant double* maxexp, __private uint n, __private uint num_groups) {
    uint i = get_global_id(0);
    uint row = i / n;
    uint col = i % n;
    uint d = RM(row, col, n);
    res[d] = exp(res[d] - maxexp[row*num_groups]);
}

__kernel void sum_gpu_mat(__global double* maxGroups,
                          __local double *localMaxs,
                          __private uint array_n_cols) {

    uint global_id_row = get_global_id(0);
    uint global_id_col = get_global_id(1);
    uint n_cols = get_global_size(1);
    uint local_id = get_local_id(1);
    uint group_size = get_local_size(1);
    uint group_id = get_group_id(1);
//   FIXME: This code returns num_groups = 3 for global_size = 1000 and local_size = 256, so it does not work as expected
//      when local_work_size does not evenly divide global_work_size.
//    uint num_groups = get_num_groups(1);

    //This is equal to ceil(n_cols/group_size): https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
    uint num_groups = (n_cols + group_size - 1) / group_size;

    if (group_id+1 == num_groups) {
        group_size = get_global_size(1) - group_id*group_size;
    }

    localMaxs[local_id] = maxGroups[global_id_row*array_n_cols + global_id_col];

    while (group_size > 1) {
        int stride = group_size / 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (group_size % 2 == 0) {
            if (local_id < stride) {
                localMaxs[local_id] += localMaxs[local_id+stride];
            }

            group_size = group_size / 2;
        }
        else {
            if (local_id < stride) {
                localMaxs[local_id+1] += localMaxs[local_id+stride+1];
            }
            group_size = (group_size / 2) + 1;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    if (local_id == 0) {
        maxGroups[global_id_row*array_n_cols + group_id] = localMaxs[0];
    }
}

__kernel void log_and_sum_mat(__global double* res,
                                __constant double *summed_mat,
                                __constant double* maxexp,
                                __private uint n_col,
                                __private uint num_groups) {
    uint idx = get_global_id(0);
    res[idx] = log(summed_mat[idx*n_col]) + maxexp[idx*num_groups];
}

/**
##########################################
########## Denominator Only KDE ##########
##########################################
*/

#line 525

#line 530

__kernel void substract_without_origin_rowmajor_rowmajor(__constant double *train_data,
                                                                __private uint train_leading_dimension,
                                                                __constant double *test_data,
                                                                __private uint test_leading_dimension,
                                                                __global double *res,
                                                                __private uint test_row,
                                                                __private uint n_cols) {

    int gid = get_global_id(0);

    int r = gid / n_cols;
    int c = gid % n_cols;

    res[RM(r, c, n_cols)] = test_data[RM(test_row, c+1, test_leading_dimension)]
                        - train_data[RM(r, c+1, train_leading_dimension)];
}


#line 530

__kernel void substract_without_origin_rowmajor_columnmajor(__constant double *train_data,
                                                                __private uint train_leading_dimension,
                                                                __constant double *test_data,
                                                                __private uint test_leading_dimension,
                                                                __global double *res,
                                                                __private uint test_row,
                                                                __private uint n_cols) {

    int gid = get_global_id(0);

    int r = gid / n_cols;
    int c = gid % n_cols;

    res[RM(r, c, n_cols)] = test_data[CM(test_row, c+1, test_leading_dimension)]
                        - train_data[RM(r, c+1, train_leading_dimension)];
}




#line 525

#line 530

__kernel void substract_without_origin_columnmajor_rowmajor(__constant double *train_data,
                                                                __private uint train_leading_dimension,
                                                                __constant double *test_data,
                                                                __private uint test_leading_dimension,
                                                                __global double *res,
                                                                __private uint test_row,
                                                                __private uint n_cols) {

    int gid = get_global_id(0);

    int r = gid / n_cols;
    int c = gid % n_cols;

    res[RM(r, c, n_cols)] = test_data[RM(test_row, c+1, test_leading_dimension)]
                        - train_data[CM(r, c+1, train_leading_dimension)];
}


#line 530

__kernel void substract_without_origin_columnmajor_columnmajor(__constant double *train_data,
                                                                __private uint train_leading_dimension,
                                                                __constant double *test_data,
                                                                __private uint test_leading_dimension,
                                                                __global double *res,
                                                                __private uint test_row,
                                                                __private uint n_cols) {

    int gid = get_global_id(0);

    int r = gid / n_cols;
    int c = gid % n_cols;

    res[RM(r, c, n_cols)] = test_data[CM(test_row, c+1, test_leading_dimension)]
                        - train_data[CM(r, c+1, train_leading_dimension)];
}





__kernel void precompute_marginal_precision(__constant double* precision,
                                                        __private double inv_precision_variable,
                                                        __private uint d,
                                                        __global double* res) {
    uint gid = get_global_id(0);

    uint r = gid / (d-1);
    uint c = gid % (d-1);


    res[RM(r,c,d-1)] = precision[r+1]*precision[c+1]*inv_precision_variable - precision[RM(r+1,c+1,d)];
}

__kernel void onlykde_exponent_coefficients_iterate_test(__global double* Ti,
        __private uint nparents_kde,
        __global double* marginal_precision,
        __global double* train_coefficients,
        __local double* sums_buffer
)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    uint instance = get_group_id(0);

    uint p = lid / nparents_kde;
    uint q = lid % nparents_kde;

    uint base_pos = BASE_RM(instance, 0, nparents_kde);

    double Tp = Ti[base_pos + ADD_BASE_RM(instance, p, nparents_kde)];
    double Tq = Ti[base_pos + ADD_BASE_RM(instance, q, nparents_kde)];


    sums_buffer[lid] = Tq*marginal_precision[RM(q,p,nparents_kde)];


    uint remaining_sum = nparents_kde;

    if (q < remaining_sum / 2) {
        while (remaining_sum > 1) {
            uint stride = remaining_sum / 2;
            barrier(CLK_LOCAL_MEM_FENCE);

            if (q < stride) {
                sums_buffer[lid] += sums_buffer[lid+stride];
            }

            if (remaining_sum % 2 != 0 && q == 0) {
                sums_buffer[lid] += sums_buffer[lid+remaining_sum-1];
            }

            remaining_sum = stride;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (q == 0) {
        sums_buffer[p] = sums_buffer[lid] * Tp;
    }

    remaining_sum = nparents_kde;

    if (lid < remaining_sum / 2) {
        while (remaining_sum > 1) {
            uint stride = remaining_sum / 2;
            barrier(CLK_LOCAL_MEM_FENCE);

            if (lid < stride) {
                sums_buffer[lid] += sums_buffer[lid+stride];
            }

            if (remaining_sum % 2 != 0 && lid == 0) {
                sums_buffer[lid] += sums_buffer[lid+remaining_sum-1];
            }

            remaining_sum = stride;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        train_coefficients[instance] = 0.5*sums_buffer[0];
    }
}

__kernel void onlykde_exponent_coefficients_iterate_train_high_memory(__global double* Ti,
        __private uint nparents_kde,
        __global double* marginal_precision,
        __global double* train_coefficients,
        __local double* sums_buffer,
        __private uint train_index,
        __private uint n
)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    uint instance = get_group_id(0);

    uint p = lid / nparents_kde;
    uint q = lid % nparents_kde;

    uint base_pos = BASE_RM(instance, 0, nparents_kde);

    double Tp = Ti[base_pos + ADD_BASE_RM(instance, p, nparents_kde)];
    double Tq = Ti[base_pos + ADD_BASE_RM(instance, q, nparents_kde)];

    sums_buffer[lid] = Tq*marginal_precision[RM(q,p,nparents_kde)];

    uint remaining_sum = nparents_kde;

    if (q < remaining_sum / 2) {
        while (remaining_sum > 1) {
            uint stride = remaining_sum / 2;
            barrier(CLK_LOCAL_MEM_FENCE);

            if (q < stride) {
                sums_buffer[lid] += sums_buffer[lid+stride];
            }

            if (remaining_sum % 2 != 0 && q == 0) {
                sums_buffer[lid] += sums_buffer[lid+remaining_sum-1];
            }

            remaining_sum = stride;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (q == 0) {
        sums_buffer[p] = sums_buffer[lid] * Tp;
    }

    remaining_sum = nparents_kde;

    if (lid < remaining_sum / 2) {
        while (remaining_sum > 1) {
            uint stride = remaining_sum / 2;
            barrier(CLK_LOCAL_MEM_FENCE);

            if (lid < stride) {
                sums_buffer[lid] += sums_buffer[lid+stride];
            }

            if (remaining_sum % 2 != 0 && lid == 0) {
                sums_buffer[lid] += sums_buffer[lid+remaining_sum-1];
            }

            remaining_sum = stride;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        train_coefficients[RM(instance, train_index, n)] = 0.5*sums_buffer[0];
    }
}

__kernel void onlykde_exponent_coefficients_iterate_train_low_memory_checkmax(__global double* Ti,
        __private uint nparents_kde,
        __global double* marginal_precision,
        __global double* max_buffer,
        __local double* sums_buffer,
        __private uint n
)
{

    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    uint instance = get_group_id(0);

    uint p = lid / nparents_kde;
    uint q = lid % nparents_kde;

    uint base_pos = BASE_RM(instance, 0, nparents_kde);

    double Tp = Ti[base_pos + ADD_BASE_RM(instance, p, nparents_kde)];
    double Tq = Ti[base_pos + ADD_BASE_RM(instance, q, nparents_kde)];

    sums_buffer[lid] = Tq*marginal_precision[RM(q,p,nparents_kde)];

    uint remaining_sum = nparents_kde;

    if (q < remaining_sum / 2) {
        while (remaining_sum > 1) {
            uint stride = remaining_sum / 2;
            barrier(CLK_LOCAL_MEM_FENCE);

            if (q < stride) {
                sums_buffer[lid] += sums_buffer[lid+stride];
            }

            if (remaining_sum % 2 != 0 && q == 0) {
                sums_buffer[lid] += sums_buffer[lid+remaining_sum-1];
            }

            remaining_sum = stride;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (q == 0) {
        sums_buffer[p] = sums_buffer[lid] * Tp;
    }

    remaining_sum = nparents_kde;

    if (lid < remaining_sum / 2) {
        while (remaining_sum > 1) {
            uint stride = remaining_sum / 2;
            barrier(CLK_LOCAL_MEM_FENCE);

            if (lid < stride) {
                sums_buffer[lid] += sums_buffer[lid+stride];
            }

            if (remaining_sum % 2 != 0 && lid == 0) {
                sums_buffer[lid] += sums_buffer[lid+remaining_sum-1];
            }

            remaining_sum = stride;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        max_buffer[instance] = max(max_buffer[instance], 0.5*sums_buffer[0]);
    }
}

__kernel void onlykde_exponent_coefficients_iterate_train_low_memory_compute(__global double* Ti,
        __private uint nparents_kde,
        __global double* marginal_precision,
        __global double* final_result,
        __global double* max_buffer,
        __local double* sums_buffer,
        __private uint n
)
{

    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    uint instance = get_group_id(0);

    uint p = lid / nparents_kde;
    uint q = lid % nparents_kde;

    uint base_pos = BASE_RM(instance, 0, nparents_kde);

    double Tp = Ti[base_pos + ADD_BASE_RM(instance, p, nparents_kde)];
    double Tq = Ti[base_pos + ADD_BASE_RM(instance, q, nparents_kde)];

    sums_buffer[lid] = Tq*marginal_precision[RM(q,p,nparents_kde)];

    uint remaining_sum = nparents_kde;

    while (remaining_sum > 1) {
        uint stride = remaining_sum / 2;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (q < stride) {
            sums_buffer[lid] += sums_buffer[lid+stride];
        }

        if (remaining_sum % 2 != 0 && q == 0) {
            sums_buffer[lid] += sums_buffer[lid+remaining_sum-1];
        }

        remaining_sum = stride;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (q == 0) {
        sums_buffer[p] = sums_buffer[lid] * Tp;
    }

    remaining_sum = nparents_kde;

    while (remaining_sum > 1) {
        uint stride = remaining_sum / 2;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid<stride) {
            sums_buffer[lid] += sums_buffer[lid+stride];
        }

        if (remaining_sum % 2 != 0 && lid == 0) {
            sums_buffer[lid] += sums_buffer[lid+remaining_sum-1];
        }

        remaining_sum = stride;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        final_result[instance] += exp(0.5*sums_buffer[0] - max_buffer[instance]);
    }
}

/**
##########################################
####### Denominator Only Gaussian ########
##########################################
*/

#line 864

__kernel void s1_and_s3_sum_parents_rowmajor(__constant double* test_dataset,
                                        __private uint leading_dimension,
                                        __constant double* beta,
                                        __private uint variable_index,
                                        __constant uint* evidence_index,
                                        __private uint len_evidence,
                                        __private double inv_variance,
                                        __global double* s1,
                                        __global double* s3
)
{
    uint row = get_global_id(0);
    double Cj = beta[0];

    uint base_pos = BASE_RM(row, 0, leading_dimension);
    for(int i = 0; i < len_evidence; i++) {
        Cj += beta[i+2]*test_dataset[base_pos + ADD_BASE_RM(row, evidence_index[i], leading_dimension)];
    }

    double diff = (Cj - test_dataset[base_pos + ADD_BASE_RM(row, variable_index, leading_dimension)]);
    s1[row] += beta[1]*diff*inv_variance;
    s3[row] += diff*diff*inv_variance;
}

__kernel void s1_and_s3_sum_constant_rowmajor(__constant double* test_dataset,
                                        __private uint leading_dimension,
                                        __constant double* beta,
                                        __private uint variable_index,
                                        __private double inv_variance,
                                        __global double* s1,
                                        __global double* s3
)
{
    uint row = get_global_id(0);
    double Cj = beta[0];

    double diff = (Cj - test_dataset[RM(row, variable_index, leading_dimension)]);
    s1[row] += beta[1]*diff*inv_variance;
    s3[row] += diff*diff*inv_variance;
}


#line 864

__kernel void s1_and_s3_sum_parents_columnmajor(__constant double* test_dataset,
                                        __private uint leading_dimension,
                                        __constant double* beta,
                                        __private uint variable_index,
                                        __constant uint* evidence_index,
                                        __private uint len_evidence,
                                        __private double inv_variance,
                                        __global double* s1,
                                        __global double* s3
)
{
    uint row = get_global_id(0);
    double Cj = beta[0];

    uint base_pos = BASE_CM(row, 0, leading_dimension);
    for(int i = 0; i < len_evidence; i++) {
        Cj += beta[i+2]*test_dataset[base_pos + ADD_BASE_CM(row, evidence_index[i], leading_dimension)];
    }

    double diff = (Cj - test_dataset[base_pos + ADD_BASE_CM(row, variable_index, leading_dimension)]);
    s1[row] += beta[1]*diff*inv_variance;
    s3[row] += diff*diff*inv_variance;
}

__kernel void s1_and_s3_sum_constant_columnmajor(__constant double* test_dataset,
                                        __private uint leading_dimension,
                                        __constant double* beta,
                                        __private uint variable_index,
                                        __private double inv_variance,
                                        __global double* s1,
                                        __global double* s3
)
{
    uint row = get_global_id(0);
    double Cj = beta[0];

    double diff = (Cj - test_dataset[CM(row, variable_index, leading_dimension)]);
    s1[row] += beta[1]*diff*inv_variance;
    s3[row] += diff*diff*inv_variance;
}



#line 912
__kernel void onlygaussian_exponent_coefficients_iterate_test_rowmajor(__global double* training_dataset,
                                             __private uint train_leading_dimension,
                                             __global double* precision,
                                             __global double* s1,
                                             __private double inv_a,
                                             __global double* s3,
                                             __private uint test_index,
                                             __global double* train_coefficients
                                             )
{
    int i = get_global_id(0);

    double precisionK = precision[0];
    double instanceK = training_dataset[BASE_RM(i, 0, train_leading_dimension)];

    double diff_numerator = instanceK*precisionK - s1[test_index];
    train_coefficients[i] = diff_numerator*diff_numerator*inv_a - 0.5*instanceK*instanceK*precisionK - 0.5*s3[test_index];
}

__kernel void onlygaussian_exponent_coefficients_iterate_train_high_memory_rowmajor(__global double* training_dataset,
                                                                        __private uint train_leading_dimension,
                                                                        __global double* precision,
                                                                        __global double* s1,
                                                                        __private double inv_a,
                                                                        __global double* s3,
                                                                        __global double* train_coefficients,
                                                                        __private uint n
)
{
    int i = get_global_id(0);

    int test_index = i / n;
    int train_index = i % n;

    double precisionK = precision[0];
    double instanceK = training_dataset[BASE_RM(train_index, 0, train_leading_dimension)];
    double diff_numerator = instanceK*precisionK - s1[test_index];

    train_coefficients[RM(test_index, train_index, n)] =
            diff_numerator*diff_numerator*inv_a - 0.5*instanceK*instanceK*precisionK - 0.5*s3[test_index];
}


__kernel void onlygaussian_exponent_coefficients_iterate_train_low_memory_checkmax_rowmajor(__global double* training_dataset,
                                                                __private uint train_leading_dimension,
                                                                __global double* precision,
                                                                __global double* s1,
                                                                __private double inv_a,
                                                                __global double* s3,
                                                                __private uint train_index,
                                                                __global double* max_array
)
{
    int i = get_global_id(0);

    double precisionK = precision[0];
    double instanceK = training_dataset[BASE_RM(train_index, 0, train_leading_dimension)];
    double diff_numerator = instanceK*precisionK - s1[i];

    double coeff = diff_numerator*diff_numerator*inv_a - 0.5*instanceK*instanceK*precisionK - 0.5*s3[i];

    max_array[i] = max(max_array[i], coeff);
}

__kernel void onlygaussian_exponent_coefficients_iterate_train_low_memory_compute_rowmajor(__global double* training_dataset,
                                                                __private uint train_leading_dimension,
                                                                __global double* precision,
                                                                __global double* s1,
                                                                __private double inv_a,
                                                                __global double* s3,
                                                                __private uint train_index,
                                                                __global double* max_array,
                                                                __global double* final_result
)
{
    int i = get_global_id(0);

    double precisionK = precision[0];
    double instanceK = training_dataset[BASE_RM(train_index, 0, train_leading_dimension)];
    double diff_numerator = instanceK*precisionK - s1[i];

    final_result[i] += exp(diff_numerator*diff_numerator*inv_a - 0.5*instanceK*instanceK*precisionK - 0.5*s3[i]
                    - max_array[i]);
}


#line 912
__kernel void onlygaussian_exponent_coefficients_iterate_test_columnmajor(__global double* training_dataset,
                                             __private uint train_leading_dimension,
                                             __global double* precision,
                                             __global double* s1,
                                             __private double inv_a,
                                             __global double* s3,
                                             __private uint test_index,
                                             __global double* train_coefficients
                                             )
{
    int i = get_global_id(0);

    double precisionK = precision[0];
    double instanceK = training_dataset[BASE_CM(i, 0, train_leading_dimension)];

    double diff_numerator = instanceK*precisionK - s1[test_index];
    train_coefficients[i] = diff_numerator*diff_numerator*inv_a - 0.5*instanceK*instanceK*precisionK - 0.5*s3[test_index];
}

__kernel void onlygaussian_exponent_coefficients_iterate_train_high_memory_columnmajor(__global double* training_dataset,
                                                                        __private uint train_leading_dimension,
                                                                        __global double* precision,
                                                                        __global double* s1,
                                                                        __private double inv_a,
                                                                        __global double* s3,
                                                                        __global double* train_coefficients,
                                                                        __private uint n
)
{
    int i = get_global_id(0);

    int test_index = i / n;
    int train_index = i % n;

    double precisionK = precision[0];
    double instanceK = training_dataset[BASE_CM(train_index, 0, train_leading_dimension)];
    double diff_numerator = instanceK*precisionK - s1[test_index];

    train_coefficients[RM(test_index, train_index, n)] =
            diff_numerator*diff_numerator*inv_a - 0.5*instanceK*instanceK*precisionK - 0.5*s3[test_index];
}


__kernel void onlygaussian_exponent_coefficients_iterate_train_low_memory_checkmax_columnmajor(__global double* training_dataset,
                                                                __private uint train_leading_dimension,
                                                                __global double* precision,
                                                                __global double* s1,
                                                                __private double inv_a,
                                                                __global double* s3,
                                                                __private uint train_index,
                                                                __global double* max_array
)
{
    int i = get_global_id(0);

    double precisionK = precision[0];
    double instanceK = training_dataset[BASE_CM(train_index, 0, train_leading_dimension)];
    double diff_numerator = instanceK*precisionK - s1[i];

    double coeff = diff_numerator*diff_numerator*inv_a - 0.5*instanceK*instanceK*precisionK - 0.5*s3[i];

    max_array[i] = max(max_array[i], coeff);
}

__kernel void onlygaussian_exponent_coefficients_iterate_train_low_memory_compute_columnmajor(__global double* training_dataset,
                                                                __private uint train_leading_dimension,
                                                                __global double* precision,
                                                                __global double* s1,
                                                                __private double inv_a,
                                                                __global double* s3,
                                                                __private uint train_index,
                                                                __global double* max_array,
                                                                __global double* final_result
)
{
    int i = get_global_id(0);

    double precisionK = precision[0];
    double instanceK = training_dataset[BASE_CM(train_index, 0, train_leading_dimension)];
    double diff_numerator = instanceK*precisionK - s1[i];

    final_result[i] += exp(diff_numerator*diff_numerator*inv_a - 0.5*instanceK*instanceK*precisionK - 0.5*s3[i]
                    - max_array[i]);
}




/**
##########################################
############ Denominator Mix #############
##########################################
*/

#line 1010

#line 1015

__kernel void substract_without_origin_from_indices_iterate_test_rowmajor_rowmajor(__constant double *train_data,
                                                            __private uint train_leading_dimension,
                                                            __constant double *test_data,
                                                            __private uint test_leading_dimension,
                                                            __global double *res,
                                                            __private uint test_row,
                                                            __private uint nparents_kde,
                                                            __constant uint* kde_indices) {

    int gid = get_global_id(0);

    int r = gid / nparents_kde;
    int c = gid % nparents_kde;

    res[RM(r, c, nparents_kde)] = test_data[RM(test_row, kde_indices[c], test_leading_dimension)]
                            - train_data[RM(r, c+1, train_leading_dimension)];
}


#line 1015

__kernel void substract_without_origin_from_indices_iterate_test_rowmajor_columnmajor(__constant double *train_data,
                                                            __private uint train_leading_dimension,
                                                            __constant double *test_data,
                                                            __private uint test_leading_dimension,
                                                            __global double *res,
                                                            __private uint test_row,
                                                            __private uint nparents_kde,
                                                            __constant uint* kde_indices) {

    int gid = get_global_id(0);

    int r = gid / nparents_kde;
    int c = gid % nparents_kde;

    res[RM(r, c, nparents_kde)] = test_data[CM(test_row, kde_indices[c], test_leading_dimension)]
                            - train_data[RM(r, c+1, train_leading_dimension)];
}




#line 1010

#line 1015

__kernel void substract_without_origin_from_indices_iterate_test_columnmajor_rowmajor(__constant double *train_data,
                                                            __private uint train_leading_dimension,
                                                            __constant double *test_data,
                                                            __private uint test_leading_dimension,
                                                            __global double *res,
                                                            __private uint test_row,
                                                            __private uint nparents_kde,
                                                            __constant uint* kde_indices) {

    int gid = get_global_id(0);

    int r = gid / nparents_kde;
    int c = gid % nparents_kde;

    res[RM(r, c, nparents_kde)] = test_data[RM(test_row, kde_indices[c], test_leading_dimension)]
                            - train_data[CM(r, c+1, train_leading_dimension)];
}


#line 1015

__kernel void substract_without_origin_from_indices_iterate_test_columnmajor_columnmajor(__constant double *train_data,
                                                            __private uint train_leading_dimension,
                                                            __constant double *test_data,
                                                            __private uint test_leading_dimension,
                                                            __global double *res,
                                                            __private uint test_row,
                                                            __private uint nparents_kde,
                                                            __constant uint* kde_indices) {

    int gid = get_global_id(0);

    int r = gid / nparents_kde;
    int c = gid % nparents_kde;

    res[RM(r, c, nparents_kde)] = test_data[CM(test_row, kde_indices[c], test_leading_dimension)]
                            - train_data[CM(r, c+1, train_leading_dimension)];
}





#line 1042

#line 1047

__kernel void substract_without_origin_from_indices_iterate_train_rowmajor_rowmajor(__constant double *train_data,
        __private uint train_leading_dimension,
        __constant double *test_data,
        __private uint test_leading_dimension,
        __global double *res,
        __private uint train_row,
        __private uint nparents_kde,
        __constant uint* kde_indices) {

    int gid = get_global_id(0);

    int r = gid / nparents_kde;
    int c = gid % nparents_kde;

    res[RM(r, c, nparents_kde)] = test_data[RM(r, kde_indices[c], test_leading_dimension)]
                                - train_data[RM(train_row, c+1, train_leading_dimension)];
}


#line 1047

__kernel void substract_without_origin_from_indices_iterate_train_rowmajor_columnmajor(__constant double *train_data,
        __private uint train_leading_dimension,
        __constant double *test_data,
        __private uint test_leading_dimension,
        __global double *res,
        __private uint train_row,
        __private uint nparents_kde,
        __constant uint* kde_indices) {

    int gid = get_global_id(0);

    int r = gid / nparents_kde;
    int c = gid % nparents_kde;

    res[RM(r, c, nparents_kde)] = test_data[CM(r, kde_indices[c], test_leading_dimension)]
                                - train_data[RM(train_row, c+1, train_leading_dimension)];
}




#line 1042

#line 1047

__kernel void substract_without_origin_from_indices_iterate_train_columnmajor_rowmajor(__constant double *train_data,
        __private uint train_leading_dimension,
        __constant double *test_data,
        __private uint test_leading_dimension,
        __global double *res,
        __private uint train_row,
        __private uint nparents_kde,
        __constant uint* kde_indices) {

    int gid = get_global_id(0);

    int r = gid / nparents_kde;
    int c = gid % nparents_kde;

    res[RM(r, c, nparents_kde)] = test_data[RM(r, kde_indices[c], test_leading_dimension)]
                                - train_data[CM(train_row, c+1, train_leading_dimension)];
}


#line 1047

__kernel void substract_without_origin_from_indices_iterate_train_columnmajor_columnmajor(__constant double *train_data,
        __private uint train_leading_dimension,
        __constant double *test_data,
        __private uint test_leading_dimension,
        __global double *res,
        __private uint train_row,
        __private uint nparents_kde,
        __constant uint* kde_indices) {

    int gid = get_global_id(0);

    int r = gid / nparents_kde;
    int c = gid % nparents_kde;

    res[RM(r, c, nparents_kde)] = test_data[CM(r, kde_indices[c], test_leading_dimension)]
                                - train_data[CM(train_row, c+1, train_leading_dimension)];
}






__kernel void mahalanobis(__constant double *Ti,
                        __constant double* precision,
                        __global double* res,
                        __local double* sums_buffer,
                        __private uint nparents_kde
)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    uint instance = get_group_id(0);

    uint p = lid / nparents_kde;
    uint q = lid % nparents_kde;

    sums_buffer[lid] = Ti[RM(instance, p, nparents_kde)]*Ti[RM(instance, q, nparents_kde)]*
                        precision[RM(p+1,q+1,nparents_kde+1)];

    uint remaining_sum = nparents_kde*nparents_kde;

    while (remaining_sum > 1) {
        uint stride = remaining_sum / 2;
        barrier(CLK_LOCAL_MEM_FENCE);

            if (lid < stride) {
                sums_buffer[lid] += sums_buffer[lid+stride];
            }

            if (remaining_sum % 2 != 0 && lid == 0) {
                sums_buffer[lid] += sums_buffer[lid+remaining_sum-1];
            }

            remaining_sum = stride;
    }

    if (lid == 0) {
        res[instance] = sums_buffer[0];
    }
}

__kernel void mahalanobis_mat(__constant double *Ti,
        __constant double* precision,
        __global double* res,
        __local double* sums_buffer,
        __private uint nparents_kde,
        __private uint train_index,
        __private uint n
)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    uint instance = get_group_id(0);

    uint p = lid / nparents_kde;
    uint q = lid % nparents_kde;

    sums_buffer[lid] = Ti[RM(instance, p, nparents_kde)]*Ti[RM(instance, q, nparents_kde)]*
    precision[RM(p+1,q+1,nparents_kde+1)];

    uint remaining_sum = nparents_kde*nparents_kde;

    while (remaining_sum > 1) {
        uint stride = remaining_sum / 2;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid < stride) {
            sums_buffer[lid] += sums_buffer[lid+stride];
        }

        if (remaining_sum % 2 != 0 && lid == 0) {
            sums_buffer[lid] += sums_buffer[lid+remaining_sum-1];
        }

        remaining_sum = stride;
    }

    if (lid == 0) {
        res[RM(instance, train_index, n)] = sums_buffer[0];
    }
}


__kernel void dotproduct(__constant double *Ti,
            __constant double* precision,
            __global double* res,
            __local double* sums_buffer,
            __private uint nparents_kde
)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    uint instance = get_group_id(0);


    sums_buffer[lid] = Ti[RM(instance, lid, nparents_kde)]*precision[lid+1];


    uint remaining_sum = nparents_kde;

    while (remaining_sum > 1) {
        uint stride = remaining_sum / 2;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid < stride) {
            sums_buffer[lid] += sums_buffer[lid+stride];
        }

        if (remaining_sum % 2 != 0 && lid == 0) {
            sums_buffer[lid] += sums_buffer[lid+remaining_sum-1];
        }

        remaining_sum = stride;
    }

    if (lid == 0) {
        res[instance] = sums_buffer[0];
    }
}

#line 1196

__kernel void exponent_coefficients_iterate_test_rowmajor(__constant double *train_data,
                                                __private uint train_leading_dimension,
                                                __constant double* precision,
                                                __global double* mahalanobis,
                                                __constant double* dotproduct,
                                                __constant double* s1,
                                                __private double inv_a,
                                                __constant double* s3,
                                                __private uint test_index

)
{
    uint gid = get_global_id(0);

    double train_variable = train_data[BASE_RM(gid, 0, train_leading_dimension)];
    double dot_instance = dotproduct[gid];

    double bi = train_variable*precision[0] - dot_instance - s1[test_index];

    double ci = 0.5*(mahalanobis[gid] - 2*train_variable*dot_instance
            + train_variable*train_variable*precision[0] + s3[test_index]);

    mahalanobis[gid] = bi*bi*inv_a - ci;
}


#line 1196

__kernel void exponent_coefficients_iterate_test_columnmajor(__constant double *train_data,
                                                __private uint train_leading_dimension,
                                                __constant double* precision,
                                                __global double* mahalanobis,
                                                __constant double* dotproduct,
                                                __constant double* s1,
                                                __private double inv_a,
                                                __constant double* s3,
                                                __private uint test_index

)
{
    uint gid = get_global_id(0);

    double train_variable = train_data[BASE_CM(gid, 0, train_leading_dimension)];
    double dot_instance = dotproduct[gid];

    double bi = train_variable*precision[0] - dot_instance - s1[test_index];

    double ci = 0.5*(mahalanobis[gid] - 2*train_variable*dot_instance
            + train_variable*train_variable*precision[0] + s3[test_index]);

    mahalanobis[gid] = bi*bi*inv_a - ci;
}



#line 1228

__kernel void exponent_coefficients_iterate_train_high_memory_rowmajor(__constant double *train_data,
                                        __private uint train_leading_dimension,
                                        __constant double* precision,
                                        __global double* coeffs,
                                        __constant double* dotproduct,
                                        __constant double* s1,
                                        __private double inv_a,
                                        __constant double* s3,
                                        __private uint train_index,
                                        __private uint n
)
{
    uint gid = get_global_id(0);

    double train_variable = train_data[BASE_RM(train_index, 0, train_leading_dimension)];
//    Negate the dotproduct because we computed Train - Test in substract instead of Test - Train.
    double dot_instance = dotproduct[gid];

    double bi = train_variable*precision[0] - dot_instance - s1[gid];


    double ci = 0.5*(coeffs[RM(gid, train_index, n)] - 2*train_variable*dot_instance
                 + train_variable*train_variable*precision[0] + s3[gid]);

//    printf("gid %d, train_index %d, bi %f, ci %f, dotproduct %f",
//            gid, train_index, bi, ci, dot_instance);

    coeffs[RM(gid, train_index, n)] = bi*bi*inv_a - ci;
}


#line 1228

__kernel void exponent_coefficients_iterate_train_high_memory_columnmajor(__constant double *train_data,
                                        __private uint train_leading_dimension,
                                        __constant double* precision,
                                        __global double* coeffs,
                                        __constant double* dotproduct,
                                        __constant double* s1,
                                        __private double inv_a,
                                        __constant double* s3,
                                        __private uint train_index,
                                        __private uint n
)
{
    uint gid = get_global_id(0);

    double train_variable = train_data[BASE_CM(train_index, 0, train_leading_dimension)];
//    Negate the dotproduct because we computed Train - Test in substract instead of Test - Train.
    double dot_instance = dotproduct[gid];

    double bi = train_variable*precision[0] - dot_instance - s1[gid];


    double ci = 0.5*(coeffs[RM(gid, train_index, n)] - 2*train_variable*dot_instance
                 + train_variable*train_variable*precision[0] + s3[gid]);

//    printf("gid %d, train_index %d, bi %f, ci %f, dotproduct %f",
//            gid, train_index, bi, ci, dot_instance);

    coeffs[RM(gid, train_index, n)] = bi*bi*inv_a - ci;
}



#line 1265

__kernel void exponent_coefficients_iterate_train_low_memory_checkmax_rowmajor(__constant double *train_data,
                                        __private uint train_leading_dimension,
                                        __constant double* precision,
                                        __constant double* mahalanobis,
                                        __global double* max_coeffs,
                                        __constant double* dotproduct,
                                        __constant double* s1,
                                        __private double inv_a,
                                        __constant double* s3,
                                        __private uint train_index
)
{
    uint gid = get_global_id(0);

    double train_variable = train_data[BASE_RM(train_index, 0, train_leading_dimension)];
    double dot_instance = dotproduct[gid];

    double bi = train_variable*precision[0] - dot_instance - s1[gid];

    double ci = 0.5*(mahalanobis[gid] - 2*train_variable*dot_instance
                     + train_variable*train_variable*precision[0] + s3[gid]);

    max_coeffs[gid] = max(max_coeffs[gid], bi*bi*inv_a - ci);
}


#line 1265

__kernel void exponent_coefficients_iterate_train_low_memory_checkmax_columnmajor(__constant double *train_data,
                                        __private uint train_leading_dimension,
                                        __constant double* precision,
                                        __constant double* mahalanobis,
                                        __global double* max_coeffs,
                                        __constant double* dotproduct,
                                        __constant double* s1,
                                        __private double inv_a,
                                        __constant double* s3,
                                        __private uint train_index
)
{
    uint gid = get_global_id(0);

    double train_variable = train_data[BASE_CM(train_index, 0, train_leading_dimension)];
    double dot_instance = dotproduct[gid];

    double bi = train_variable*precision[0] - dot_instance - s1[gid];

    double ci = 0.5*(mahalanobis[gid] - 2*train_variable*dot_instance
                     + train_variable*train_variable*precision[0] + s3[gid]);

    max_coeffs[gid] = max(max_coeffs[gid], bi*bi*inv_a - ci);
}



#line 1297

__kernel void exponent_coefficients_iterate_train_low_memory_compute_rowmajor(__constant double *train_data,
                                        __private uint train_leading_dimension,
                                        __constant double* precision,
                                        __constant double* mahalanobis,
                                        __global double* coeffs,
                                        __constant double* max_coeffs,
                                        __constant double* dotproduct,
                                        __constant double* s1,
                                        __private double inv_a,
                                        __constant double* s3,
                                        __private uint train_index
)
{
    uint gid = get_global_id(0);

    double train_variable = train_data[BASE_RM(train_index, 0, train_leading_dimension)];
    double dot_instance = dotproduct[gid];

    double bi = train_variable*precision[0] - dot_instance - s1[gid];

    double ci = 0.5*(mahalanobis[gid] - 2*train_variable*dot_instance
                     + train_variable*train_variable*precision[0] + s3[gid]);

    coeffs[gid] += exp(bi*bi*inv_a - ci - max_coeffs[gid]);
}


#line 1297

__kernel void exponent_coefficients_iterate_train_low_memory_compute_columnmajor(__constant double *train_data,
                                        __private uint train_leading_dimension,
                                        __constant double* precision,
                                        __constant double* mahalanobis,
                                        __global double* coeffs,
                                        __constant double* max_coeffs,
                                        __constant double* dotproduct,
                                        __constant double* s1,
                                        __private double inv_a,
                                        __constant double* s3,
                                        __private uint train_index
)
{
    uint gid = get_global_id(0);

    double train_variable = train_data[BASE_CM(train_index, 0, train_leading_dimension)];
    double dot_instance = dotproduct[gid];

    double bi = train_variable*precision[0] - dot_instance - s1[gid];

    double ci = 0.5*(mahalanobis[gid] - 2*train_variable*dot_instance
                     + train_variable*train_variable*precision[0] + s3[gid]);

    coeffs[gid] += exp(bi*bi*inv_a - ci - max_coeffs[gid]);
}



// Don't let /**end repeat**/ to be the last line: conv_template.py won't work.
"#;
}
#[macro_use]
mod util_macros {
    /// This macro copies some slice-like data into OpenCL buffers. The buffers are returned in a tuple
    /// containing all the buffers. If some buffer returns with error while creating it, it returns
    /// setting a memory error. There are two variants of the macro, depending on the return value if
    /// the allocation fails:
    /// * The simpler variant just makes a `return;`
    /// * The more complex variant adds a `=> $ret` suffix to return with a `return $ret;`
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate kde_rust;
    /// // Simple variant
    /// fn main() {
    ///     let pro_que = ProQue::builder().src(r#"__kernel void dummy {}"#).build().unwrap();
    ///     let mut error = Error::NoError;
    ///     let error_ptr = &mut error as *mut Error;
    ///     let vec = vec![0.0f64; 10];
    ///     let vec2 = vec![1.0f64; 20];
    ///     // It does a return; if the allocation fails:
    ///     let (buffer,) = copy_buffers!(pro_que, error_ptr, vec);
    ///     // Multiple buffering copies:
    ///     let (buffer1, buffer2) = copy_buffers!(pro_que, error_ptr, vec, vec2);
    /// }
    /// ```
    /// ```
    /// # #[macro_use] extern crate kde_rust;
    ///  // Complex variant.
    ///  fn dummy() -> bool {
    ///     let pro_que = ProQue::builder().src(r#"__kernel void dummy {}"#).build().unwrap();
    ///     let mut error = Error::NoError;
    ///     let error_ptr = &mut error as *mut Error;
    ///     let vec = vec![0.0f64; 10];
    ///     let vec2 = vec![1.0f64; 20];
    ///     // It does a  return true; if the allocation fails:
    ///     let (buffer,) = copy_buffers!(pro_que, error_ptr, vec => true);
    ///     // Multiple buffering copies:
    ///     let (buffer1, buffer2) = copy_buffers!(pro_que, error_ptr, vec, vec2 => true);
    ///  }
    /// ```
    ///
    #[macro_export]
    macro_rules! copy_buffers {
        ($ pro_que : expr, $ error : expr, $ ($ slice : expr), +) =>
        {
            {
                ($
                 (match Buffer :: builder () . context
                  ($ pro_que . context ()) . len ($ slice . len ()) .
                  copy_host_slice (& $ slice) . build ()
                  {
                      Ok (b) => b, Err (_) =>
                      { * $ error = Error :: MemoryError ; return ; }
                  },) +)
            }
        } ;
        ($ pro_que : expr, $ error : expr, $ ($ slice : expr), + => $ ret :
         expr) =>
        {
            {
                ($
                 (match Buffer :: builder () . context
                  ($ pro_que . context ()) . len ($ slice . len ()) .
                  copy_host_slice (& $ slice) . build ()
                  {
                      Ok (b) => b, Err (_) =>
                      { * $ error = Error :: MemoryError ; return $ ret ; }
                  },) +)
            }
        } ;
    }
    /// This macro creates some new OpenCL buffer, given its type and their lengths. The buffers are
    /// returned in a tuple containing all the buffers. If some buffer returns with error while
    /// creating it, it returns setting a memory error.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate kde_rust;
    /// fn main() {
    ///     let pro_que = ProQue::builder().src(r#"__kernel void dummy {}"#).build().unwrap();
    ///     let mut error = Error::NoError;
    ///     let error_ptr = &mut error as *mut Error;
    ///     // It does a return; if the allocation fails.
    ///     let (buffer,) = empty_buffers!(pro_que, error_ptr, f64, 10);
    ///     // Multiple buffer allocations
    ///     let (buffer, buffer2, buffer3) = empty_buffers!(pro_que, error_ptr, f64, 10, 35, 40);
    /// }
    /// ```
    #[macro_export]
    macro_rules! empty_buffers {
        ($ pro_que : expr, $ error : expr, $ type : ty, $ ($ len : expr), +)
        =>
        {
            {
                ($
                 (match Buffer :: < $ type > :: builder () . context
                  ($ pro_que . context ()) . len ($ len) . build ()
                  {
                      Ok (b) => b, Err (_) =>
                      { * $ error = Error :: MemoryError ; return ; }
                  },) +)
            }
        } ;
        ($ pro_que : expr, $ error : expr, $ type : ty, $ ($ len : expr), + =>
         $ ret : expr) =>
        {
            {
                ($
                 (match Buffer :: < $ type > :: builder () . context
                  ($ pro_que . context ()) . len ($ len) . build ()
                  {
                      Ok (b) => b, Err (_) =>
                      { * $ error = Error :: MemoryError ; return $ ret ; }
                  },) +)
            }
        } ;
    }
    #[macro_export]
    macro_rules! print_buffers {
        ($ pro_que : expr, $ ($ buffer : expr), +) =>
        {
            $
            (let len_buffer = $ buffer . len () ; let mut vec = vec !
             [Default :: default () ; len_buffer] ; $ buffer . cmd () . queue
             ($ pro_que . queue ()) . read (& mut vec) . enq () . expect
             ("Error reading result data.") ; println !
             ("{} = {:?}", stringify ! ($ buffer), vec) ;) +
        } ;
    }
    #[macro_export]
    macro_rules! print_buffers_simple {
        ($ pro_que : expr, $ ($ buffer : expr), +) =>
        {
            $
            (let len_buffer = $ buffer . len () ; let mut vec = vec !
             [Default :: default () ; len_buffer] ; $ buffer . cmd () . queue
             ($ pro_que . queue ()) . read (& mut vec) . enq () . expect
             ("Error reading result data.") ; if vec . len () > 20
             {
                 println !
                 ("{} = [{:?}, {:?}]", stringify ! ($ buffer), & vec [.. 10],
                  & vec [vec . len () - 10 ..]) ;
             } else
             { println ! ("{} = {:?}", stringify ! ($ buffer), & vec) ; }) +
        } ;
    }
    #[macro_export]
    macro_rules! to_cpu {
        ($ pro_que : expr, $ ($ buffer : expr), +) =>
        { { ($ (to_cpu_single ! ($ pro_que, $ buffer),) +) } } ;
    }
    #[macro_export]
    macro_rules! to_cpu_single {
        ($ pro_que : expr, $ buffer : expr) =>
        {
            {
                let len_buffer = $ buffer . len () ; let mut vec = vec !
                [Default :: default () ; len_buffer] ; $ buffer . cmd () .
                queue ($ pro_que . queue ()) . read (& mut vec) . enq () .
                expect ("Error reading result data.") ; vec
            }
        } ;
    }
}
/// This struct represents a double Numpy array
#[repr(C)]
pub struct DoubleNumpyArray {
    ptr: *mut c_double,
    size: size_t,
    ndim: size_t,
    shape: *mut size_t,
    strides: *mut size_t,
}
/// Loads the meta-data of a Numpy array: its data pointer, and its shape and strides.
fn load_numpy_metadata<'a>(array: *const DoubleNumpyArray)
 -> (Vec<f64>, &'a [usize], &'a [usize]) {
    let array_vec =
        unsafe {
            slice::from_raw_parts((*array).ptr, (*array).size).to_vec()
        };
    let shape =
        unsafe { slice::from_raw_parts((*array).shape, (*array).ndim) };
    let strides =
        unsafe { slice::from_raw_parts((*array).strides, (*array).ndim) };
    (array_vec, shape, strides)
}
/// Loads a  2D Numpy array as a Rust's ndarray array. This function creates an
/// [Array2](https://docs.rs/ndarray/*/ndarray/type.Array2.html) instead of
/// an [ArrayD](https://docs.rs/ndarray/*/ndarray/type.ArrayD.html).
fn load_numpy_2d(array: *const DoubleNumpyArray) -> Array2<f64> {
    let (array_vec, shape, strides) = load_numpy_metadata(array);
    let mut arr_shape = [0; 2];
    arr_shape.copy_from_slice(shape);
    let mut arr_strides = [0; 2];
    arr_strides.copy_from_slice(strides);
    arr_strides.iter_mut().for_each(|s| *s = *s / mem::size_of::<f64>());
    unsafe {
        Array2::from_shape_vec_unchecked(arr_shape.strides(arr_strides),
                                         array_vec)
    }
}
/// This enum represents the posible errors that can arise during the execution of the KDE.
/// The error codes can be used by Python to give an adequate response.
#[repr(C)]
pub enum Error { NoError = 0, MemoryError = 1, NotFinished = 2, }
#[automatically_derived]
#[allow(unused_qualifications)]
impl ::core::cmp::PartialEq for Error {
    #[inline]
    fn eq(&self, other: &Error) -> bool {
        {
            let __self_vi =
                unsafe { ::core::intrinsics::discriminant_value(&*self) } as
                    i32;
            let __arg_1_vi =
                unsafe { ::core::intrinsics::discriminant_value(&*other) } as
                    i32;
            if true && __self_vi == __arg_1_vi {
                match (&*self, &*other) { _ => true, }
            } else { false }
        }
    }
}
/// Implements a KDE density model in OpenCL.
pub struct GaussianKDE {
    /// Number of train instances.
    n: usize,
    /// Dimensionality of the training data.
    d: usize,
    /// Buffer containing the training data.
    training_data: Buffer<f64>,
    /// Buffer containing the Cholesky decomposition of the covariance matrix of the KDE.
    chol_cov: Buffer<f64>,
    /// Log of the normalization factor of the Gaussian.
    /// That is:
    /// ```math
    /// \log (2\pi)^{d/2} \sqrt{\lvert\mathbf{\Sigma}}\rvert
    /// ```
    lognorm_factor: f64,
    rowmajor: bool,
    leading_dimension: usize,
}
#[automatically_derived]
#[allow(unused_qualifications)]
impl ::core::fmt::Debug for GaussianKDE {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        match *self {
            GaussianKDE {
            n: ref __self_0_0,
            d: ref __self_0_1,
            training_data: ref __self_0_2,
            chol_cov: ref __self_0_3,
            lognorm_factor: ref __self_0_4,
            rowmajor: ref __self_0_5,
            leading_dimension: ref __self_0_6 } => {
                let mut debug_trait_builder = f.debug_struct("GaussianKDE");
                let _ = debug_trait_builder.field("n", &&(*__self_0_0));
                let _ = debug_trait_builder.field("d", &&(*__self_0_1));
                let _ =
                    debug_trait_builder.field("training_data",
                                              &&(*__self_0_2));
                let _ =
                    debug_trait_builder.field("chol_cov", &&(*__self_0_3));
                let _ =
                    debug_trait_builder.field("lognorm_factor",
                                              &&(*__self_0_4));
                let _ =
                    debug_trait_builder.field("rowmajor", &&(*__self_0_5));
                let _ =
                    debug_trait_builder.field("leading_dimension",
                                              &&(*__self_0_6));
                debug_trait_builder.finish()
            }
        }
    }
}
/// Gets the maximum work group size of the default device. This is the preferred local work size
/// for many operations (especially reductions, such as sums or finding maximums).
fn get_max_work_size(pro_que: &ProQue) -> usize {
    match pro_que.device().info(DeviceInfo::MaxWorkGroupSize).expect("The maximum local work size could not be detected.")
        {
        DeviceInfoResult::MaxWorkGroupSize(s) => s,
        _ => {
            {
                ::std::rt::begin_panic("internal error: entered unreachable code",
                                       &("src/lib.rs", 191u32, 14u32))
            }
        }
    }
}
/// Fills a given OpenCL buffer with a value.
fn buffer_fill_value(pro_que: &ProQue, b: &Buffer<f64>, size: usize,
                     value: f64) {
    let kernel_zeros =
        pro_que.kernel_builder("fill_value").global_work_size(size).arg(b).arg(value).build().expect("Final result initialization failed.");
    unsafe { kernel_zeros.enq().expect("Error while filling the buffer."); }
}
/// Computes the lognorm factor: See [KDEDensityOcl](struct.KDEDensityOcl.html)
fn lognorm_factor(n: usize, d: usize, chol_cov: &Array2<f64>) -> f64 {
    let norm_factor =
        0.5 * (d as f64) * (2.0 * f64::consts::PI).ln() + (n as f64).ln();
    norm_factor +
        chol_cov.diag().fold(0., |accum: f64, v| accum + (*v as f64).ln())
}
#[no_mangle]
pub unsafe extern "C" fn new_proque() -> *mut ProQue {
    let pro_que =
        ProQue::builder().src(open_cl_code::OPEN_CL_CODE).build().expect("Error while creating OpenCL ProQue.");
    let proque_box = Box::new(pro_que);
    let ptr = Box::into_raw(proque_box);
    ptr
}
/// Initializes a `KDEDensityOcl`. It expects two `DoubleNumpyArray` with the cholesky decomposition
/// of the covariance matrix and the training data. The training data is expected to have shape
/// ($`n`$, $`d`$), where n is the number of instances and $`d`$ is the number of atributes. The
/// covariance matrix should have shape ($`d`$,$`d`$).
///
/// # Safety
///
/// This function is unsafe because it receives some Numpy array pointers wrapped in the
/// `DoubleNumpyArray` struct. Those matrices should not point to invalid data.
#[no_mangle]
pub unsafe extern "C" fn gaussian_kde_init(pro_que: *mut ProQue,
                                           chol: *const DoubleNumpyArray,
                                           training_data:
                                               *const DoubleNumpyArray,
                                           error: *mut Error)
 -> *mut GaussianKDE {
    *error = Error::NotFinished;
    let n = *(*training_data).shape;
    let d = *(*chol).shape;
    let chol_cov = load_numpy_2d(chol);
    let lognorm_factor = lognorm_factor(n, d, &chol_cov);
    let chol_vec = chol_cov.into_raw_vec();
    let pro_que = Box::from_raw(pro_que);
    let training_slice =
        slice::from_raw_parts((*training_data).ptr, (*training_data).size);
    let (training_buffer, chol_buffer) =
        {
            (match Buffer::builder().context(pro_que.context()).len(training_slice.len()).copy_host_slice(&training_slice).build()
                 {
                 Ok(b) => b,
                 Err(_) => {
                     *error = Error::MemoryError;
                     return ptr::null_mut();
                 }
             },
             match Buffer::builder().context(pro_que.context()).len(chol_vec.len()).copy_host_slice(&chol_vec).build()
                 {
                 Ok(b) => b,
                 Err(_) => {
                     *error = Error::MemoryError;
                     return ptr::null_mut();
                 }
             })
        };
    let (rowmajor, leading_dimension) = is_rowmajor(training_data);
    let kde =
        Box::new(GaussianKDE{n,
                             d,
                             training_data: training_buffer,
                             chol_cov: chol_buffer,
                             lognorm_factor,
                             rowmajor,
                             leading_dimension,});
    let ptr_kde = Box::into_raw(kde);
    Box::into_raw(pro_que);
    *error = Error::NoError;
    ptr_kde
}
/// Frees the `KDEDensityOcl`.
#[no_mangle]
pub extern "C" fn gaussian_kde_free(kde: *mut GaussianKDE) {
    if kde.is_null() { return; }
    unsafe { Box::from_raw(kde); }
}
/// Frees the `ProQue`.
#[no_mangle]
pub extern "C" fn gaussian_proque_free(pro_que: *mut ProQue) {
    if pro_que.is_null() { return; }
    unsafe { Box::from_raw(pro_que); }
}
/// Checks if `x` is stored as a rowmajor array in memory. It also returns the dimension of the
/// leading axis: columns if rowmajor, rows if column major.
unsafe fn is_rowmajor(x: *const DoubleNumpyArray) -> (bool, usize) {
    let row_stride = *(*x).strides;
    let column_stride = *(*x).strides.offset(1);
    if row_stride > column_stride {
        (true, *(*x).shape.offset(1))
    } else { (false, *(*x).shape) }
}
/// Returns the name of the kernel `substract` depending on whether the train/test datasets are
/// stored in rowmajor or columnmajor.
fn kernel_substract_name(train_rowmajor: bool, test_rowmajor: bool)
 -> &'static str {
    if train_rowmajor {
        if test_rowmajor {
            "substract_rowmajor_rowmajor"
        } else { "substract_rowmajor_columnmajor" }
    } else {
        if test_rowmajor {
            "substract_columnmajor_rowmajor"
        } else { "substract_columnmajor_columnmajor" }
    }
}
/// Computes the probability density function (pdf) evaluation of $`m`$ points given a KDE model.
/// The $`m`$ testing points are in the `testing_data` `DoubleNumpyArray` with shape ($`m`$, $`d`$).
/// The result is saved in the `result` array, that should have at least length $`m`$.
///
/// # Safety
///
/// This function is unsafe because it receives a Numpy array pointers wrapped in the
/// `DoubleNumpyArray` struct. Those matrices should not point to invalid data. Also, the kde and
/// result pointers should not point to NULL.
///
/// # Implementation
///
/// To compute the pdf, it iterates over the training data or the test data depending on which data
/// set has more instances. The general procedure is discussed in the
/// [main description](index.html).
///
/// ## Iterating over the test data
///
/// If $`\mathbf{D}`$ is the $`n \times d`$ matrix containing the training instances, and
/// $`\mathbf{t}^{k}`$ is a test instance. We iterate over all the $`\mathbf{t}^{k},\;
/// k=1,\ldots,m`$
///
/// ### Substract kernel
///
/// The substract OpenCL kernel, substracts the test instance from
/// all the training data:
///
/// ```math
/// \mathbf{D} = \begin{bmatrix}
/// d_{11} & \cdots & d_{1d}\\
/// \vdots & \ddots & \vdots\\
/// d_{n1} & \cdots & d_{nd}\\
/// \end{bmatrix},
/// \mathbf{t}^{k} = \begin{bmatrix} t_{1}^{k} & \cdots & t_{d}^{k}\end{bmatrix},\;
/// \text{substract}(\mathbf{D}, \mathbf{t}) = \mathbf{S}^{k} = \begin{bmatrix}
/// d_{11} - t_{1}^{k} & \cdots & d_{1d} - t_{d}^{k}\\
/// \vdots & \ddots & \vdots\\
/// d_{n1} - t_{1}^{k} & \cdots & d_{nd} - t_{d}^{k}\\
/// \end{bmatrix}
/// ```
///
/// ### Solve kernel
///
/// The solve kernel performs forward-solving over the substracted matrix. If $`\mathbf{L}`$ is the:
/// Cholesky matrix, and $`\mathbf{S}`$ the substracted matrix, the solved matrix $`\mathbf{V}`$
/// should hold:
///
/// ```math
///  \mathbf{L}\mathbf{V}^{k} = \mathbf{S}^{k}
/// ```
/// ```math
///  \begin{bmatrix}
/// l_{11} & \mathbf{0} & 0\\
/// \vdots & l_{ii} & \mathbf{0}\\
/// l_{d1} & \cdots & l_{dd}\\
///  \end{bmatrix}
/// \begin{bmatrix}
/// v_{11} & \cdots & v_{1n}\\
/// \vdots & \ddots & \vdots\\
/// v_{d1} & \cdots & v_{dn}\\
/// \end{bmatrix} =
/// \begin{bmatrix}
/// s_{11} & \cdots & s_{1n}\\
/// \vdots & \ddots & \vdots\\
/// s_{d1} & \cdots & s_{dn}\\
/// \end{bmatrix}
/// ```
/// Then,
///
/// ```math
/// \begin{bmatrix}
/// v_{11} & \cdots & v_{1n}\\
/// \vdots & \ddots & \vdots\\
/// v_{d1} & \cdots & v_{dn}\\
/// \end{bmatrix} =
/// \begin{bmatrix}
/// s_{11} / l_{11} & \cdots & v_{1n} / l_{11}\\
/// \vdots & \ddots & \vdots\\
/// \frac{s_{d1} - \sum_{j=1}^{d-1} v_{j1}l_{dj}}{l_{dd}} & \cdots &
/// \frac{s_{dn} - \sum_{j=1}^{d-1} v_{jn}l_{dj}}{l_{dd}}\\
/// \end{bmatrix} =
///
/// \begin{bmatrix} (\mathbf{y}^{1})^{T} & \cdots (\mathbf{y}^{i})^{T} &
/// \cdots (\mathbf{y}^{n})^{T}\end{bmatrix}
/// ```
///
/// ### Square kernel
///
/// The square kernel squares every element of a matrix. So,
/// ```math
/// \begin{bmatrix}
/// v_{11} & \cdots & v_{1n}\\
/// \vdots & \ddots & \vdots\\
/// v_{d1} & \cdots & v_{dn}\\
/// \end{bmatrix},\; \text{square}(\mathbf{V}) = \mathbf{W}^{k} =
/// \begin{bmatrix}
/// v_{11}^{2} & \cdots & v_{1n}^{2}\\
/// \vdots & \ddots & \vdots\\
/// v_{d1}^{2} & \cdots & v_{dn}^{2}\\
/// \end{bmatrix}
/// ```
///
/// ### Sumout kernel
///
/// It sums over all the rows in the $`\mathbf{V}`$ matrix and exponentiates the sum.
///
/// ```math
/// \mathbf{W} = \begin{bmatrix}
/// w_{11} & \cdots & w_{1n}\\
/// \vdots & \ddots & \vdots\\
/// w_{d1} & \cdots & w_{dn}\\
/// \end{bmatrix},\;
/// \text{sumout}(\mathbf{W}) =
/// \mathbf{u}^{k} = \begin{bmatrix}
/// \exp\left(\sum_{i=1}^{d} w_{i1}\right) & \cdots & \exp\left(\sum_{i=1}^{d} w_{in}\right)\\
/// \end{bmatrix}
/// ```
///
/// ### sum_gpu function
///
/// It sums all the elements in a vector. The sum of the vector $`\mathbf{u}`$ is the pdf for the
/// instance $`\mathbf{t}`$:
///
/// ```math
/// \hat{f}_{n}(\mathbf{t}) = \sum_{i=1}^{n} u_{i}^{k}
/// ```
/// ## Iterating over the train data
///
/// When iterating over the train data, the kernels `substract`, `solve`, `square` and `sumout` are
/// applied exactly as in [Iterating over the test data](#iterating-over-the-test-data), but in
/// this case the test data and the train instance $`\mathbf{r}^{k},\; k=1,\ldots,n`$
/// is substracted.
///
/// Then the vector $`\mathbf{u}^{k}`$ represents the contribution of the $`\mathbf{r}^{k}`$
/// instance to every other test instance.
///
/// ### Sum_vectors kernel
///
/// The pdf result for a test instance implies the sum over all the $`\mathbf{u}^{k}`$ vectors:
///
/// ```math
///     \hat{f}_{n}(\mathbf{t}^{j}) = \sum_{k=1}^{n} u_{i}^{j}
/// ```
///
/// These sums are performed all at once using the sum_vectors kernel, that sums two vectors. If
/// $`\mathbf{v} = \begin{bmatrix} v_{1} & \cdots & v_{n}\end{bmatrix}`$ and
/// $`\mathbf{w} = \begin{bmatrix} w_{1} & \cdots & w_{n}\end{bmatrix}`$, then:
///
/// ```math
///     \text{sum\_vectors}(\mathbf{v}, \mathbf{w}) = \begin{bmatrix} v_{1} + w_{1} & \cdots &
///         v_{n} + w_{n}\end{bmatrix}
/// ```
#[no_mangle]
pub unsafe extern "C" fn gaussian_kde_pdf(kde: *mut GaussianKDE,
                                          pro_que: *mut ProQue,
                                          testing_data:
                                              *const DoubleNumpyArray,
                                          result: *mut c_double,
                                          error: *mut Error) {
    *error = Error::NotFinished;
    let kde = Box::from_raw(kde);
    let pro_que = Box::from_raw(pro_que);
    let m = *(*testing_data).shape;
    let d = kde.d;
    let max_work_size = get_max_work_size(&pro_que);
    let len_iteration = if kde.n >= m { kde.n } else { m };
    let local_size =
        if len_iteration < max_work_size {
            len_iteration
        } else { max_work_size };
    let num_groups =
        (len_iteration as f32 / local_size as f32).ceil() as usize;
    let test_slice = slice::from_raw_parts((*testing_data).ptr, m * d);
    let (test_instances_buffer,) =
        {
            (match Buffer::builder().context(pro_que.context()).len(test_slice.len()).copy_host_slice(&test_slice).build()
                 {
                 Ok(b) => b,
                 Err(_) => { *error = Error::MemoryError; return; }
             },)
        };
    let (test_rowmajor, test_leading_dimension) = is_rowmajor(testing_data);
    let (final_result_buffer, tmp_matrix_buffer, tmp_vec_buffer) =
        {
            (match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                 {
                 Ok(b) => b,
                 Err(_) => { *error = Error::MemoryError; return; }
             },
             match Buffer::<f64>::builder().context(pro_que.context()).len(len_iteration
                                                                               *
                                                                               d).build()
                 {
                 Ok(b) => b,
                 Err(_) => { *error = Error::MemoryError; return; }
             },
             match Buffer::<f64>::builder().context(pro_que.context()).len(len_iteration).build()
                 {
                 Ok(b) => b,
                 Err(_) => { *error = Error::MemoryError; return; }
             })
        };
    let kernel_substract =
        {
            let (matrix_buf, vec_buf, mat_ld, vec_ld, name) =
                if kde.n >= m {
                    (&kde.training_data, &test_instances_buffer,
                     kde.leading_dimension, test_leading_dimension,
                     kernel_substract_name(kde.rowmajor, test_rowmajor))
                } else {
                    (&test_instances_buffer, &kde.training_data,
                     test_leading_dimension, kde.leading_dimension,
                     kernel_substract_name(test_rowmajor, kde.rowmajor))
                };
            pro_que.kernel_builder(name).global_work_size(len_iteration *
                                                              d).arg(matrix_buf).arg(d
                                                                                         as
                                                                                         u32).arg(vec_buf).arg(&tmp_matrix_buffer).arg_named("row",
                                                                                                                                             &0u32).arg(mat_ld
                                                                                                                                                            as
                                                                                                                                                            u32).arg(vec_ld
                                                                                                                                                                         as
                                                                                                                                                                         u32).build().expect("Kernel substract build failed.")
        };
    let kernel_solve =
        pro_que.kernel_builder("solve").global_work_size(len_iteration).arg(&tmp_matrix_buffer).arg(&kde.chol_cov).arg(d
                                                                                                                           as
                                                                                                                           u32).build().expect("Kernel solve build failed.");
    let kernel_square =
        pro_que.kernel_builder("square").global_work_size(len_iteration *
                                                              d).arg(&tmp_matrix_buffer).build().expect("Kernel square build failed.");
    let kernel_sumout =
        pro_que.kernel_builder("sumout").global_work_size(len_iteration).arg(&tmp_matrix_buffer).arg(&tmp_vec_buffer).arg(d
                                                                                                                              as
                                                                                                                              u32).arg(kde.lognorm_factor).build().expect("Kernel sumout build failed.");
    if kde.n >= m {
        for i in 0..m {
            kernel_substract.set_arg("row", i as u32).unwrap();
            kernel_substract.enq().expect("Error while executing substract kernel.");
            kernel_solve.enq().expect("Error while executing solve kernel.");
            kernel_square.enq().expect("Error while executing square kernel.");
            kernel_sumout.enq().expect("Error while executing sumout kernel.");
            sum_gpu_vec(&pro_que, &tmp_vec_buffer, len_iteration,
                        max_work_size, local_size, num_groups);
            tmp_vec_buffer.copy(&final_result_buffer, Some(i),
                                Some(1)).queue(pro_que.queue()).enq().expect("Error copying to result buffer.");
        }
    } else {
        buffer_fill_value(&pro_que, &final_result_buffer, m, 0.0f64);
        for i in 0..kde.n {
            kernel_substract.set_arg("row", i as u32).unwrap();
            kernel_substract.enq().expect("Error while executing substract kernel.");
            kernel_solve.enq().expect("Error while executing solve kernel.");
            kernel_square.enq().expect("Error while executing square kernel.");
            kernel_sumout.enq().expect("Error while executing sumout kernel.");
            let kernel_sumvectors =
                pro_que.kernel_builder("sum_vectors").global_work_size(m).arg(&final_result_buffer).arg(&tmp_vec_buffer).build().expect("Kernel sum_vectors build failed.");
            kernel_sumvectors.enq().expect("Error while executing sum_vectors kernel.");
        }
    }
    let final_result = slice::from_raw_parts_mut(result, m);
    final_result_buffer.cmd().queue(pro_que.queue()).read(final_result).enq().expect("Error reading result data.");
    *error = Error::NoError;
    Box::into_raw(pro_que);
    Box::into_raw(kde);
}
/// Sums all the elements in the vector buffer `sum_buffer` and places the result in the first
/// position of the `sum_buffer` (i.e., `sum_buffer[0]`). Keep in mind that the rest of the elements
/// of the buffer will be modified, so **it invalidates the rest of the data in the buffer**.
///
/// `global_size` is the length of the `sum_buffer`. `max_work_size` is the maximum
/// number of work items in a work group for the selected device. `local_size` is the actual number
/// of work items in each work group. `num_groups` is the actual number of work groups.
///
/// So, if `sum_buffer` is equal to:
///
/// ```math
///     \begin{bmatrix} a_{1} & \ldots & a_{n}\end{bmatrix}
/// ```
///
/// After calling `sum_gpu_vec`, `sum_buffer` will be equal to:
///
/// ```math
///     \begin{bmatrix} \sum_{i=1}^{n} a_{i} & \ldots & \text{unexpected values} & \ldots
/// \end{bmatrix}
/// ```
fn sum_gpu_vec(pro_que: &ProQue, sum_buffer: &Buffer<f64>,
               mut global_size: usize, max_work_size: usize,
               mut local_size: usize, mut num_groups: usize) {
    while global_size > 1 {
        let kernel_sum_gpu =
            pro_que.kernel_builder("sum_gpu_vec").global_work_size(global_size).local_work_size(local_size).arg(sum_buffer).arg_local::<f64>(local_size).build().expect("Kernel sum_gpu_vec build failed.");
        unsafe {
            kernel_sum_gpu.enq().expect("Error while executing sum_gpu_vec kernel.");
        }
        global_size = num_groups;
        local_size =
            if global_size < max_work_size {
                global_size
            } else { max_work_size };
        num_groups = (global_size as f32 / local_size as f32).ceil() as usize;
    }
}
/// Computes the logarithm of the probability density function (log pdf) evaluation of $`m`$ points
/// given a KDE model. The $`m`$ testing points are in the `testing_data` `DoubleNumpyArray` with
/// shape ($`m`$, $`d`$). The result is saved in the `result` array, that should have at least
/// length $`m`$.
///
/// The [LogSumExp trick](https://en.wikipedia.org/wiki/LogSumExp) is used instead of applying a
/// logarithm to the results of `KDEDensity_logpdf_multiple_ocl` to obtain better precision.
///
/// # Safety
///
/// This function is unsafe because it receives a Numpy array pointers wrapped in the
/// `DoubleNumpyArray` struct. Those matrices should not point to invalid data. Also, the kde and
/// result pointers should not point to NULL.
///
/// # Implementation
///
/// To compute the pdf, it iterates over the training data or the test data depending on which data
/// set has more instances. The general procedure is discussed in the
/// [main description](index.html).
///
/// ## Iterating over the test data
///
/// If $`\mathbf{D}`$ is the $`n \times d`$ matrix containing the training instances, and
/// $`\mathbf{t}^{k}`$ is a test instance. We iterate over all the $`\mathbf{t}^{k},\;
/// k=1,\ldots,m`$
///
/// The first steps are as in [gaussian_kde_pdf](fn.gaussian_kde_pdf.html), applying the kernels
/// `substract`, `solve` and `square` exactly the same. However, instead of executing the `sumout`
/// kernel, the `logsumout` kernel is applied.
///
/// ### Logsumout kernel
///
/// It sums over all the rows in the $`\mathbf{V}`$ matrix without exponentiating the sum.
///
/// ```math
/// \mathbf{W} = \begin{bmatrix}
/// w_{11} & \cdots & w_{1n}\\
/// \vdots & \ddots & \vdots\\
/// w_{d1} & \cdots & w_{dn}\\
/// \end{bmatrix},\;
/// \text{logsumout}(\mathbf{W}) =
/// \mathbf{u}^{k} = \begin{bmatrix}
/// \sum_{i=1}^{d} w_{i1} & \cdots & \sum_{i=1}^{d} w_{in}\\
/// \end{bmatrix}
/// ```
///
/// ### max_gpu_vec_copy function
///
/// The `max_gpu_vec_copy` finds the maximum of the $`\mathbf{u}^{k}`$ vector.
/// [Its documentation](fn.max_gpu_vec_copy.html) contains more details about this function.
///
/// ```math
///     \text{max\_gpu\_vec\_copy}(\mathbf{u}^{k}) = \text{maxexp}= \max_{i} u_{i}^{k}
/// ```
/// The max value in the $`\mathbf{u}^{k}`$ vector is necessary to apply the
/// [LogSumExp trick](index.html#logsumexp-trick).
///
/// ### log_sum_gpu_vec function
///
/// The `log_sum_gpu_vec` sums the exponentiation of every element in the vector $`\mathbf{u}^{k}`$
/// substracting its maximum value. [Its documentation](fn.log_sum_gpu_vec.html) contains more
/// details about this function.
///
/// ```math
///     \text{log\_sum\_gpu\_vec}(\mathbf{u}^{k}) = \sum_{i=1}^{m}\exp\left(u_{i}^{k} -
///     \text{maxexp}\right)
/// ```
/// The substraction before the exponentiation is necessary to apply the
/// [LogSumExp trick](index.html#logsumexp-trick).
///
/// ### copy_log_pdf_result
///
/// It exponentiates the result of the `log_sum_gpu_vec` function and sums `maxexp`, to obtain the
/// log pdf for $`\mathbf{t}^{k}`$
///
/// ## Iterating over the train data
///
/// When iterating over the train data, there are two modes of behaviour depending on the available
/// memory in the OpenCL device as described in the
/// [log_pdf_iterate_train](fn.logpdf_iterate_train.html) documentation.
///
///
/// ### Low memory mode
///
/// If there is no enough memory to hold the logpdf for every pair of train and test instances, it
/// iterates twice over the train data.
///
/// In the first iteration, it finds the `maxexp` using a
/// different version of the `logsumout` kernel, that also saves the `maxexp` for each test
/// instance, called `logsumout_checkmax`. The previous steps are as in the
/// [Iterating over the test data](fn.gaussian_kde_logpdf.html#iterating-over-the-test-data) (i.e.,
/// applying the `substract`, `solve` and `square` kernels)
///
/// In the second iteration, it uses the `maxexp` to apply the
/// [LogSumExp trick](index.html#logsumexp-trick) with the `exp_and_sum` and `log_and_sum` kernels.
///
/// ### High memory mode
///
/// In this mode, the logpdf of each train and test instance pair is saved in a $`m \times n`$
/// matrix. This matrix is filled using the `substract`, `solve` and `square` while iterating over
/// all the train data and a new version of the
/// [logsumout kernel](fn.gaussian_kde_logpdf.html#logsumout-kernel) that works in the matrix called
/// `logsumout_to_matrix`.
///
/// Once the matrix is filled, the function `max_gpu_mat` finds the maximum element over each row
/// of the matrix to find the `maxexp` of each test instance. Then, the `exp_and_sum_mat` kernel is
/// executed (a modified kernel of the `exp_and_sum` kernel designed to work in matrices by making
/// the sum over every row after exponentiating the logpdf substracted with `maxexp`). Finally, the
/// `sum_gpu_mat` sums every exponentiated logpdf and the kernel `log_and_sum_mat` makes the
/// logarithm of the previous step result and sums `maxexp`.
#[no_mangle]
pub unsafe extern "C" fn gaussian_kde_logpdf(kde: *mut GaussianKDE,
                                             pro_que: *mut ProQue,
                                             x: *const DoubleNumpyArray,
                                             result: *mut c_double,
                                             error: *mut Error) {
    *error = Error::NotFinished;
    let mut kde_box = Box::from_raw(kde);
    let mut pro_que = Box::from_raw(pro_que);
    let m = *(*x).shape;
    if kde_box.n >= m {
        logpdf_iterate_test(&mut kde_box, &mut pro_que, x, result, error);
    } else {
        logpdf_iterate_train(&mut kde_box, &mut pro_que, x, result, error);
    }
    Box::into_raw(kde_box);
    Box::into_raw(pro_que);
}
/// We iterate over the test points if there are more training points.
unsafe fn logpdf_iterate_test(kde: &mut Box<GaussianKDE>,
                              pro_que: &mut Box<ProQue>,
                              x: *const DoubleNumpyArray,
                              result: *mut c_double, error: *mut Error) {
    let m = *(*x).shape;
    let d = kde.d;
    let n = kde.n;
    {
        ::std::io::_print(::core::fmt::Arguments::new_v1(&["m = ", ", d = ",
                                                           ", n = ", "\n"],
                                                         &match (&m, &d, &n) {
                                                              (arg0, arg1,
                                                               arg2) =>
                                                              [::core::fmt::ArgumentV1::new(arg0,
                                                                                            ::core::fmt::Display::fmt),
                                                               ::core::fmt::ArgumentV1::new(arg1,
                                                                                            ::core::fmt::Display::fmt),
                                                               ::core::fmt::ArgumentV1::new(arg2,
                                                                                            ::core::fmt::Display::fmt)],
                                                          }));
    };
    let max_work_size = get_max_work_size(&pro_que);
    {
        ::std::io::_print(::core::fmt::Arguments::new_v1(&["max_work_size = ",
                                                           "\n"],
                                                         &match (&max_work_size,)
                                                              {
                                                              (arg0,) =>
                                                              [::core::fmt::ArgumentV1::new(arg0,
                                                                                            ::core::fmt::Display::fmt)],
                                                          }));
    };
    let local_work_size = if n < max_work_size { n } else { max_work_size };
    {
        ::std::io::_print(::core::fmt::Arguments::new_v1(&["local_work_size = ",
                                                           "\n"],
                                                         &match (&local_work_size,)
                                                              {
                                                              (arg0,) =>
                                                              [::core::fmt::ArgumentV1::new(arg0,
                                                                                            ::core::fmt::Display::fmt)],
                                                          }));
    };
    let num_groups = (n as f32 / local_work_size as f32).ceil() as usize;
    {
        ::std::io::_print(::core::fmt::Arguments::new_v1(&["num_groups = ",
                                                           "\n"],
                                                         &match (&num_groups,)
                                                              {
                                                              (arg0,) =>
                                                              [::core::fmt::ArgumentV1::new(arg0,
                                                                                            ::core::fmt::Display::fmt)],
                                                          }));
    };
    let test_slice = slice::from_raw_parts((*x).ptr, m * d);
    let (test_instances_buffer,) =
        {
            (match Buffer::builder().context(pro_que.context()).len(test_slice.len()).copy_host_slice(&test_slice).build()
                 {
                 Ok(b) => b,
                 Err(_) => { *error = Error::MemoryError; return; }
             },)
        };
    let len_buffer = kde.training_data.len();
    let mut vec = ::alloc::vec::from_elem(Default::default(), len_buffer);
    kde.training_data.cmd().queue(pro_que.queue()).read(&mut vec).enq().expect("Error reading result data.");
    if vec.len() > 20 {
        {
            ::std::io::_print(::core::fmt::Arguments::new_v1(&["", " = [",
                                                               ", ", "]\n"],
                                                             &match (&"kde.training_data",
                                                                     &&vec[..10],
                                                                     &&vec[vec.len()
                                                                               -
                                                                               10..])
                                                                  {
                                                                  (arg0, arg1,
                                                                   arg2) =>
                                                                  [::core::fmt::ArgumentV1::new(arg0,
                                                                                                ::core::fmt::Display::fmt),
                                                                   ::core::fmt::ArgumentV1::new(arg1,
                                                                                                ::core::fmt::Debug::fmt),
                                                                   ::core::fmt::ArgumentV1::new(arg2,
                                                                                                ::core::fmt::Debug::fmt)],
                                                              }));
        };
    } else {
        {
            ::std::io::_print(::core::fmt::Arguments::new_v1(&["", " = ",
                                                               "\n"],
                                                             &match (&"kde.training_data",
                                                                     &&vec) {
                                                                  (arg0, arg1)
                                                                  =>
                                                                  [::core::fmt::ArgumentV1::new(arg0,
                                                                                                ::core::fmt::Display::fmt),
                                                                   ::core::fmt::ArgumentV1::new(arg1,
                                                                                                ::core::fmt::Debug::fmt)],
                                                              }));
        };
    }
    let len_buffer = test_instances_buffer.len();
    let mut vec = ::alloc::vec::from_elem(Default::default(), len_buffer);
    test_instances_buffer.cmd().queue(pro_que.queue()).read(&mut vec).enq().expect("Error reading result data.");
    if vec.len() > 20 {
        {
            ::std::io::_print(::core::fmt::Arguments::new_v1(&["", " = [",
                                                               ", ", "]\n"],
                                                             &match (&"test_instances_buffer",
                                                                     &&vec[..10],
                                                                     &&vec[vec.len()
                                                                               -
                                                                               10..])
                                                                  {
                                                                  (arg0, arg1,
                                                                   arg2) =>
                                                                  [::core::fmt::ArgumentV1::new(arg0,
                                                                                                ::core::fmt::Display::fmt),
                                                                   ::core::fmt::ArgumentV1::new(arg1,
                                                                                                ::core::fmt::Debug::fmt),
                                                                   ::core::fmt::ArgumentV1::new(arg2,
                                                                                                ::core::fmt::Debug::fmt)],
                                                              }));
        };
    } else {
        {
            ::std::io::_print(::core::fmt::Arguments::new_v1(&["", " = ",
                                                               "\n"],
                                                             &match (&"test_instances_buffer",
                                                                     &&vec) {
                                                                  (arg0, arg1)
                                                                  =>
                                                                  [::core::fmt::ArgumentV1::new(arg0,
                                                                                                ::core::fmt::Display::fmt),
                                                                   ::core::fmt::ArgumentV1::new(arg1,
                                                                                                ::core::fmt::Debug::fmt)],
                                                              }));
        };
    };
    let (max_buffer, final_result_buffer, tmp_matrix_buffer, tmp_vec_buffer) =
        {
            (match Buffer::<f64>::builder().context(pro_que.context()).len(num_groups).build()
                 {
                 Ok(b) => b,
                 Err(_) => { *error = Error::MemoryError; return; }
             },
             match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                 {
                 Ok(b) => b,
                 Err(_) => { *error = Error::MemoryError; return; }
             },
             match Buffer::<f64>::builder().context(pro_que.context()).len(n *
                                                                               d).build()
                 {
                 Ok(b) => b,
                 Err(_) => { *error = Error::MemoryError; return; }
             },
             match Buffer::<f64>::builder().context(pro_que.context()).len(n).build()
                 {
                 Ok(b) => b,
                 Err(_) => { *error = Error::MemoryError; return; }
             })
        };
    let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);
    let substract_name = kernel_substract_name(kde.rowmajor, test_rowmajor);
    let kernel_substract =
        pro_que.kernel_builder(substract_name).global_work_size(n *
                                                                    d).arg(&kde.training_data).arg(d
                                                                                                       as
                                                                                                       u32).arg(&test_instances_buffer).arg(&tmp_matrix_buffer).arg_named("row",
                                                                                                                                                                          &0u32).arg(kde.leading_dimension
                                                                                                                                                                                         as
                                                                                                                                                                                         u32).arg(test_leading_dimension
                                                                                                                                                                                                      as
                                                                                                                                                                                                      u32).build().expect("Kernel substract build failed.");
    let kernel_solve =
        pro_que.kernel_builder("solve").global_work_size(n).arg(&tmp_matrix_buffer).arg(&kde.chol_cov).arg(d
                                                                                                               as
                                                                                                               u32).build().expect("Kernel solve build failed.");
    let kernel_square =
        pro_que.kernel_builder("square").global_work_size(n *
                                                              d).arg(&tmp_matrix_buffer).build().expect("Kernel square build failed.");
    let kernel_sumout =
        pro_que.kernel_builder("logsumout").global_work_size(n).arg(&tmp_matrix_buffer).arg(&tmp_vec_buffer).arg(d
                                                                                                                     as
                                                                                                                     u32).arg(kde.lognorm_factor).build().expect("Kernel logsumout build failed.");
    let kernel_log_sum_gpu =
        pro_que.kernel_builder("copy_logpdf_result").global_work_size(1).arg(&tmp_vec_buffer).arg(&max_buffer).arg(&final_result_buffer).arg_named("offset",
                                                                                                                                                   &0u32).build().expect("Kernel copy_logpdf_result build failed.");
    for i in 0..m {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract.enq().expect("Error while executing substract kernel.");
        if i == 0 {
            let len_buffer = tmp_matrix_buffer.len();
            let mut vec =
                ::alloc::vec::from_elem(Default::default(), len_buffer);
            tmp_matrix_buffer.cmd().queue(pro_que.queue()).read(&mut vec).enq().expect("Error reading result data.");
            if vec.len() > 20 {
                {
                    ::std::io::_print(::core::fmt::Arguments::new_v1(&["",
                                                                       " = [",
                                                                       ", ",
                                                                       "]\n"],
                                                                     &match (&"tmp_matrix_buffer",
                                                                             &&vec[..10],
                                                                             &&vec[vec.len()
                                                                                       -
                                                                                       10..])
                                                                          {
                                                                          (arg0,
                                                                           arg1,
                                                                           arg2)
                                                                          =>
                                                                          [::core::fmt::ArgumentV1::new(arg0,
                                                                                                        ::core::fmt::Display::fmt),
                                                                           ::core::fmt::ArgumentV1::new(arg1,
                                                                                                        ::core::fmt::Debug::fmt),
                                                                           ::core::fmt::ArgumentV1::new(arg2,
                                                                                                        ::core::fmt::Debug::fmt)],
                                                                      }));
                };
            } else {
                {
                    ::std::io::_print(::core::fmt::Arguments::new_v1(&["",
                                                                       " = ",
                                                                       "\n"],
                                                                     &match (&"tmp_matrix_buffer",
                                                                             &&vec)
                                                                          {
                                                                          (arg0,
                                                                           arg1)
                                                                          =>
                                                                          [::core::fmt::ArgumentV1::new(arg0,
                                                                                                        ::core::fmt::Display::fmt),
                                                                           ::core::fmt::ArgumentV1::new(arg1,
                                                                                                        ::core::fmt::Debug::fmt)],
                                                                      }));
                };
            };
        }
        kernel_solve.enq().expect("Error while executing solve kernel.");
        if i == 0 {
            let len_buffer = tmp_matrix_buffer.len();
            let mut vec =
                ::alloc::vec::from_elem(Default::default(), len_buffer);
            tmp_matrix_buffer.cmd().queue(pro_que.queue()).read(&mut vec).enq().expect("Error reading result data.");
            if vec.len() > 20 {
                {
                    ::std::io::_print(::core::fmt::Arguments::new_v1(&["",
                                                                       " = [",
                                                                       ", ",
                                                                       "]\n"],
                                                                     &match (&"tmp_matrix_buffer",
                                                                             &&vec[..10],
                                                                             &&vec[vec.len()
                                                                                       -
                                                                                       10..])
                                                                          {
                                                                          (arg0,
                                                                           arg1,
                                                                           arg2)
                                                                          =>
                                                                          [::core::fmt::ArgumentV1::new(arg0,
                                                                                                        ::core::fmt::Display::fmt),
                                                                           ::core::fmt::ArgumentV1::new(arg1,
                                                                                                        ::core::fmt::Debug::fmt),
                                                                           ::core::fmt::ArgumentV1::new(arg2,
                                                                                                        ::core::fmt::Debug::fmt)],
                                                                      }));
                };
            } else {
                {
                    ::std::io::_print(::core::fmt::Arguments::new_v1(&["",
                                                                       " = ",
                                                                       "\n"],
                                                                     &match (&"tmp_matrix_buffer",
                                                                             &&vec)
                                                                          {
                                                                          (arg0,
                                                                           arg1)
                                                                          =>
                                                                          [::core::fmt::ArgumentV1::new(arg0,
                                                                                                        ::core::fmt::Display::fmt),
                                                                           ::core::fmt::ArgumentV1::new(arg1,
                                                                                                        ::core::fmt::Debug::fmt)],
                                                                      }));
                };
            };
        }
        kernel_square.enq().expect("Error while executing square kernel.");
        if i == 0 {
            let len_buffer = tmp_matrix_buffer.len();
            let mut vec =
                ::alloc::vec::from_elem(Default::default(), len_buffer);
            tmp_matrix_buffer.cmd().queue(pro_que.queue()).read(&mut vec).enq().expect("Error reading result data.");
            if vec.len() > 20 {
                {
                    ::std::io::_print(::core::fmt::Arguments::new_v1(&["",
                                                                       " = [",
                                                                       ", ",
                                                                       "]\n"],
                                                                     &match (&"tmp_matrix_buffer",
                                                                             &&vec[..10],
                                                                             &&vec[vec.len()
                                                                                       -
                                                                                       10..])
                                                                          {
                                                                          (arg0,
                                                                           arg1,
                                                                           arg2)
                                                                          =>
                                                                          [::core::fmt::ArgumentV1::new(arg0,
                                                                                                        ::core::fmt::Display::fmt),
                                                                           ::core::fmt::ArgumentV1::new(arg1,
                                                                                                        ::core::fmt::Debug::fmt),
                                                                           ::core::fmt::ArgumentV1::new(arg2,
                                                                                                        ::core::fmt::Debug::fmt)],
                                                                      }));
                };
            } else {
                {
                    ::std::io::_print(::core::fmt::Arguments::new_v1(&["",
                                                                       " = ",
                                                                       "\n"],
                                                                     &match (&"tmp_matrix_buffer",
                                                                             &&vec)
                                                                          {
                                                                          (arg0,
                                                                           arg1)
                                                                          =>
                                                                          [::core::fmt::ArgumentV1::new(arg0,
                                                                                                        ::core::fmt::Display::fmt),
                                                                           ::core::fmt::ArgumentV1::new(arg1,
                                                                                                        ::core::fmt::Debug::fmt)],
                                                                      }));
                };
            };
        }
        kernel_sumout.enq().expect("Error while executing logsumout kernel.");
        if i == 0 {
            {
                ::std::io::_print(::core::fmt::Arguments::new_v1(&["max_work_size = ",
                                                                   " , local_work_size = ",
                                                                   ", num_groups = ",
                                                                   "\n"],
                                                                 &match (&max_work_size,
                                                                         &local_work_size,
                                                                         &num_groups)
                                                                      {
                                                                      (arg0,
                                                                       arg1,
                                                                       arg2)
                                                                      =>
                                                                      [::core::fmt::ArgumentV1::new(arg0,
                                                                                                    ::core::fmt::Display::fmt),
                                                                       ::core::fmt::ArgumentV1::new(arg1,
                                                                                                    ::core::fmt::Display::fmt),
                                                                       ::core::fmt::ArgumentV1::new(arg2,
                                                                                                    ::core::fmt::Display::fmt)],
                                                                  }));
            };
        }
        if i == 0 {
            let sumout_cpu =
                {
                    ({
                         let len_buffer = tmp_vec_buffer.len();
                         let mut vec =
                             ::alloc::vec::from_elem(Default::default(),
                                                     len_buffer);
                         tmp_vec_buffer.cmd().queue(pro_que.queue()).read(&mut vec).enq().expect("Error reading result data.");
                         vec
                     },)
                };
            let mut max = std::f64::MIN;
            for i in sumout_cpu.iter() { if i > max { max = i; } }
            {
                ::std::io::_print(::core::fmt::Arguments::new_v1(&["max_cpu = ",
                                                                   " \n"],
                                                                 &match (&max,)
                                                                      {
                                                                      (arg0,)
                                                                      =>
                                                                      [::core::fmt::ArgumentV1::new(arg0,
                                                                                                    ::core::fmt::Debug::fmt)],
                                                                  }));
            };
        }
        max_gpu_vec_copy(&pro_que, &tmp_vec_buffer, &max_buffer, n,
                         max_work_size, local_work_size, num_groups);
        if i == 0 {
            {
                ::std::io::_print(::core::fmt::Arguments::new_v1(&["max_work_size = ",
                                                                   " , local_work_size = ",
                                                                   ", num_groups = ",
                                                                   "\n"],
                                                                 &match (&max_work_size,
                                                                         &local_work_size,
                                                                         &num_groups)
                                                                      {
                                                                      (arg0,
                                                                       arg1,
                                                                       arg2)
                                                                      =>
                                                                      [::core::fmt::ArgumentV1::new(arg0,
                                                                                                    ::core::fmt::Display::fmt),
                                                                       ::core::fmt::ArgumentV1::new(arg1,
                                                                                                    ::core::fmt::Display::fmt),
                                                                       ::core::fmt::ArgumentV1::new(arg2,
                                                                                                    ::core::fmt::Display::fmt)],
                                                                  }));
            };
        }
        if i == 0 {
            let len_buffer = max_buffer.len();
            let mut vec =
                ::alloc::vec::from_elem(Default::default(), len_buffer);
            max_buffer.cmd().queue(pro_que.queue()).read(&mut vec).enq().expect("Error reading result data.");
            if vec.len() > 20 {
                {
                    ::std::io::_print(::core::fmt::Arguments::new_v1(&["",
                                                                       " = [",
                                                                       ", ",
                                                                       "]\n"],
                                                                     &match (&"max_buffer",
                                                                             &&vec[..10],
                                                                             &&vec[vec.len()
                                                                                       -
                                                                                       10..])
                                                                          {
                                                                          (arg0,
                                                                           arg1,
                                                                           arg2)
                                                                          =>
                                                                          [::core::fmt::ArgumentV1::new(arg0,
                                                                                                        ::core::fmt::Display::fmt),
                                                                           ::core::fmt::ArgumentV1::new(arg1,
                                                                                                        ::core::fmt::Debug::fmt),
                                                                           ::core::fmt::ArgumentV1::new(arg2,
                                                                                                        ::core::fmt::Debug::fmt)],
                                                                      }));
                };
            } else {
                {
                    ::std::io::_print(::core::fmt::Arguments::new_v1(&["",
                                                                       " = ",
                                                                       "\n"],
                                                                     &match (&"max_buffer",
                                                                             &&vec)
                                                                          {
                                                                          (arg0,
                                                                           arg1)
                                                                          =>
                                                                          [::core::fmt::ArgumentV1::new(arg0,
                                                                                                        ::core::fmt::Display::fmt),
                                                                           ::core::fmt::ArgumentV1::new(arg1,
                                                                                                        ::core::fmt::Debug::fmt)],
                                                                      }));
                };
            };
        }
        log_sum_gpu_vec(&pro_que, &tmp_vec_buffer, &max_buffer, n,
                        max_work_size, local_work_size, num_groups);
        kernel_log_sum_gpu.set_arg("offset", i as u32).unwrap();
        kernel_log_sum_gpu.enq().expect("Error while executing copy_logpdf_result kernel.");
    }
    let final_result = slice::from_raw_parts_mut(result, m);
    final_result_buffer.cmd().queue(pro_que.queue()).read(final_result).enq().expect("Error reading result data.");
    *error = Error::NoError;
}
/// Finds the maximum element in the vector buffer `max_buffer` and places the result in the first
/// position of `result_buffer` (i.e., `result_buffer[0]`). **This operation invalidates the rest
/// of the data in `result_buffer`**, but keeps constant `max_buffer`.
///
/// `global_size` is the length of the `max_buffer`. `max_work_size` is the maximum
/// number of work items in a work group for the selected device. `local_size` is the actual number
/// of work items in each work group. `num_groups` is the actual number of work groups.
///
/// So, if `max_buffer` is equal to:
///
/// ```math
///     \begin{bmatrix} a_{1} & \ldots & a_{n}\end{bmatrix}
/// ```
///
/// After calling `max_gpu_vec_copy`, `max_buffer` will be equal to:
///
/// ```math
///     \begin{bmatrix} a_{1} & \ldots & a_{n}\end{bmatrix}
/// ```
/// and `result_buffer`:
///
/// ```math
///     \begin{bmatrix} \max_{i} a_{i} & \ldots & \text{unexpected values} & \ldots \end{bmatrix}
/// ```
fn max_gpu_vec_copy(pro_que: &ProQue, max_buffer: &Buffer<f64>,
                    result_buffer: &Buffer<f64>, mut global_size: usize,
                    max_work_size: usize, mut local_size: usize,
                    mut num_groups: usize) {
    let kernel_max_gpu =
        pro_que.kernel_builder("max_gpu_vec_copy").global_work_size(global_size).local_work_size(local_size).arg(max_buffer).arg(result_buffer).arg_local::<f64>(local_size).build().expect("Kernel max_gpu_vec_copy build failed.");
    unsafe {
        kernel_max_gpu.enq().expect("Error while executing max_gpu_vec_copy kernel.");
    }
    global_size = num_groups;
    local_size =
        if global_size < max_work_size { global_size } else { max_work_size };
    num_groups = (global_size as f32 / local_size as f32).ceil() as usize;
    while global_size > 1 {
        let kernel_max_gpu =
            pro_que.kernel_builder("max_gpu_vec").global_work_size(global_size).local_work_size(local_size).arg(result_buffer).arg_local::<f64>(local_size).build().expect("Kernel max_gpu_vec build failed.");
        unsafe {
            kernel_max_gpu.enq().expect("Error while executing max_gpu_vec kernel.");
        }
        global_size = num_groups;
        local_size =
            if global_size < max_work_size {
                global_size
            } else { max_work_size };
        num_groups = (global_size as f32 / local_size as f32).ceil() as usize;
    }
}
/// Given a vector buffer `sum_buffer`:
///
/// ```math
///     \begin{bmatrix} s_{1} & \ldots & s_{n}\end{bmatrix}
/// ```
///
/// and a value located in the first position of `maxexp` (i.e., `maxexp[0]`):
///
/// Saves in the first position of `sum_buffer` (i.e. `sum_buffer[0]`), the following expression:
///
/// ```math
///     \sum_{i}^{n} \exp(s_{i} - \text{maxexp})
/// ```
///
/// `global_size` is the length of the `sum_buffer`. `max_work_size` is the maximum
/// number of work items in a work group for the selected device. `local_size` is the actual number
/// of work items in each work group. `num_groups` is the actual number of work groups.
///
/// This operation is a partial step to make to apply the
/// [LogSumExp trick](https://en.wikipedia.org/wiki/LogSumExp). `maxexp[0]` should be the maximum
/// of all the elements in `sum_buffer`. **This operation invalidates the rest of the data in
/// `sum_buffer`**.
fn log_sum_gpu_vec(pro_que: &ProQue, sum_buffer: &Buffer<f64>,
                   maxexp: &Buffer<f64>, mut global_size: usize,
                   max_work_size: usize, mut local_size: usize,
                   mut num_groups: usize) {
    let kernel_log_sum_gpu =
        pro_que.kernel_builder("log_sum_gpu_vec").global_work_size(global_size).local_work_size(local_size).arg(sum_buffer).arg_local::<f64>(local_size).arg(maxexp).build().expect("Kernel log_sum_gpu_vec build failed.");
    unsafe {
        kernel_log_sum_gpu.enq().expect("Error while executing log_sum_gpu_vec kernel.");
    }
    global_size = num_groups;
    local_size =
        if global_size < max_work_size { global_size } else { max_work_size };
    num_groups = (global_size as f32 / local_size as f32).ceil() as usize;
    while global_size > 1 {
        let kernel_sum_gpu_vec =
            pro_que.kernel_builder("sum_gpu_vec").global_work_size(global_size).local_work_size(local_size).arg(sum_buffer).arg_local::<f64>(local_size).build().expect("Kernel sum_gpu_vec build failed.");
        unsafe {
            kernel_sum_gpu_vec.enq().expect("Error while executing sum_gpu_vec kernel.");
        }
        global_size = num_groups;
        local_size =
            if global_size < max_work_size {
                global_size
            } else { max_work_size };
        num_groups = (global_size as f32 / local_size as f32).ceil() as usize;
    }
}
/// Iterates over the training data because there are less training points. There are two approaches
/// for iterating over the training data:
///
/// * The faster approach computes the log-likelihood for each training and test pair points in a
/// $`m \times n`$ matrix. Then, apply the
/// [LogSumExp trick](https://en.wikipedia.org/wiki/LogSumExp) on each row of the matrix. This is
/// implemented in [logpdf_iterate_train_high_memory](fn.logpdf_iterate_train_high_memory.html).
/// * However, a $`m \times n`$ matrix can be too much large for the OpenCL device. The alternative
/// is to iterate twice along the training data. In the first iteration, the maximum
/// log-likelihood for each test point is stored in a $`m`$ vector buffer. In the second pass along
/// the train data, the logpdf can be computed using the LogSumExp trick with the pre-computed
/// maximums. This is implemented in
/// [logpdf_iterate_train_low_memory](fn.logpdf_iterate_train_low_memory.html).
unsafe fn logpdf_iterate_train(kde: &mut Box<GaussianKDE>,
                               pro_que: &mut Box<ProQue>,
                               x: *const DoubleNumpyArray,
                               result: *mut c_double, error: *mut Error) {
    let m = *(*x).shape;
    let tmp_vec_buffer =
        Buffer::<f64>::builder().context(pro_que.context()).len(m *
                                                                    kde.n).build();
    match tmp_vec_buffer {
        Ok(b) =>
        logpdf_iterate_train_high_memory(kde, pro_que, x, result, &b, error),
        Err(_) => {
            let (tmp_vec_buffer,) =
                {
                    (match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                         {
                         Ok(b) => b,
                         Err(_) => { *error = Error::MemoryError; return; }
                     },)
                };
            logpdf_iterate_train_low_memory(kde, pro_que, x, result,
                                            &tmp_vec_buffer, error);
        }
    }
}
/// Iterates twice over the training data to compute the logpdf of each test point using a $`m`$
/// vector as described in [logpdf_iterate_train](fn.logpdf_iterate_train.html).
unsafe fn logpdf_iterate_train_low_memory(kde: &mut Box<GaussianKDE>,
                                          pro_que: &mut Box<ProQue>,
                                          x: *const DoubleNumpyArray,
                                          result: *mut c_double,
                                          tmp_vec_buffer: &Buffer<f64>,
                                          error: *mut Error) {
    let m = *(*x).shape;
    let d = kde.d;
    let n = kde.n;
    let test_slice = slice::from_raw_parts((*x).ptr, m * d);
    let (test_instances_buffer,) =
        {
            (match Buffer::builder().context(pro_que.context()).len(test_slice.len()).copy_host_slice(&test_slice).build()
                 {
                 Ok(b) => b,
                 Err(_) => { *error = Error::MemoryError; return; }
             },)
        };
    let (max_buffer, final_result_buffer, tmp_matrix_buffer) =
        {
            (match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                 {
                 Ok(b) => b,
                 Err(_) => { *error = Error::MemoryError; return; }
             },
             match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                 {
                 Ok(b) => b,
                 Err(_) => { *error = Error::MemoryError; return; }
             },
             match Buffer::<f64>::builder().context(pro_que.context()).len(m *
                                                                               d).build()
                 {
                 Ok(b) => b,
                 Err(_) => { *error = Error::MemoryError; return; }
             })
        };
    buffer_fill_value(&pro_que, &max_buffer, m, f64::MIN);
    buffer_fill_value(&pro_que, &final_result_buffer, m, 0.0f64);
    let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);
    let substract_name = kernel_substract_name(test_rowmajor, kde.rowmajor);
    let kernel_substract =
        pro_que.kernel_builder(substract_name).global_work_size(m *
                                                                    d).arg(&test_instances_buffer).arg(d
                                                                                                           as
                                                                                                           u32).arg(&kde.training_data).arg(&tmp_matrix_buffer).arg_named("row",
                                                                                                                                                                          &0u32).arg(test_leading_dimension
                                                                                                                                                                                         as
                                                                                                                                                                                         u32).arg(kde.leading_dimension
                                                                                                                                                                                                      as
                                                                                                                                                                                                      u32).build().expect("Kernel substract build failed.");
    let kernel_solve =
        pro_que.kernel_builder("solve").global_work_size(m).arg(&tmp_matrix_buffer).arg(&kde.chol_cov).arg(d
                                                                                                               as
                                                                                                               u32).build().expect("Kernel solve build failed.");
    let kernel_square =
        pro_que.kernel_builder("square").global_work_size(m *
                                                              d).arg(&tmp_matrix_buffer).build().expect("Kernel square build failed.");
    let kernel_sumout =
        pro_que.kernel_builder("logsumout").global_work_size(m).arg(&tmp_matrix_buffer).arg(tmp_vec_buffer).arg(d
                                                                                                                    as
                                                                                                                    u32).arg(kde.lognorm_factor).build().expect("Kernel logsumout build failed.");
    let kernel_sumout_checkmax =
        pro_que.kernel_builder("logsumout_checkmax").global_work_size(m).arg(&tmp_matrix_buffer).arg(tmp_vec_buffer).arg(&max_buffer).arg(d
                                                                                                                                              as
                                                                                                                                              u32).arg(kde.lognorm_factor).build().expect("Kernel logsumout_checkmax build failed.");
    let kernel_expsum =
        pro_que.kernel_builder("exp_and_sum").global_work_size(m).arg(tmp_vec_buffer).arg(&max_buffer).arg(&final_result_buffer).build().expect("Kernel exp_and_sum build failed.");
    let kernel_log_and_sum =
        pro_que.kernel_builder("log_and_sum").global_work_size(m).arg(&final_result_buffer).arg(&max_buffer).build().expect("Kernel log_and_sum build failed.");
    for i in 0..n {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract.enq().expect("Error while executing substract kernel.");
        kernel_solve.enq().expect("Error while executing solve kernel.");
        kernel_square.enq().expect("Error while executing square kernel.");
        kernel_sumout_checkmax.enq().expect("Error while executing logsumout_checkmax kernel.");
    }
    for i in 0..n {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract.enq().expect("Error while executing substract kernel.");
        kernel_solve.enq().expect("Error while executing solve kernel.");
        kernel_square.enq().expect("Error while executing square kernel.");
        kernel_sumout.enq().expect("Error while executing logsumout kernel.");
        kernel_expsum.enq().expect("Error while executing exp_and_sum kernel.");
    }
    kernel_log_and_sum.enq().expect("Error while executing log_and_sum kernel.");
    let final_result = slice::from_raw_parts_mut(result, m);
    final_result_buffer.cmd().queue(pro_que.queue()).read(final_result).enq().expect("Error reading result data.");
    *error = Error::NoError;
}
/// Iterates over the training data to compute the logpdf of each test point using a $`m \times n`$
/// matrix as described in [logpdf_iterate_train](fn.logpdf_iterate_train.html).
unsafe fn logpdf_iterate_train_high_memory(kde: &mut Box<GaussianKDE>,
                                           pro_que: &mut Box<ProQue>,
                                           x: *const DoubleNumpyArray,
                                           result: *mut c_double,
                                           tmp_vec_buffer: &Buffer<f64>,
                                           error: *mut Error) {
    let m = *(*x).shape;
    let d = kde.d;
    let final_result = slice::from_raw_parts_mut(result, m);
    let n = kde.n;
    pro_que.set_dims(m * d);
    let max_work_size = get_max_work_size(&pro_que);
    let local_work_size = if n < max_work_size { n } else { max_work_size };
    let num_groups = (n as f32 / local_work_size as f32).ceil() as usize;
    let test_slice = slice::from_raw_parts((*x).ptr, m * d);
    let (test_instances_buffer,) =
        {
            (match Buffer::builder().context(pro_que.context()).len(test_slice.len()).copy_host_slice(&test_slice).build()
                 {
                 Ok(b) => b,
                 Err(_) => { *error = Error::MemoryError; return; }
             },)
        };
    let (max_buffer, final_result_buffer, tmp_matrix_buffer) =
        {
            (match Buffer::<f64>::builder().context(pro_que.context()).len(m *
                                                                               num_groups).build()
                 {
                 Ok(b) => b,
                 Err(_) => { *error = Error::MemoryError; return; }
             },
             match Buffer::<f64>::builder().context(pro_que.context()).len(m).build()
                 {
                 Ok(b) => b,
                 Err(_) => { *error = Error::MemoryError; return; }
             },
             match Buffer::<f64>::builder().context(pro_que.context()).len(m *
                                                                               d).build()
                 {
                 Ok(b) => b,
                 Err(_) => { *error = Error::MemoryError; return; }
             })
        };
    let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);
    let substract_name = kernel_substract_name(test_rowmajor, kde.rowmajor);
    let kernel_substract =
        pro_que.kernel_builder(substract_name).global_work_size(m *
                                                                    d).arg(&test_instances_buffer).arg(d
                                                                                                           as
                                                                                                           u32).arg(&kde.training_data).arg(&tmp_matrix_buffer).arg_named("row",
                                                                                                                                                                          &0u32).arg(test_leading_dimension
                                                                                                                                                                                         as
                                                                                                                                                                                         u32).arg(kde.leading_dimension
                                                                                                                                                                                                      as
                                                                                                                                                                                                      u32).build().expect("Kernel substract build failed.");
    let kernel_solve =
        pro_que.kernel_builder("solve").global_work_size(m).arg(&tmp_matrix_buffer).arg(&kde.chol_cov).arg(d
                                                                                                               as
                                                                                                               u32).build().expect("Kernel solve build failed.");
    let kernel_square =
        pro_que.kernel_builder("square").global_work_size(m *
                                                              d).arg(&tmp_matrix_buffer).build().expect("Kernel square build failed.");
    let kernel_sumout =
        pro_que.kernel_builder("logsumout_to_matrix").global_work_size(m).arg(&tmp_matrix_buffer).arg(tmp_vec_buffer).arg(d
                                                                                                                              as
                                                                                                                              u32).arg_named("sol_row",
                                                                                                                                             &0u32).arg(n
                                                                                                                                                            as
                                                                                                                                                            u32).arg(kde.lognorm_factor).build().expect("Kernel logsumout_to_matrix build failed.");
    let kernel_exp_and_sum =
        pro_que.kernel_builder("exp_and_sum_mat").global_work_size(m *
                                                                       n).arg(tmp_vec_buffer).arg(&max_buffer).arg(n
                                                                                                                       as
                                                                                                                       u32).arg(num_groups
                                                                                                                                    as
                                                                                                                                    u32).build().expect("Kernel exp_and_sum_mat build failed.");
    let kernel_log_and_sum =
        pro_que.kernel_builder("log_and_sum_mat").global_work_size(m).arg(&final_result_buffer).arg(tmp_vec_buffer).arg(&max_buffer).arg(n
                                                                                                                                             as
                                                                                                                                             u32).arg(num_groups
                                                                                                                                                          as
                                                                                                                                                          u32).build().expect("Kernel log_and_sum_mat build failed.");
    for i in 0..n {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract.enq().expect("Error while executing substract kernel.");
        kernel_solve.enq().expect("Error while executing solve kernel.");
        kernel_square.enq().expect("Error while executing square kernel.");
        kernel_sumout.set_arg("sol_row", i as u32).unwrap();
        kernel_sumout.enq().expect("Error while executing logsumout_to_matrix kernel.");
    }
    max_gpu_mat(&pro_que, tmp_vec_buffer, &max_buffer, m, n, max_work_size,
                local_work_size, num_groups);
    kernel_exp_and_sum.enq().expect("Error while executing exp_and_sum_mat kernel.");
    sum_gpu_mat(&pro_que, tmp_vec_buffer, m, n, max_work_size,
                local_work_size, num_groups);
    kernel_log_and_sum.enq().expect("Error while executing kernel log_and_sum_mat kernel.");
    final_result_buffer.cmd().queue(pro_que.queue()).read(final_result).enq().expect("Error reading result data.");
    *error = Error::NoError;
}
/// Finds the maximum element of each row in the matrix buffer `max_buffer` and saves the result in
/// the first column of each row of the matrix buffer `result_buffer`. **This operation invalidates
/// the rest of the data in `result_buffer`**, but keeps constant `max_buffer`. That is:
/// If `max_buffer` is equal to:
///
/// ```math
///     \begin{bmatrix} a_{11} & \cdots & a_{1n}\\
///                     \vdots &\ddots  & \vdots\\
///                     a_{m1} & \cdots & a_{mn}
///     \end{bmatrix}
/// ```
///
/// After calling `max_gpu_mat`, `result_buffer` will be equal to:
///
/// ```math
///     \begin{bmatrix} \max_{i}a_{1i} &  \cdots & \text{unexpected values} & \ldots\\
///                      \vdots & \cdots & \text{unexpected values} & \ldots\\
///                     \max_{i}a_{mi} & \cdots & \text{unexpected values} & \ldots\\
/// \end{bmatrix}
/// ```
///
/// `n_rows` is the number of rows in `max_buffer`. `n_cols` is the number of columns of
/// `max_buffer`. `max_work_size` is the maximum number of work items in a work group for the
/// selected device. `local_size` is the actual number of work items in each work group.
/// `num_groups` is the actual number of work groups and the number of columns in `result_buffer`.
fn max_gpu_mat(pro_que: &ProQue, max_buffer: &Buffer<f64>,
               result_buffer: &Buffer<f64>, n_rows: usize, mut n_cols: usize,
               max_work_size: usize, mut local_size: usize,
               mut num_groups: usize) {
    let kernel_max_gpu =
        pro_que.kernel_builder("max_gpu_mat_copy").global_work_size((n_rows,
                                                                     n_cols)).local_work_size((1,
                                                                                               local_size)).arg(max_buffer).arg(result_buffer).arg_local::<f64>(local_size).arg(n_cols
                                                                                                                                                                                    as
                                                                                                                                                                                    u32).build().expect("Kernel max_gpu_mat_copy build failed.");
    unsafe {
        kernel_max_gpu.enq().expect("Error while executing max_gpu_mat_copy kernel.");
    }
    n_cols = num_groups;
    local_size = if n_cols < max_work_size { n_cols } else { max_work_size };
    let matrix_actual_cols = num_groups;
    num_groups = (n_cols as f32 / local_size as f32).ceil() as usize;
    while n_cols > 1 {
        let kernel_max_gpu =
            pro_que.kernel_builder("max_gpu_mat").global_work_size((n_rows,
                                                                    n_cols)).local_work_size((1,
                                                                                              local_size)).arg(result_buffer).arg_local::<f64>(local_size).arg(matrix_actual_cols
                                                                                                                                                                   as
                                                                                                                                                                   u32).build().expect("Kernel max_gpu_mat build failed.");
        unsafe {
            kernel_max_gpu.enq().expect("Error while executing max_gpu_mat kernel.");
        }
        n_cols = num_groups;
        local_size =
            if n_cols < max_work_size { n_cols } else { max_work_size };
        num_groups = (n_cols as f32 / local_size as f32).ceil() as usize;
    }
}
/// Sums all the elements of each row in the matrix buffer `sum_buffer` and saves the result in
/// the first column of each row (i.e. `max_buffer[i][0]`). **This operation invalidates
/// the rest of the data in `sum_buffer`**. That is:
/// If `sum_buffer` is equal to:
///
/// ```math
///     \begin{bmatrix} a_{11} & \cdots & a_{1n}\\
///                     \vdots &\ddots  & \vdots\\
///                     a_{m1} & \cdots & a_{mn}
///     \end{bmatrix}
/// ```
///
/// After calling `sum_gpu_mat`, `sum_buffer` will be equal to:
///
/// ```math
///     \begin{bmatrix} \sum_{i}^{n}a_{1i} &  \cdots & \text{unexpected values} & \ldots\\
///                      \vdots & \cdots & \text{unexpected values} & \ldots\\
///                     \sum_{i}^{n}a_{mi} & \cdots & \text{unexpected values} & \ldots\\
/// \end{bmatrix}
/// ```
///
/// `n_rows` is the number of rows in `max_buffer`. `n_cols` is the number of columns of
/// `max_buffer`. `max_work_size` is the maximum number of work items in a work group for the
/// selected device. `local_size` is the actual number of work items in each work group.
/// `num_groups` is the actual number of work groups and the number of columns in `result_buffer`.
fn sum_gpu_mat(pro_que: &ProQue, sum_buffer: &Buffer<f64>, n_rows: usize,
               mut n_cols: usize, max_work_size: usize, mut local_size: usize,
               mut num_groups: usize) {
    let n_cols_orig = n_cols as u32;
    while n_cols > 1 {
        let kernel_sum_gpu_mat =
            pro_que.kernel_builder("sum_gpu_mat").global_work_size((n_rows,
                                                                    n_cols)).local_work_size((1,
                                                                                              local_size)).arg(sum_buffer).arg_local::<f64>(local_size).arg(n_cols_orig).build().expect("Kernel sum_gpu_mat build failed.");
        unsafe {
            kernel_sum_gpu_mat.enq().expect("Error while executing sum_gpu_mat kernel.");
        }
        n_cols = num_groups;
        local_size =
            if n_cols < max_work_size { n_cols } else { max_work_size };
        num_groups = (n_cols as f32 / local_size as f32).ceil() as usize;
    }
}
