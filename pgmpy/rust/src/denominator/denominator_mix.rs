use crate::{empty_buffers, copy_buffers, Error, DoubleNumpyArray, buffer_fill_value,
            get_max_work_size, is_rowmajor, max_gpu_vec_copy, log_sum_gpu_vec, max_gpu_mat,
            sum_gpu_mat};

use crate::denominator::{CKDE, s1_s3_coefficients, s2};

use std::slice;
use std::f64;

use ocl::{Buffer, ProQue};
use libc::{c_double};

fn substract_without_origin_from_indices_iterate_test_name(train_rowmajor: bool, test_rowmajor: bool) -> &'static str {
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

fn substract_without_origin_from_indices_iterate_train_name(train_rowmajor: bool, test_rowmajor: bool) -> &'static str {
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

fn exponent_coefficients_iterate_test_name(train_rowmajor: bool) -> &'static str {
    if train_rowmajor {
        "exponent_coefficients_iterate_test_rowmajor"
    } else {
        "exponent_coefficients_iterate_test_columnmajor"
    }
}

fn exponent_coefficients_iterate_train_high_memory_name(train_rowmajor: bool) -> &'static str {
    if train_rowmajor {
        "exponent_coefficients_iterate_train_high_memory_rowmajor"
    } else {
        "exponent_coefficients_iterate_train_high_memory_columnmajor"
    }
}

fn exponent_coefficients_iterate_train_low_memory_checkmax_name(train_rowmajor: bool) -> &'static str {
    if train_rowmajor {
        "exponent_coefficients_iterate_train_low_memory_checkmax_rowmajor"
    } else {
        "exponent_coefficients_iterate_train_low_memory_checkmax_columnmajor"
    }
}

fn exponent_coefficients_iterate_train_low_memory_compute_name(train_rowmajor: bool) -> &'static str {
    if train_rowmajor {
        "exponent_coefficients_iterate_train_low_memory_compute_rowmajor"
    } else {
        "exponent_coefficients_iterate_train_low_memory_compute_columnmajor"
    }
}

#[no_mangle]
pub unsafe extern "C" fn logdenominator_dataset(
    ckde: *mut CKDE,
    pro_que: *mut ProQue,
    x: *const DoubleNumpyArray,
    result: *mut c_double,
    error: *mut Error,
) {
    let mut ckde = Box::from_raw(ckde);
    let mut pro_que = Box::from_raw(pro_que);
    let m = *(*x).shape;

    *error = Error::NoError;
    if (*ckde.kde).n >= m {
        logdenominator_iterate_test(&mut ckde, &mut pro_que, x, result, error);
    } else {
        logdenominator_iterate_train(&mut ckde, &mut pro_que, x, result, error);
    }

    Box::into_raw(ckde);
    Box::into_raw(pro_que);
}

unsafe fn logdenominator_iterate_test(
    ckde: &Box<CKDE>,
    pro_que: &mut Box<ProQue>,
    x: *const DoubleNumpyArray,
    result: *mut c_double,
    error: *mut Error,
) {
    let kde = Box::from_raw(ckde.kde);
    let test_shape = slice::from_raw_parts((*x).shape, 2);

    let m = test_shape[0];
    let d_test = test_shape[1];
    let nparents_kde = kde.d - 1;
    let n = kde.n;

    let max_work_size = get_max_work_size(&pro_que);
    let local_work_size = if n < max_work_size { n } else { max_work_size };
    let num_groups = (n as f32 / local_work_size as f32).ceil() as usize;

    let test_slice = slice::from_raw_parts((*x).ptr, m * d_test);

    let (test_instances_buffer,) = copy_buffers!(pro_que, error, test_slice);

    let (s1, s3, final_result_buffer, mahalanobis_buffer, dotproduct_buffer, ti_buffer,
        max_coefficients) =
        empty_buffers!(pro_que, error, f64, m, m, m, n, n, n*nparents_kde, num_groups);

    let a = 0.5 * (ckde.precision_variable + s2(ckde));

    let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);

    s1_s3_coefficients(
        ckde,
        pro_que,
        &test_instances_buffer,
        test_leading_dimension as u32,
        test_rowmajor,
        &s1,
        &s3,
        m,
    );

    let kernel_substract = pro_que
        .kernel_builder(substract_without_origin_from_indices_iterate_test_name(kde.rowmajor, test_rowmajor))
        .global_work_size(n * nparents_kde)
        .arg(&kde.training_data)
        .arg(kde.leading_dimension as u32)
        .arg(&test_instances_buffer)
        .arg(test_leading_dimension as u32)
        .arg(&ti_buffer)
        .arg_named("row", &0u32)
        .arg(nparents_kde as u32)
        .arg(ckde.kde_indices.as_ref().unwrap())
        .build()
        .expect("Kernel substract_without_origin_from_indices build failed.");

    let kernel_mahalanobis = pro_que
        .kernel_builder("mahalanobis")
        .global_work_size(n * nparents_kde * nparents_kde)
        .local_work_size(nparents_kde * nparents_kde)
        .arg(&ti_buffer)
        .arg(&ckde.precision)
        .arg(&mahalanobis_buffer)
        .arg_local::<f64>(nparents_kde*nparents_kde)
        .arg(nparents_kde as u32)
        .build()
        .expect("Kernel mahalanobis build failed.");

    let kernel_dotproduct = pro_que
        .kernel_builder("dotproduct")
        .global_work_size(n * nparents_kde)
        .local_work_size(nparents_kde)
        .arg(&ti_buffer)
        .arg(&ckde.precision)
        .arg(&dotproduct_buffer)
        .arg_local::<f64>(nparents_kde)
        .arg(nparents_kde as u32)
        .build()
        .expect("Kernel dotproduct build failed.");

    let kernel_coefficients = pro_que
        .kernel_builder(exponent_coefficients_iterate_test_name(kde.rowmajor))
        .global_work_size(n)
        .arg(&kde.training_data)
        .arg(kde.leading_dimension as u32)
        .arg(&ckde.precision)
        .arg(&mahalanobis_buffer)
        .arg(&dotproduct_buffer)
        .arg(&s1)
        .arg((4.0f64*a).recip())
        .arg(&s3)
        .arg_named("offset", &0u32)
        .build()
        .expect("Kernel exponent_coefficients_iterate_test build failed.");

    let kernel_log_sum_gpu = pro_que
        .kernel_builder("copy_logpdf_result")
        .global_work_size(1)
        .arg(&mahalanobis_buffer)
        .arg(&max_coefficients)
        .arg(&final_result_buffer)
        .arg_named("offset", &0u32)
        .build()
        .expect("Kernel copy_logpdf_result build failed.");

    for i in 0..m {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract
            .enq()
            .expect("Error while executing substract_without_origin_from_indices kernel.");

        kernel_mahalanobis
            .enq()
            .expect("Error while executing mahalanobis kernel.");

        kernel_dotproduct
            .enq()
            .expect("Error while executing dotproduct kernel.");

        kernel_coefficients.set_arg("offset", i as u32).unwrap();
        kernel_coefficients
            .enq()
            .expect("Error while executing exponent_coefficients_iterate_test kernel.");

        max_gpu_vec_copy(
            pro_que,
            &mahalanobis_buffer,
            &max_coefficients,
            n,
            max_work_size,
            local_work_size,
            num_groups,
        );

        log_sum_gpu_vec(
            &pro_que,
            &mahalanobis_buffer,
            &max_coefficients,
            n,
            max_work_size,
            local_work_size,
            num_groups,
        );

        kernel_log_sum_gpu.set_arg("offset", i as u32).unwrap();
        kernel_log_sum_gpu
            .enq()
            .expect("Error while executing copy_logpdf_result kernel.");
    }

    let kernel_sum_constant = pro_que
        .kernel_builder("sum_constant")
        .global_work_size(m)
        .arg(&final_result_buffer)
        .arg(ckde.lognorm_factor)
        .build()
        .expect("Kernel sum_constant build failed.");

    kernel_sum_constant.enq().expect("Error while executing sum_constant kernel.");

    let final_result = slice::from_raw_parts_mut(result, m);

    final_result_buffer
        .cmd()
        .queue(pro_que.queue())
        .read(final_result)
        .enq()
        .expect("Error reading result data.");

    Box::into_raw(kde);

}

unsafe fn logdenominator_iterate_train(
    ckde: &Box<CKDE>,
    pro_que: &mut Box<ProQue>,
    x: *const DoubleNumpyArray,
    result: *mut c_double,
    error: *mut Error,
) {
    let m = *(*x).shape;
    let tmp_vec_buffer = Buffer::<f64>::builder()
        .context(pro_que.context())
        .len(m * (*ckde.kde).n)
        .build();

    match tmp_vec_buffer {
        Ok(b) => {
            logdenominator_iterate_train_high_memory(ckde, pro_que, x, result, &b, error)
        }
        Err(_) => {
            // TODO: If n < 2m, is it better to iterate over the training data?
            logdenominator_iterate_train_low_memory(
                ckde,
                pro_que,
                x,
                result,
                error,
            );
        }
    }
}

unsafe fn logdenominator_iterate_train_low_memory(
    ckde: &Box<CKDE>,
    pro_que: &mut Box<ProQue>,
    x: *const DoubleNumpyArray,
    result: *mut c_double,
    error: *mut Error,
) {
    let kde = Box::from_raw(ckde.kde);
    let test_shape = slice::from_raw_parts((*x).shape, 2);

    let m = test_shape[0];
    let d_test = test_shape[1];
    let nparents_kde = kde.d - 1;
    let n = kde.n;

    let test_slice = slice::from_raw_parts((*x).ptr, m * d_test);

    let (test_instances_buffer,) = copy_buffers!(pro_que, error, test_slice);

    let (s1, s3, final_result_buffer, ti_buffer, mahalanobis_buffer, dotproduct_buffer, max_buffer) =
        empty_buffers!(pro_que, error, f64, m, m, m, m*nparents_kde, m, m, m);

    buffer_fill_value(&pro_que, &max_buffer, m, f64::MIN);
    buffer_fill_value(&pro_que, &final_result_buffer, m, 0.0f64);

    let a = 0.5 * (ckde.precision_variable + s2(ckde));

    let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);

    s1_s3_coefficients(
        ckde,
        pro_que,
        &test_instances_buffer,
        test_leading_dimension as u32,
        test_rowmajor,
        &s1,
        &s3,
        m,
    );

    let kernel_substract = pro_que
        .kernel_builder(substract_without_origin_from_indices_iterate_train_name(kde.rowmajor, test_rowmajor))
        .global_work_size(m * nparents_kde)
        .arg(&kde.training_data)
        .arg(kde.leading_dimension as u32)
        .arg(&test_instances_buffer)
        .arg(test_leading_dimension as u32)
        .arg(&ti_buffer)
        .arg_named("row", &0u32)
        .arg(nparents_kde as u32)
        .arg(ckde.kde_indices.as_ref().unwrap())
        .build()
        .expect("Kernel substract_without_origin_from_indices build failed.");

    let kernel_mahalanobis = pro_que
        .kernel_builder("mahalanobis")
        .global_work_size(m * nparents_kde * nparents_kde)
        .local_work_size(nparents_kde * nparents_kde)
        .arg(&ti_buffer)
        .arg(&ckde.precision)
        .arg(&mahalanobis_buffer)
        .arg_local::<f64>(nparents_kde*nparents_kde)
        .arg(nparents_kde as u32)
        .build()
        .expect("Kernel mahalanobis build failed.");

    let kernel_dotproduct = pro_que
        .kernel_builder("dotproduct")
        .global_work_size(m * nparents_kde)
        .local_work_size(nparents_kde)
        .arg(&ti_buffer)
        .arg(&ckde.precision)
        .arg(&dotproduct_buffer)
        .arg_local::<f64>(nparents_kde)
        .arg(nparents_kde as u32)
        .build()
        .expect("Kernel dotproduct build failed.");

    let kernel_coefficients_checkmax = pro_que
        .kernel_builder(exponent_coefficients_iterate_train_low_memory_checkmax_name(kde.rowmajor))
        .global_work_size(m)
        .arg(&kde.training_data)
        .arg(kde.leading_dimension as u32)
        .arg(&ckde.precision)
        .arg(&mahalanobis_buffer)
        .arg(&max_buffer)
        .arg(&dotproduct_buffer)
        .arg(&s1)
        .arg((4.0*a).recip())
        .arg(&s3)
        .arg_named("offset", &0u32)
        .build()
        .expect("Kernel exponent_coefficients_iterate_train_low_memory_checkmax build failed.");

    let kernel_coefficients_compute = pro_que
        .kernel_builder(exponent_coefficients_iterate_train_low_memory_compute_name(kde.rowmajor))
        .global_work_size(m)
        .arg(&kde.training_data)
        .arg(kde.leading_dimension as u32)
        .arg(&ckde.precision)
        .arg(&mahalanobis_buffer)
        .arg(&final_result_buffer)
        .arg(&max_buffer)
        .arg(&dotproduct_buffer)
        .arg(&s1)
        .arg((4.0*a).recip())
        .arg(&s3)
        .arg_named("offset", &0u32)
        .build()
        .expect("Kernel exponent_coefficients_iterate_train_low_memory_compute build failed.");

    let kernel_log_and_sum = pro_que
        .kernel_builder("log_and_sum")
        .global_work_size(m)
        .arg(&final_result_buffer)
        .arg(&max_buffer)
        .build()
        .expect("Kernel log_and_sum build failed.");

    for i in 0..n {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract
            .enq()
            .expect("Error while executing substract_without_origin_from_indices kernel.");

        kernel_mahalanobis
            .enq()
            .expect("Error while executing mahalanobis kernel.");

        kernel_dotproduct
            .enq()
            .expect("Error while executing dotproduct kernel.");

        kernel_coefficients_checkmax.set_arg("offset", i as u32).unwrap();
        kernel_coefficients_checkmax
            .enq()
            .expect("Error while executing exponent_coefficients_iterate_train_low_memory_checkmax kernel.");
    }

    for i in 0..n {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract
            .enq()
            .expect("Error while executing substract_without_origin_from_indices kernel.");

        kernel_mahalanobis
            .enq()
            .expect("Error while executing mahalanobis kernel.");

        kernel_dotproduct
            .enq()
            .expect("Error while executing dotproduct kernel.");

        kernel_coefficients_compute.set_arg("offset", i as u32).unwrap();
        kernel_coefficients_compute
            .enq()
            .expect("Error while executing exponent_coefficients_iterate_train_low_memory_compute kernel.");
    }

    kernel_log_and_sum
        .enq()
        .expect("Error while executing kernel_log_and_sum kernel.");

    let kernel_sum_constant = pro_que
        .kernel_builder("sum_constant")
        .global_work_size(m)
        .arg(&final_result_buffer)
        .arg(ckde.lognorm_factor)
        .build()
        .expect("Kernel sum_constant build failed.");

    kernel_sum_constant.enq().expect("Error while executing sum_constant kernel.");

    let final_result = slice::from_raw_parts_mut(result, m);
    final_result_buffer
        .cmd()
        .queue(pro_que.queue())
        .read(final_result)
        .enq()
        .expect("Error reading result data.");

    Box::into_raw(kde);
}

unsafe fn logdenominator_iterate_train_high_memory(
    ckde: &Box<CKDE>,
    pro_que: &mut Box<ProQue>,
    x: *const DoubleNumpyArray,
    result: *mut c_double,
    coeffs: &Buffer<f64>,
    error: *mut Error,
) {
    let kde = Box::from_raw(ckde.kde);
    let test_shape = slice::from_raw_parts((*x).shape, 2);

    let m = test_shape[0];
    let d_test = test_shape[1];
    let nparents_kde = kde.d - 1;
    let n = kde.n;

    let max_work_size = get_max_work_size(&pro_que);
    let local_work_size = if n < max_work_size { n } else { max_work_size };
    let num_groups = (n as f32 / local_work_size as f32).ceil() as usize;

    let test_slice = slice::from_raw_parts((*x).ptr, m * d_test);

    let (test_instances_buffer,) = copy_buffers!(pro_que, error, test_slice);

    let (s1, s3, final_result_buffer, ti_buffer, dotproduct_buffer, max_buffer) =
        empty_buffers!(pro_que, error, f64, m, m, m, m*nparents_kde, m, m * num_groups);

    let a = 0.5 * (ckde.precision_variable + s2(ckde));

    let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);

    s1_s3_coefficients(
        ckde,
        pro_que,
        &test_instances_buffer,
        test_leading_dimension as u32,
        test_rowmajor,
        &s1,
        &s3,
        m,
    );

    let kernel_substract = pro_que
        .kernel_builder(substract_without_origin_from_indices_iterate_train_name(kde.rowmajor, test_rowmajor))
        .global_work_size(m * nparents_kde)
        .arg(&kde.training_data)
        .arg(kde.leading_dimension as u32)
        .arg(&test_instances_buffer)
        .arg(test_leading_dimension as u32)
        .arg(&ti_buffer)
        .arg_named("row", &0u32)
        .arg(nparents_kde as u32)
        .arg(ckde.kde_indices.as_ref().unwrap())
        .build()
        .expect("Kernel substract_without_origin_from_indices build failed.");

    let kernel_mahalanobis = pro_que
        .kernel_builder("mahalanobis_mat")
        .global_work_size(m * nparents_kde * nparents_kde)
        .local_work_size(nparents_kde * nparents_kde)
        .arg(&ti_buffer)
        .arg(&ckde.precision)
        .arg(coeffs)
        .arg_local::<f64>(nparents_kde*nparents_kde)
        .arg(nparents_kde as u32)
        .arg_named("offset", &0u32)
        .arg(n as u32)
        .build()
        .expect("Kernel mahalanobis_mat build failed.");

    let kernel_dotproduct = pro_que
        .kernel_builder("dotproduct")
        .global_work_size(m * nparents_kde)
        .local_work_size(nparents_kde)
        .arg(&ti_buffer)
        .arg(&ckde.precision)
        .arg(&dotproduct_buffer)
        .arg_local::<f64>(nparents_kde)
        .arg(nparents_kde as u32)
        .build()
        .expect("Kernel dotproduct build failed.");

    let kernel_coefficients = pro_que
        .kernel_builder(exponent_coefficients_iterate_train_high_memory_name(kde.rowmajor))
        .global_work_size(m)
        .arg(&kde.training_data)
        .arg(kde.leading_dimension as u32)
        .arg(&ckde.precision)
        .arg(coeffs)
        .arg(&dotproduct_buffer)
        .arg(&s1)
        .arg((4.0*a).recip())
        .arg(&s3)
        .arg_named("offset", &0u32)
        .arg(n as u32)
        .build()
        .expect("Kernel exponent_coefficients_iterate_train_high_memory build failed.");

    let kernel_exp_and_sum_mat = pro_que
        .kernel_builder("exp_and_sum_mat")
        .global_work_size(m * n)
        .arg(coeffs)
        .arg(&max_buffer)
        .arg(n as u32)
        .arg(num_groups as u32)
        .build()
        .expect("Kernel exp_and_sum_mat build failed.");

    let kernel_log_and_sum = pro_que
        .kernel_builder("log_and_sum_mat")
        .global_work_size(m)
        .arg(&final_result_buffer)
        .arg(coeffs)
        .arg(&max_buffer)
        .arg(n as u32)
        .arg(num_groups as u32)
        .build()
        .expect("Kernel log_and_sum_mat build failed.");

    for i in 0..n {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract
            .enq()
            .expect("Error while executing substract_without_origin_from_indices kernel.");


        kernel_mahalanobis.set_arg("offset", i as u32).unwrap();
        kernel_mahalanobis
            .enq()
            .expect("Error while executing mahalanobis_mat kernel.");

        kernel_dotproduct
            .enq()
            .expect("Error while executing dotproduct kernel.");

        kernel_coefficients.set_arg("offset", i as u32).unwrap();
        kernel_coefficients
            .enq()
            .expect("Error while executing exponent_coefficients_iterate_train_high_memory kernel.");

    }

    max_gpu_mat(
        &pro_que,
        coeffs,
        &max_buffer,
        m,
        n,
        max_work_size,
        local_work_size,
        num_groups,
    );

    kernel_exp_and_sum_mat
        .enq()
        .expect("Error while executing exp_and_sum_mat kernel.");

    sum_gpu_mat(
        &pro_que,
        coeffs,
        m,
        n,
        max_work_size,
        local_work_size,
        num_groups,
    );

    kernel_log_and_sum
        .enq()
        .expect("Error while executing kernel_log_and_sum kernel.");

    let kernel_sum_constant = pro_que
        .kernel_builder("sum_constant")
        .global_work_size(m)
        .arg(&final_result_buffer)
        .arg(ckde.lognorm_factor)
        .build()
        .expect("Kernel sum_constant build failed.");

    kernel_sum_constant.enq().expect("Error while executing sum_constant kernel.");

    let final_result = slice::from_raw_parts_mut(result, m);
    final_result_buffer
        .cmd()
        .queue(pro_que.queue())
        .read(final_result)
        .enq()
        .expect("Error reading result data.");

    Box::into_raw(kde);
}