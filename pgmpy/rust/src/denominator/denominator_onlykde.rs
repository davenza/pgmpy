use crate::{empty_buffers, copy_buffers, Error, DoubleNumpyArray, buffer_fill_value,
            get_max_work_size, is_rowmajor, max_gpu_vec_copy, max_gpu_mat,
            sum_gpu_vec, sum_gpu_mat, create_reduction_buffers_gpu_vec,
            create_reduction_buffers_gpu_mat};

use crate::denominator::{CKDE};

use std::f64;
use std::slice;

use ocl::{Buffer, ProQue};
use libc::{c_double};

#[no_mangle]
pub unsafe extern "C" fn logdenominator_dataset_onlykde(
    ckde: *mut CKDE,
    pro_que: *mut ProQue,
    x: *const DoubleNumpyArray,
    result: *mut c_double,
    error: *mut Error,
) {
    *error = Error::NotFinished;
    let mut ckde = Box::from_raw(ckde);
    let mut pro_que = Box::from_raw(pro_que);
    let m = *(*x).shape;

    *error = Error::NoError;
    if (*ckde.kde).n >= m {
        logdenominator_iterate_test_onlykde(
            &mut ckde,
            &mut pro_que,
            x,
            result,
            error,
        );
    } else {
        logdenominator_iterate_train_onlykde(
            &mut ckde,
            &mut pro_que,
            x,
            result,
            error,
        );
    }

    Box::into_raw(ckde);
    Box::into_raw(pro_que);
}

fn substract_without_origin_name(train_rowmajor: bool, test_rowmajor: bool) -> &'static str {
    if train_rowmajor {
        if test_rowmajor {
            "substract_without_origin_rowmajor_rowmajor"
        } else {
            "substract_without_origin_rowmajor_columnmajor"
        }
    } else {
        if test_rowmajor {
            "substract_without_origin_columnmajor_rowmajor"
        } else {
            "substract_without_origin_columnmajor_columnmajor"
        }
    }
}

unsafe fn logdenominator_iterate_test_onlykde(
    ckde: &mut Box<CKDE>,
    pro_que: &mut Box<ProQue>,
    x: *const DoubleNumpyArray,
    result: *mut c_double,
    error: *mut Error,
) {
    let kde = Box::from_raw(ckde.kde);

    let m = *(*x).shape;
    let d = kde.d;
    let nparents_kde = d - 1;
    let n = kde.n;

    let max_work_size = get_max_work_size(&pro_que);
    let local_work_size = if n < max_work_size { n } else { max_work_size };
    let num_groups = (n as f32 / local_work_size as f32).ceil() as usize;

    let test_slice = slice::from_raw_parts((*x).ptr, m * d);

    let (test_instances_buffer,) =
        copy_buffers!(pro_que, error, test_slice);

    let (ti_buffer, final_result_buffer, coeffs, max_coefficients) =
        empty_buffers!(pro_que, error, f64, n * nparents_kde, m, n, 1);

    let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);

    let kernel_substract = pro_que
        .kernel_builder(substract_without_origin_name(kde.rowmajor, test_rowmajor))
        .global_work_size(n * nparents_kde)
        .arg(&kde.training_data)
        .arg(kde.leading_dimension as u32)
        .arg(&test_instances_buffer)
        .arg(test_leading_dimension as u32)
        .arg(&ti_buffer)
        .arg_named("row", &0u32)
        .arg(nparents_kde as u32)
        .build()
        .expect("Kernel substract_without_origin build failed.");

    let kernel_exponent_coefficients = pro_que
        .kernel_builder("onlykde_exponent_coefficients_iterate_test")
        .global_work_size(n * nparents_kde * nparents_kde)
        .local_work_size(nparents_kde * nparents_kde)
        .arg(&ti_buffer)
        .arg(nparents_kde as u32)
        .arg(&ckde.marginal_precision)
        .arg(&coeffs)
        .arg_local::<f64>(nparents_kde * nparents_kde)
        .build()
        .expect("Kernel onlykde_exponent_coefficients_iterate_test build failed.");

    let kernel_logsumexp_coeffs = pro_que
        .kernel_builder("logsumexp_coeffs")
        .global_work_size(n)
        .arg(&coeffs)
        .arg(&max_coefficients)
        .build()
        .expect("Kernel logsumexp_coeffs build failed.");

    let kernel_copy_logpdf = pro_que
        .kernel_builder("copy_logpdf_result")
        .global_work_size(1)
        .arg(&coeffs)
        .arg(&max_coefficients)
        .arg(&final_result_buffer)
        .arg_named("offset", &0u32)
        .build()
        .expect("Kernel copy_logpdf_result build failed.");

    let tmp_reduc_buffers = create_reduction_buffers_gpu_vec(pro_que, error, n, max_work_size);

    if *error == Error::MemoryError { return; };

    for i in 0..m {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract
            .enq()
            .expect("Error while executing substract_without_origin kernel.");

        kernel_exponent_coefficients
            .enq()
            .expect("Error while executing onlykde_exponent_coefficients_iterate_test kernel.");

        max_gpu_vec_copy(
            pro_que,
            &coeffs,
            &tmp_reduc_buffers,
            &max_coefficients,
            max_work_size,
            local_work_size,
            num_groups,
        );

        kernel_logsumexp_coeffs
            .enq()
            .expect("Error while executing logsumexp_coeffs kernel.");

        sum_gpu_vec(&pro_que, &coeffs, &tmp_reduc_buffers, max_work_size, local_work_size, num_groups);

        kernel_copy_logpdf.set_arg("offset", i as u32).unwrap();
        kernel_copy_logpdf
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
    *error = Error::NoError;
}

unsafe fn logdenominator_iterate_train_onlykde(
    ckde: &mut Box<CKDE>,
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
        Ok(b) => logdenominator_iterate_train_high_memory_onlykde(
            ckde, pro_que, x, result, &b, error,
        ),
        Err(_) => {
            // TODO: If n < 2m, is it better to iterate over the training data?
            logdenominator_iterate_train_low_memory_onlykde(
                ckde, pro_que, x, result, error
            );
        }
    }
}

unsafe fn logdenominator_iterate_train_low_memory_onlykde(
    ckde: &mut Box<CKDE>,
    pro_que: &mut Box<ProQue>,
    x: *const DoubleNumpyArray,
    result: *mut c_double,
    error: *mut Error,
) {
    let kde = Box::from_raw(ckde.kde);

    let m = *(*x).shape;
    let d = kde.d;
    let nparents_kde = d - 1;
    let n = kde.n;

    let test_slice = slice::from_raw_parts((*x).ptr, m * d);

    let (test_instances_buffer,) =
        copy_buffers!(pro_que, error, test_slice);

    let (ti_buffer, final_result_buffer, max_buffer) = empty_buffers!(
        pro_que,
        error,
        f64,
        m*nparents_kde,
        m,
        m
    );

    buffer_fill_value(&pro_que, &max_buffer, m, f64::MIN);
    buffer_fill_value(&pro_que, &final_result_buffer, m, 0.0f64);

    let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);

    let kernel_substract = pro_que
        .kernel_builder(substract_without_origin_name(kde.rowmajor, test_rowmajor))
        .global_work_size(m * nparents_kde)
        .arg(&test_instances_buffer)
        .arg(test_leading_dimension as u32)
        .arg(&kde.training_data)
        .arg(kde.leading_dimension as u32)
        .arg(&ti_buffer)
        .arg_named("row", &0u32)
        .arg(nparents_kde as u32)
        .build()
        .expect("Kernel substract_without_origin build failed.");

    let kernel_coefficients_checkmax = pro_que
        .kernel_builder("onlykde_exponent_coefficients_iterate_train_low_memory_checkmax")
        .global_work_size(m * nparents_kde * nparents_kde)
        .local_work_size(nparents_kde * nparents_kde)
        .arg(&ti_buffer)
        .arg(nparents_kde as u32)
        .arg(&ckde.marginal_precision)
        .arg(&max_buffer)
        .arg_local::<f64>(nparents_kde * nparents_kde)
        .build()
        .expect("Kernel onlykde_exponent_coefficients_iterate_train_low_memory_checkmax build failed.");

    let kernel_coefficients_compute = pro_que
        .kernel_builder("onlykde_exponent_coefficients_iterate_train_low_memory_compute")
        .global_work_size(m * nparents_kde * nparents_kde)
        .local_work_size(nparents_kde * nparents_kde)
        .arg(&ti_buffer)
        .arg(nparents_kde as u32)
        .arg(&ckde.marginal_precision)
        .arg(&final_result_buffer)
        .arg(&max_buffer)
        .arg_local::<f64>(nparents_kde * nparents_kde)
        .build()
        .expect("Kernel onlykde_exponent_coefficients_iterate_train_low_memory_compute build failed.");

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
            .expect("Error while executing substract_without_origin kernel.");

        kernel_coefficients_checkmax
            .enq()
            .expect("Error while executing onlykde_exponent_coefficients_iterate_train_low_memory_checkmax kernel.");
    }

    for i in 0..n {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract
            .enq()
            .expect("Error while executing substract_without_origin kernel.");

        kernel_coefficients_compute
            .enq()
            .expect("Error while executing onlykde_exponent_coefficients_iterate_train_low_memory_compute kernel.");
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
    *error = Error::NoError;
}

unsafe fn logdenominator_iterate_train_high_memory_onlykde(
    ckde: &mut Box<CKDE>,
    pro_que: &mut Box<ProQue>,
    x: *const DoubleNumpyArray,
    result: *mut c_double,
    coeffs: &Buffer<f64>,
    error: *mut Error,
) {
    let kde = Box::from_raw(ckde.kde);

    let m = *(*x).shape;
    let d = kde.d;
    let nparents_kde = d - 1;
    let n = kde.n;

    let max_work_size = get_max_work_size(&pro_que);
    let local_work_size = if n < max_work_size { n } else { max_work_size };
    let num_groups = (n as f32 / local_work_size as f32).ceil() as usize;

    let test_slice = slice::from_raw_parts((*x).ptr, m * d);

    let (test_instances_buffer,) =
        copy_buffers!(pro_que, error, test_slice);

    let (ti_buffer, final_result_buffer, max_buffer) = empty_buffers!(
        pro_que,
        error,
        f64,
        m*nparents_kde,
        m,
        m
    );

    let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);

    let tmp_reduc_buffers = create_reduction_buffers_gpu_mat(pro_que, error, m, n, max_work_size);

    if *error == Error::MemoryError { return; };

    let kernel_substract = pro_que
        .kernel_builder(substract_without_origin_name(test_rowmajor, kde.rowmajor))
        .global_work_size(m * nparents_kde)
        .arg(&test_instances_buffer)
        .arg(test_leading_dimension as u32)
        .arg(&kde.training_data)
        .arg(kde.leading_dimension as u32)
        .arg(&ti_buffer)
        .arg_named("row", &0u32)
        .arg(nparents_kde as u32)
        .build()
        .expect("Kernel substract_without_origin build failed.");

    let kernel_exponent_coefficients = pro_que
        .kernel_builder("onlykde_exponent_coefficients_iterate_train_high_memory")
        .global_work_size(m * nparents_kde * nparents_kde)
        .local_work_size(nparents_kde * nparents_kde)
        .arg(&ti_buffer)
        .arg(nparents_kde as u32)
        .arg(&ckde.marginal_precision)
        .arg(coeffs)
        .arg_local::<f64>(nparents_kde * nparents_kde)
        .arg_named("offset", &0u32)
        .arg(n as u32)
        .build()
        .expect("Kernel onlykde_exponent_coefficients_iterate_test build failed.");

    let kernel_expmax_mat = pro_que
        .kernel_builder("expmax_mat")
        .global_work_size(m * n)
        .arg(coeffs)
        .arg(&max_buffer)
        .arg(n as u32)
        .build()
        .expect("Kernel expmax_mat build failed.");

    let kernel_log_and_sum = pro_que
        .kernel_builder("log_and_sum_mat")
        .global_work_size(m)
        .arg(&final_result_buffer)
        .arg(coeffs)
        .arg(&max_buffer)
        .arg(n as u32)
        .build()
        .expect("Kernel log_and_sum_mat build failed.");

    for i in 0..n {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract
            .enq()
            .expect("Error while executing substract_without_origin kernel.");

        kernel_exponent_coefficients.set_arg("offset", i as u32).unwrap();
        kernel_exponent_coefficients
            .enq()
            .expect("Error while executing onlykde_exponent_coefficients_iterate_train_high_memory kernel.");
    }

    max_gpu_mat(
        &pro_que,
        coeffs,
        &tmp_reduc_buffers,
        &max_buffer,
        m,
        n,
        max_work_size,
        local_work_size,
        num_groups,
    );

    kernel_expmax_mat
        .enq()
        .expect("Error while executing expmax_mat kernel.");

    sum_gpu_mat(
        &pro_que,
        coeffs,
        &tmp_reduc_buffers,
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
    *error = Error::NoError;
}
