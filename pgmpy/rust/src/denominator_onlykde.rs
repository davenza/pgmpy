use crate::{copy_buffers, empty_buffers};

use crate::{GaussianKDE, Error, DoubleNumpyArray, get_max_work_size, max_gpu_vec_copy,
            log_sum_gpu_vec, buffer_fill_value, max_gpu_mat, sum_gpu_mat};

use ocl::{Buffer, ProQue};
use libc::{c_double};

use std::slice;
use std::f64;

#[no_mangle]
pub unsafe extern "C" fn logdenominator_dataset_onlykde (
    kde: *mut GaussianKDE,
    pro_que: *mut ProQue,
    x: *const DoubleNumpyArray,
    precision: *const DoubleNumpyArray,
    result: *mut c_double,
    error: *mut Error,
)
{
    let mut kde_box = Box::from_raw(kde);
    let mut pro_que = Box::from_raw(pro_que);
    let m = *(*x).shape;

    *error = Error::NoError;
    if kde_box.n >= m {
        logdenominator_iterate_test_onlykde(&mut kde_box, &mut pro_que, x, precision, result, error);
    } else {
        logdenominator_iterate_train_onlykde(&mut kde_box, &mut pro_que, x, precision, result, error);
    }

    Box::into_raw(kde_box);
    Box::into_raw(pro_que);
}

unsafe fn logdenominator_iterate_test_onlykde(kde: &mut Box<GaussianKDE>,
                                              pro_que: &mut Box<ProQue>,
                                              x: *const DoubleNumpyArray,
                                              precision: *const DoubleNumpyArray,
                                              result: *mut c_double,
                                              error: *mut Error)
{
    let m = *(*x).shape;
    let d = kde.d;
    let nparents_kde = d - 1;
    let n = kde.n;

    let test_slice = slice::from_raw_parts((*x).ptr, m*d);
    let precision_slice = slice::from_raw_parts((*precision).ptr, (*precision).size);

    let (test_instances_buffer, precision_buffer) =
        copy_buffers!(pro_que, error, test_slice, precision_slice);


    let (ti_buffer, final_result_buffer, bi, ci) =
        empty_buffers!(pro_que, error, f64, n*nparents_kde, m, n, n);

    let a = 0.5*precision_slice[0];

    let max_work_size = get_max_work_size(&pro_que);
    let local_work_size = if n < max_work_size { n } else { max_work_size };
    let num_groups = (n as f32 / local_work_size as f32).ceil() as usize;

    let kernel_substract = pro_que
        .kernel_builder("substract_without_origin")
        .global_work_size(n*nparents_kde)
        .arg(&kde.training_data)
        .arg(&test_instances_buffer)
        .arg(&ti_buffer)
        .arg_named("row", &0u32)
        .arg(nparents_kde as u32)
        .build()
        .expect("Kernel substract_without_origin build failed.");

    let kernel_mahalanobis = pro_que
        .kernel_builder("mahalanobis")
        .global_work_size(n)
        .arg(&ti_buffer)
        .arg(&precision_buffer)
        .arg(&ci)
        .arg(nparents_kde as u32)
        .build()
        .expect("Kernel mahalanobis build failed.");

    let kernel_dot_product_bi = pro_que
        .kernel_builder("dot_product_bi")
        .global_work_size(n)
        .arg(&ti_buffer)
        .arg(&precision_buffer)
        .arg(&bi)
        .arg(nparents_kde as u32)
        .build()
        .expect("Kernel mahalanobis build failed.");

    let kernel_generate_exponents = pro_que
        .kernel_builder("generate_exponents_onlykde")
        .global_work_size(n)
        .arg(&kde.training_data)
        .arg(a.recip())
        .arg(&bi)
        .arg(&ci)
        .arg(&precision_buffer)
        .arg(d as u32)
        .build()
        .expect("Kernel generate_exponents_onlykde build failed.");

    let kernel_copy_logpdf_result = pro_que
        .kernel_builder("copy_logpdf_result")
        .global_work_size(1)
        .arg(&bi)
        .arg(&ci)
        .arg(&final_result_buffer)
        .arg_named("res_offset", &0u32)
        .build()
        .expect("Kernel copy_logpdf_result build failed.");

    for i in 0..m {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract
            .enq()
            .expect("Error while executing substract_without_origin kernel.");
        kernel_mahalanobis
            .enq()
            .expect("Error while executing mahalanobis kernel.");
        kernel_dot_product_bi
            .enq()
            .expect("Error while executing dot_product_bi kernel.");
        kernel_generate_exponents
            .enq()
            .expect("Error while executing generate_exponents kernel.");

        max_gpu_vec_copy(&pro_que, &bi, &ci,
                         n, max_work_size, local_work_size, num_groups);

        log_sum_gpu_vec(&pro_que, &bi, &ci, n, max_work_size, local_work_size, num_groups);

        kernel_copy_logpdf_result.set_arg("res_offset", i as u32).unwrap();
        kernel_copy_logpdf_result
            .enq()
            .expect("Error while executing kernel_copy_logpdf_result kernel.");
    }

    let final_result = slice::from_raw_parts_mut(result, m);

    final_result_buffer
        .cmd()
        .queue(pro_que.queue())
        .read(final_result)
        .enq()
        .expect("Error reading result data.");


}

unsafe fn logdenominator_iterate_train_onlykde(kde: &mut Box<GaussianKDE>,
                                               pro_que: &mut Box<ProQue>,
                                               x: *const DoubleNumpyArray,
                                               precision: *const DoubleNumpyArray,
                                               result: *mut c_double,
                                               error: *mut Error)
{
    let m = *(*x).shape;
    let tmp_vec_buffer = Buffer::<f64>::builder()
        .context(pro_que.context())
        .len(m * kde.n)
        .build();

    match tmp_vec_buffer {
        Ok(b) => logdenominator_iterate_train_high_memory_onlykde(kde, pro_que, x, precision, result, &b, error),
        Err(_) => {
            // TODO: If n < 2m, is it better to iterate over the training data?
            logdenominator_iterate_train_low_memory_onlykde(kde, pro_que, x, precision, result, error);
        }
    }

}


unsafe fn logdenominator_iterate_train_low_memory_onlykde(kde: &mut Box<GaussianKDE>,
                                                          pro_que: &mut Box<ProQue>,
                                                          x: *const DoubleNumpyArray,
                                                          precision: *const DoubleNumpyArray,
                                                          result: *mut c_double,
                                                          error: *mut Error)
{
    let m = *(*x).shape;
    let d = kde.d;
    let nparents_kde = d - 1;
    let n = kde.n;

    let test_slice = slice::from_raw_parts((*x).ptr, m * d);
    let precision_slice = slice::from_raw_parts((*precision).ptr, (*precision).size);

    let (test_instances_buffer, precision_buffer) = copy_buffers!(pro_que, error, test_slice, precision_slice);

    let (max_buffer, final_result_buffer, ti_buffer, bi, ci) =
        empty_buffers!(pro_que, error, f64, m, m, m*nparents_kde, m, m);

    let a = 0.5*precision_slice[0];

    buffer_fill_value(&pro_que, &max_buffer, m, f64::MIN);
    buffer_fill_value(&pro_que, &final_result_buffer, m, 0.0f64);

    let kernel_substract = pro_que
        .kernel_builder("substract_without_origin")
        .arg(&test_instances_buffer)
        .arg(&kde.training_data)
        .arg(&ti_buffer)
        .arg_named("row", &0u32)
        .arg(nparents_kde as u32)
        .build()
        .expect("Kernel substract_without_origin build failed.");

    let kernel_mahalanobis = pro_que
        .kernel_builder("mahalanobis")
        .global_work_size(m)
        .arg(&ti_buffer)
        .arg(&precision_buffer)
        .arg(&ci)
        .arg(nparents_kde as u32)
        .build()
        .expect("Kernel mahalanobis build failed.");

    let kernel_dot_product_bi = pro_que
        .kernel_builder("dot_product_bi")
        .global_work_size(m)
        .arg(&ti_buffer)
        .arg(&precision_buffer)
        .arg(&bi)
        .arg(nparents_kde as u32)
        .build()
        .expect("Kernel dot_product_bi build failed.");

    let kernel_coefficients_checkmax = pro_que
        .kernel_builder("coefficients_checkmax_onlykde")
        .global_work_size(m)
        .arg(&kde.training_data)
        .arg(a.recip())
        .arg(&bi)
        .arg(&ci)
        .arg(&precision_buffer)
        .arg(&max_buffer)
        .arg(d as u32)
        .build()
        .expect("Kernel coefficients_checkmax_onlykde build failed.");

    let kernel_coefficients_sum = pro_que
        .kernel_builder("coefficients_sum_onlykde")
        .global_work_size(m)
        .arg(&kde.training_data)
        .arg(a.recip())
        .arg(&bi)
        .arg(&ci)
        .arg(&precision_buffer)
        .arg(&max_buffer)
        .arg(&final_result_buffer)
        .arg(d as u32)
        .build()
        .expect("Kernel coefficients_and_sum build failed.");

    let kernel_log_and_sum = pro_que
        .kernel_builder("log_and_sum")
        .global_work_size(m)
        .arg(&final_result_buffer)
        .arg(&max_buffer)
        .build()
        .expect("Kernel log_and_sum build failed.");


    // Writes the max value in the max_buffer
    // TODO: Find max with euclidian distance is probably faster.
    for i in 0..n {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract
            .enq()
            .expect("Error while executing substract_without_origin kernel.");
        kernel_mahalanobis
            .enq()
            .expect("Error while executing mahalanobis kernel.");
        kernel_dot_product_bi
            .enq()
            .expect("Error while executing dot_product_bi kernel.");
        kernel_coefficients_checkmax
            .enq()
            .expect("Error while executing coefficients_checkmax kernel.");
    }

    for i in 0..n {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract
            .enq()
            .expect("Error while executing substract_without_origin kernel.");
        kernel_mahalanobis
            .enq()
            .expect("Error while executing mahalanobis kernel.");
        kernel_dot_product_bi
            .enq()
            .expect("Error while executing dot_product_bi kernel.");
        kernel_coefficients_sum
            .enq()
            .expect("Error while executing coefficients_sum kernel.");
    }

    kernel_log_and_sum
        .enq()
        .expect("Error while executing log_and_sum kernel.");

    let final_result = slice::from_raw_parts_mut(result, m);

    final_result_buffer
        .cmd()
        .queue(pro_que.queue())
        .read(final_result)
        .enq()
        .expect("Error reading result data.");
}

unsafe fn logdenominator_iterate_train_high_memory_onlykde(kde: &mut Box<GaussianKDE>,
                                                           pro_que: &mut Box<ProQue>,
                                                           x: *const DoubleNumpyArray,
                                                           precision: *const DoubleNumpyArray,
                                                           result: *mut c_double,
                                                           coefficients_buffer: &Buffer<f64>,
                                                           error: *mut Error)
{
    let m = *(*x).shape;
    let d = kde.d;
    let nparents_kde = d - 1;
    let n = kde.n;

    let max_work_size = get_max_work_size(&pro_que);
    let local_work_size = if n < max_work_size { n } else { max_work_size };
    let num_groups = (n as f32 / local_work_size as f32).ceil() as usize;

    let test_slice = slice::from_raw_parts((*x).ptr, m * d);
    let precision_slice = slice::from_raw_parts((*precision).ptr, d * d);
    let (test_instances_buffer, precision_buffer) = copy_buffers!(pro_que, error, test_slice, precision_slice);
    let (max_buffer, final_result_buffer, ti_buffer, bi, ci) =
        empty_buffers!(pro_que, error, f64, m * num_groups, m, m*nparents_kde, m, m);

    let a = 0.5*precision_slice[0];

    let kernel_substract = pro_que
        .kernel_builder("substract_without_origin")
        .global_work_size(n*nparents_kde)
        .arg(&test_instances_buffer)
        .arg(&kde.training_data)
        .arg(&ti_buffer)
        .arg_named("row", &0u32)
        .arg(nparents_kde as u32)
        .build()
        .expect("Kernel substract_without_origin build failed.");


    let kernel_mahalanobis = pro_que
        .kernel_builder("mahalanobis")
        .global_work_size(m)
        .arg(&ti_buffer)
        .arg(&precision_buffer)
        .arg(&ci)
        .arg(nparents_kde as u32)
        .build()
        .expect("Kernel mahalanobis build failed.");

    let kernel_dot_product_bi = pro_que
        .kernel_builder("dot_product_bi")
        .global_work_size(m)
        .arg(&ti_buffer)
        .arg(&precision_buffer)
        .arg(&bi)
        .arg(nparents_kde as u32)
        .build()
        .expect("Kernel dot_product_bi build failed.");

    let kernel_fill_coefficients_mat = pro_que
        .kernel_builder("fill_coefficients_mat_onlykde")
        .global_work_size(m)
        .arg(&kde.training_data)
        .arg(a.recip())
        .arg(&bi)
        .arg(&ci)
        .arg(&precision_buffer)
        .arg(coefficients_buffer)
        .arg_named("col", &0u32)
        .arg(n)
        .build()
        .expect("Kernel fill_coefficients_mat build failed.");

    let kernel_exp_and_sum = pro_que
        .kernel_builder("exp_and_sum_mat")
        .global_work_size((m, n))
        .arg(coefficients_buffer)
        .arg(&max_buffer)
        .arg(num_groups as u32)
        .build()
        .expect("Kernel exp_and_sum_mat build failed.");

    let kernel_log_and_sum = pro_que
        .kernel_builder("log_and_sum_mat")
        .global_work_size(m)
        .arg(&final_result_buffer)
        .arg(coefficients_buffer)
        .arg(&max_buffer)
        .arg(n as u32)
        .arg(num_groups as u32)
        .build()
        .expect("Kernel log_and_sum_mat build failed.");

    for i in 0..n {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract
            .enq()
            .expect("Error while executing substract_without_origin kernel.");
        kernel_mahalanobis
            .enq()
            .expect("Error while executing mahalanobis kernel.");
        kernel_dot_product_bi
            .enq()
            .expect("Error while executing dot_product_bi kernel.");
        kernel_fill_coefficients_mat
            .enq()
            .expect("Error while executing fill_coefficients_mat kernel.");
    }

    max_gpu_mat(&pro_que, coefficients_buffer, &max_buffer, m, n,
                max_work_size, local_work_size, num_groups);
    kernel_exp_and_sum
        .enq()
        .expect("Error while executing exp_and_sum_mat kernel.");
    sum_gpu_mat(&pro_que, coefficients_buffer, m, n,
                max_work_size, local_work_size, num_groups);
    kernel_log_and_sum
        .enq()
        .expect("Error while executing kernel log_and_sum_mat kernel.");

    let final_result = slice::from_raw_parts_mut(result, m);
    final_result_buffer
        .cmd()
        .queue(pro_que.queue())
        .read(final_result)
        .enq()
        .expect("Error reading result data.");
}