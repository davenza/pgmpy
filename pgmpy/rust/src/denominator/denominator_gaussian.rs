use crate::{empty_buffers, copy_buffers, Error, DoubleNumpyArray, buffer_fill_value,
            get_max_work_size, is_rowmajor, max_gpu_vec_copy, log_sum_gpu_vec, max_gpu_mat,
            sum_gpu_mat};

use crate::denominator::{CKDE, s2, s1_s3_coefficients};

use float_cmp::{ApproxEq, F64Margin};

use std::slice;
use std::f64;

use ocl::{Buffer, ProQue};
use libc::{c_double};

use crate::{to_cpu, to_cpu_single, equal_slices};

#[no_mangle]
pub unsafe extern "C" fn logdenominator_dataset_gaussian(
    ckde: *mut CKDE,
    pro_que: *mut ProQue,
    x: *const DoubleNumpyArray,
    result: *mut c_double,
    error: *mut Error,
) {
//    println!("\t[RUST] logdenominator_gaussian {:p}", ckde);
    *error = Error::NotFinished;
    let mut ckde = Box::from_raw(ckde);
    let mut pro_que = Box::from_raw(pro_que);
    let m = *(*x).shape;

    if (*ckde.kde).n >= m {
        logdenominator_iterate_test_gaussian(&mut ckde, &mut pro_que, x, result, error);
    } else {
        logdenominator_iterate_train_gaussian(&mut ckde, &mut pro_que, x, result, error);
    }

    Box::into_raw(ckde);
    Box::into_raw(pro_que);
}

fn onlygaussian_exponent_coefficients_iterate_test_name(rowmajor: bool) -> &'static str {
    if rowmajor {
        "onlygaussian_exponent_coefficients_iterate_test_rowmajor"
    } else {
        "onlygaussian_exponent_coefficients_iterate_test_columnmajor"
    }
}

fn onlygaussian_exponent_coefficients_iterate_train_high_memory_name(
    rowmajor: bool,
) -> &'static str {
    if rowmajor {
        "onlygaussian_exponent_coefficients_iterate_train_high_memory_rowmajor"
    } else {
        "onlygaussian_exponent_coefficients_iterate_train_high_memory_columnmajor"
    }
}

fn onlygaussian_exponent_coefficients_iterate_train_low_memory_checkmax_name(
    rowmajor: bool,
) -> &'static str {
    if rowmajor {
        "onlygaussian_exponent_coefficients_iterate_train_low_memory_checkmax_rowmajor"
    } else {
        "onlygaussian_exponent_coefficients_iterate_train_low_memory_checkmax_columnmajor"
    }
}

fn onlygaussian_exponent_coefficients_iterate_train_low_memory_compute_name(
    rowmajor: bool,
) -> &'static str {
    if rowmajor {
        "onlygaussian_exponent_coefficients_iterate_train_low_memory_compute_rowmajor"
    } else {
        "onlygaussian_exponent_coefficients_iterate_train_low_memory_compute_columnmajor"
    }
}

unsafe fn logdenominator_iterate_test_gaussian(
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
    let n = kde.n;

    let max_work_size = get_max_work_size(&pro_que);
    let local_work_size = if n < max_work_size { n } else { max_work_size };
    let num_groups = (n as f32 / local_work_size as f32).ceil() as usize;

    let test_slice = slice::from_raw_parts((*x).ptr, m * d_test);

    let (test_instances_buffer,) = copy_buffers!(pro_que, error, test_slice);
    let (test_instances_buffer2,) = copy_buffers!(pro_que, error, test_slice);

    let (s1, s3, final_result_buffer, tmp_train_coefficients, max_coefficients) =
        empty_buffers!(pro_que, error, f64, m, m, m, n, num_groups);

    let (s1_2, s3_2, final_result_buffer2, tmp_train_coefficients2, max_coefficients2) =
        empty_buffers!(pro_que, error, f64, m, m, m, n, num_groups);

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

    s1_s3_coefficients(
        ckde,
        pro_que,
        &test_instances_buffer2,
        test_leading_dimension as u32,
        test_rowmajor,
        &s1_2,
        &s3_2,
        m,
    );

    let (cpu, cpu2) = to_cpu!(pro_que, s1, s1_2);

    if !equal_slices!(cpu, cpu2) {
        println!("Bug in denominator s1 coefficients");
    }

    let (cpu, cpu2) = to_cpu!(pro_que, s3, s3_2);

    if !equal_slices!(cpu, cpu2) {
        println!("Bug in denominator s3 coefficients");
    }

    let kernel_onlygaussian_exponent_coefficients = pro_que
        .kernel_builder(onlygaussian_exponent_coefficients_iterate_test_name(
            kde.rowmajor,
        ))
        .global_work_size(n)
        .arg(&kde.training_data)
        .arg(kde.leading_dimension as u32)
        .arg(&ckde.precision)
        .arg(&s1)
        .arg((4.0*a).recip())
        .arg(&s3)
        .arg_named("test_index", &0u32)
        .arg(&tmp_train_coefficients)
        .build()
        .expect("Kernel onlygaussian_exponent_coefficients build failed.");

    let kernel_onlygaussian_exponent_coefficients2 = pro_que
        .kernel_builder(onlygaussian_exponent_coefficients_iterate_test_name(
            kde.rowmajor,
        ))
        .global_work_size(n)
        .arg(&kde.training_data)
        .arg(kde.leading_dimension as u32)
        .arg(&ckde.precision)
        .arg(&s1_2)
        .arg((4.0*a).recip())
        .arg(&s3_2)
        .arg_named("test_index", &0u32)
        .arg(&tmp_train_coefficients2)
        .build()
        .expect("Kernel onlygaussian_exponent_coefficients build failed.");

    let kernel_log_sum_gpu = pro_que
        .kernel_builder("copy_logpdf_result")
        .global_work_size(1)
        .arg(&tmp_train_coefficients)
        .arg(&max_coefficients)
        .arg(&final_result_buffer)
        .arg_named("offset", &0u32)
        .build()
        .expect("Kernel copy_logpdf_result build failed.");

    let kernel_log_sum_gpu2 = pro_que
        .kernel_builder("copy_logpdf_result")
        .global_work_size(1)
        .arg(&tmp_train_coefficients2)
        .arg(&max_coefficients2)
        .arg(&final_result_buffer2)
        .arg_named("offset", &0u32)
        .build()
        .expect("Kernel copy_logpdf_result build failed.");

    for i in 0..m {
        kernel_onlygaussian_exponent_coefficients.set_arg("test_index", i as u32).unwrap();
        kernel_onlygaussian_exponent_coefficients
            .enq()
            .expect("Error while executing kernel_onlygaussian_exponent_coefficients kernel.");

        kernel_onlygaussian_exponent_coefficients2.set_arg("test_index", i as u32).unwrap();
        kernel_onlygaussian_exponent_coefficients2
            .enq()
            .expect("Error while executing kernel_onlygaussian_exponent_coefficients kernel.");

        let (cpu, cpu2) = to_cpu!(pro_que, tmp_train_coefficients, tmp_train_coefficients2);

        if !equal_slices!(cpu, cpu2) {
            println!("Bug in denominator kernel onlygaussian_exponent_coefficients in index {}!", i);
        }

        max_gpu_vec_copy(
            pro_que,
            &tmp_train_coefficients,
            &max_coefficients,
            n,
            max_work_size,
            local_work_size,
            num_groups,
        );

        max_gpu_vec_copy(
            pro_que,
            &tmp_train_coefficients2,
            &max_coefficients2,
            n,
            max_work_size,
            local_work_size,
            num_groups,
        );

        let (cpu, cpu2) = to_cpu!(pro_que, tmp_train_coefficients, tmp_train_coefficients2);

        if !equal_slices!(cpu, cpu2) {
            println!("Bug in kernel denominator max_gpu_vec_copy in index {}!", i);
        }

        let (cpu, cpu2) = to_cpu!(pro_que, max_coefficients, max_coefficients2);

        let m : F64Margin = Default::default();
        if !cpu[0].approx_eq(cpu2[0], m) {
            println!("Bug in kernel denominator max_gpu_vec_copy in index {}! Max1: {}, Max2: {}", i, cpu[0], cpu2[0]);
        }

        log_sum_gpu_vec(
            &pro_que,
            &tmp_train_coefficients,
            &max_coefficients,
            n,
            max_work_size,
            local_work_size,
            num_groups,
        );

        log_sum_gpu_vec(
            &pro_que,
            &tmp_train_coefficients2,
            &max_coefficients2,
            n,
            max_work_size,
            local_work_size,
            num_groups,
        );

        let (cpu, cpu2) = to_cpu!(pro_que, tmp_train_coefficients, tmp_train_coefficients2);

        let m : F64Margin = Default::default();
        if !cpu[0].approx_eq(cpu2[0], m) {
            println!("Bug in denominator log_sum_gpu_vec in index {}! Sum1: {}, Sum2: {}", i, cpu[0], cpu2[0]);
        }

        let (cpu, cpu2) = to_cpu!(pro_que, max_coefficients, max_coefficients2);

        let m : F64Margin = Default::default();
        if !cpu[0].approx_eq(cpu2[0], m) {
            println!("Bug in denominator log_sum_gpu_vec in index {}! Max1: {}, Max2: {}", i, cpu[0], cpu2[0]);
        }

        kernel_log_sum_gpu.set_arg("offset", i as u32).unwrap();
        kernel_log_sum_gpu
            .enq()
            .expect("Error while executing copy_logpdf_result kernel.");

        kernel_log_sum_gpu2.set_arg("offset", i as u32).unwrap();
        kernel_log_sum_gpu2
            .enq()
            .expect("Error while executing copy_logpdf_result kernel.");

        let (cpu, cpu2) = to_cpu!(pro_que, final_result_buffer, final_result_buffer2);

        let m : F64Margin = Default::default();
        if !cpu[i].approx_eq(cpu2[i], m) {
            println!("Bug while denominator kernel copy_logpdf_result final in index {}! Sum1: {}, Sum2: {}", i, cpu[i], cpu2[i]);
        }
    }

    let kernel_sum_constant = pro_que
        .kernel_builder("sum_constant")
        .global_work_size(m)
        .arg(&final_result_buffer)
        .arg(ckde.lognorm_factor)
        .build()
        .expect("Kernel sum_constant build failed.");

    let kernel_sum_constant2 = pro_que
        .kernel_builder("sum_constant")
        .global_work_size(m)
        .arg(&final_result_buffer2)
        .arg(ckde.lognorm_factor)
        .build()
        .expect("Kernel sum_constant build failed.");

    kernel_sum_constant.enq().expect("Error while executing sum_constant kernel.");
    kernel_sum_constant2.enq().expect("Error while executing sum_constant kernel.");

    let (cpu, cpu2) = to_cpu!(pro_que, final_result_buffer, final_result_buffer2);

    if !equal_slices!(cpu, cpu2) {
        println!("Bug in kernel denominator kernel_sum_constant");
    }

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


//unsafe fn logdenominator_iterate_test_gaussian(
//    ckde: &Box<CKDE>,
//    pro_que: &mut Box<ProQue>,
//    x: *const DoubleNumpyArray,
//    result: *mut c_double,
//    error: *mut Error,
//) {
//    let kde = Box::from_raw(ckde.kde);
//    let test_shape = slice::from_raw_parts((*x).shape, 2);
//
//    let m = test_shape[0];
//    let d_test = test_shape[1];
//    let n = kde.n;
//
//    let max_work_size = get_max_work_size(&pro_que);
//    let local_work_size = if n < max_work_size { n } else { max_work_size };
//    let num_groups = (n as f32 / local_work_size as f32).ceil() as usize;
//
//    let test_slice = slice::from_raw_parts((*x).ptr, m * d_test);
//
//    let (test_instances_buffer,) = copy_buffers!(pro_que, error, test_slice);
//
//    let (s1, s3, final_result_buffer, tmp_train_coefficients, max_coefficients) =
//        empty_buffers!(pro_que, error, f64, m, m, m, n, num_groups);
//
//    let a = 0.5 * (ckde.precision_variable + s2(ckde));
//
//    let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);
//
//    s1_s3_coefficients(
//        ckde,
//        pro_que,
//        &test_instances_buffer,
//        test_leading_dimension as u32,
//        test_rowmajor,
//        &s1,
//        &s3,
//        m,
//    );
//
//    let kernel_onlygaussian_exponent_coefficients = pro_que
//        .kernel_builder(onlygaussian_exponent_coefficients_iterate_test_name(
//            kde.rowmajor,
//        ))
//        .global_work_size(n)
//        .arg(&kde.training_data)
//        .arg(kde.leading_dimension as u32)
//        .arg(&ckde.precision)
//        .arg(&s1)
//        .arg((4.0*a).recip())
//        .arg(&s3)
//        .arg_named("test_index", &0u32)
//        .arg(&tmp_train_coefficients)
//        .build()
//        .expect("Kernel onlygaussian_exponent_coefficients build failed.");
//
//    let kernel_log_sum_gpu = pro_que
//        .kernel_builder("copy_logpdf_result")
//        .global_work_size(1)
//        .arg(&tmp_train_coefficients)
//        .arg(&max_coefficients)
//        .arg(&final_result_buffer)
//        .arg_named("offset", &0u32)
//        .build()
//        .expect("Kernel copy_logpdf_result build failed.");
//
//    for i in 0..m {
//        kernel_onlygaussian_exponent_coefficients.set_arg("test_index", i as u32).unwrap();
//        kernel_onlygaussian_exponent_coefficients
//            .enq()
//            .expect("Error while executing kernel_onlygaussian_exponent_coefficients kernel.");
//
//        max_gpu_vec_copy(
//            pro_que,
//            &tmp_train_coefficients,
//            &max_coefficients,
//            n,
//            max_work_size,
//            local_work_size,
//            num_groups,
//        );
//
//        log_sum_gpu_vec(
//            &pro_que,
//            &tmp_train_coefficients,
//            &max_coefficients,
//            n,
//            max_work_size,
//            local_work_size,
//            num_groups,
//        );
//
//        kernel_log_sum_gpu.set_arg("offset", i as u32).unwrap();
//        kernel_log_sum_gpu
//            .enq()
//            .expect("Error while executing copy_logpdf_result kernel.");
//    }
//
//    let kernel_sum_constant = pro_que
//        .kernel_builder("sum_constant")
//        .global_work_size(m)
//        .arg(&final_result_buffer)
//        .arg(ckde.lognorm_factor)
//        .build()
//        .expect("Kernel sum_constant build failed.");
//
//    kernel_sum_constant.enq().expect("Error while executing sum_constant kernel.");
//    let final_result = slice::from_raw_parts_mut(result, m);
//    final_result_buffer
//        .cmd()
//        .queue(pro_que.queue())
//        .read(final_result)
//        .enq()
//        .expect("Error reading result data.");
//
//    Box::into_raw(kde);
//    *error = Error::NoError;
//}

unsafe fn logdenominator_iterate_train_gaussian(
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
            logdenominator_iterate_train_gaussian_high_memory(ckde, pro_que, x, result, &b, error)
        }
        Err(_) => {
            // TODO: If n < 2m, is it better to iterate over the training data?
            logdenominator_iterate_train_gaussian_low_memory(
                ckde,
                pro_que,
                x,
                result,
                error,
            );
        }
    }
}

unsafe fn logdenominator_iterate_train_gaussian_low_memory(
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
    let n = kde.n;

    let test_slice = slice::from_raw_parts((*x).ptr, m * d_test);

    let (test_instances_buffer,) = copy_buffers!(pro_que, error, test_slice);

    let (s1, s3, final_result_buffer, max_coefficients) =
        empty_buffers!(pro_que, error, f64, m, m, m, m);

    buffer_fill_value(&pro_que, &max_coefficients, m, f64::MIN);
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

    let kernel_onlygaussian_exponent_coefficients_checkmax = pro_que
        .kernel_builder(
            onlygaussian_exponent_coefficients_iterate_train_low_memory_checkmax_name(
                kde.rowmajor,
            ),
        )
        .global_work_size(m)
        .arg(&kde.training_data)
        .arg(kde.leading_dimension as u32)
        .arg(&ckde.precision)
        .arg(&s1)
        .arg((4.0*a).recip())
        .arg(&s3)
        .arg_named("train_index", &0u32)
        .arg(&max_coefficients)
        .build()
        .expect("Kernel kernel_onlygaussian_exponent_coefficients_checkmax build failed.");

    let kernel_onlygaussian_exponent_coefficients_compute = pro_que
        .kernel_builder(
            onlygaussian_exponent_coefficients_iterate_train_low_memory_compute_name(kde.rowmajor),
        )
        .global_work_size(m)
        .arg(&kde.training_data)
        .arg(kde.leading_dimension as u32)
        .arg(&ckde.precision)
        .arg(&s1)
        .arg((4.0*a).recip())
        .arg(&s3)
        .arg_named("train_index", &0u32)
        .arg(&max_coefficients)
        .arg(&final_result_buffer)
        .build()
        .expect("Kernel kernel_onlygaussian_exponent_coefficients_checkmax build failed.");

    let kernel_log_and_sum = pro_que
        .kernel_builder("log_and_sum")
        .global_work_size(m)
        .arg(&final_result_buffer)
        .arg(&max_coefficients)
        .build()
        .expect("Kernel log_and_sum build failed.");

    for i in 0..n {
        kernel_onlygaussian_exponent_coefficients_checkmax
            .set_arg("train_index", i as u32)
            .unwrap();
        kernel_onlygaussian_exponent_coefficients_checkmax
            .enq()
            .expect(
                "Error while executing kernel_onlygaussian_exponent_coefficients_checkmax kernel.",
            );
    }

    for i in 0..n {
        kernel_onlygaussian_exponent_coefficients_compute
            .set_arg("train_index", i as u32)
            .unwrap();
        kernel_onlygaussian_exponent_coefficients_compute
            .enq()
            .expect(
                "Error while executing kernel_onlygaussian_exponent_coefficients_compute kernel.",
            );
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

unsafe fn logdenominator_iterate_train_gaussian_high_memory(
    ckde: &Box<CKDE>,
    pro_que: &mut Box<ProQue>,
    x: *const DoubleNumpyArray,
    result: *mut c_double,
    tmp_coefficients: &Buffer<f64>,
    error: *mut Error,
) {
    let kde = Box::from_raw(ckde.kde);
    let test_shape = slice::from_raw_parts((*x).shape, 2);

    let m = test_shape[0];
    let d_test = test_shape[1];
    let n = kde.n;

    let max_work_size = get_max_work_size(&pro_que);
    let local_work_size = if n < max_work_size { n } else { max_work_size };
    let num_groups = (n as f32 / local_work_size as f32).ceil() as usize;

    let test_slice = slice::from_raw_parts((*x).ptr, m * d_test);

    let (test_instances_buffer,) = copy_buffers!(pro_que, error, test_slice);

    let (s1, s3, final_result_buffer, max_coefficients) =
        empty_buffers!(pro_que, error, f64, m, m, m, m * num_groups);

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

    let kernel_onlygaussian_exponent_coefficients = pro_que
        .kernel_builder(
            onlygaussian_exponent_coefficients_iterate_train_high_memory_name(kde.rowmajor),
        )
        .global_work_size(m * n)
        .arg(&kde.training_data)
        .arg(kde.leading_dimension as u32)
        .arg(&ckde.precision)
        .arg(&s1)
        .arg((4.0*a).recip())
        .arg(&s3)
        .arg(tmp_coefficients)
        .arg(n as u32)
        .build()
        .expect("Kernel onlygaussian_exponent_coefficients build failed.");

    kernel_onlygaussian_exponent_coefficients
        .enq()
        .expect("Error while executing kernel_onlygaussian_exponent_coefficients kernel.");

    max_gpu_mat(
        &pro_que,
        tmp_coefficients,
        &max_coefficients,
        m,
        n,
        max_work_size,
        local_work_size,
        num_groups,
    );

    let kernel_exp_and_sum_mat = pro_que
        .kernel_builder("exp_and_sum_mat")
        .global_work_size(m * n)
        .arg(tmp_coefficients)
        .arg(&max_coefficients)
        .arg(n as u32)
        .arg(num_groups as u32)
        .build()
        .expect("Kernel exp_and_sum_mat build failed.");

    kernel_exp_and_sum_mat
        .enq()
        .expect("Error while executing exp_and_sum_mat kernel.");

    sum_gpu_mat(
        &pro_que,
        tmp_coefficients,
        m,
        n,
        max_work_size,
        local_work_size,
        num_groups,
    );

    let kernel_log_and_sum = pro_que
        .kernel_builder("log_and_sum_mat")
        .global_work_size(m)
        .arg(&final_result_buffer)
        .arg(tmp_coefficients)
        .arg(&max_coefficients)
        .arg(n as u32)
        .arg(num_groups as u32)
        .build()
        .expect("Kernel log_and_sum_mat build failed.");

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
