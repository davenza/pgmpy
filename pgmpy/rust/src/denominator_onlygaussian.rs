use crate::{copy_buffers, empty_buffers, print_buffers};

use crate::{
    buffer_fill_value, gaussian_kde_free, get_max_work_size, log_sum_gpu_vec, max_gpu_mat,
    max_gpu_vec_copy, sum_gpu_mat, DoubleNumpyArray, Error, GaussianKDE, is_rowmajor
};

use libc::{c_double, c_uint};
use ocl::{Buffer, ProQue};

use std::f64;
use std::mem;
use std::ptr;
use std::slice;

use std::any::type_name;

pub struct GaussianRegression {
    variable_index: c_uint,
    beta: Buffer<f64>,
    variable_beta: c_double,
    evidence_index: Option<Buffer<u32>>,
    nparents: c_uint,
    variance: c_double,
}

#[no_mangle]
pub unsafe extern "C" fn gaussian_regression_init(
    pro_que: *mut ProQue,
    variable_index: c_uint,
    beta: *mut c_double,
    evidence_index: *mut c_uint,
    nparents: c_uint,
    variance: c_double,
    error: *mut Error,
) -> *mut GaussianRegression {
    println!("Init GaussianRegression");
    let mut pro_que = Box::from_raw(pro_que);

    let beta_slice = slice::from_raw_parts(beta, (nparents + 1) as usize);
    let (beta_buffer,) = copy_buffers!(pro_que, error, beta_slice => ptr::null_mut());

    let evidence_index_buffer = if nparents > 1 {
        let evidence_index_slice = slice::from_raw_parts(evidence_index, (nparents - 1) as usize);
        let (tmp,) = copy_buffers!(pro_que, error, evidence_index_slice => ptr::null_mut());
        Some(tmp)
    } else {
        None
    };

    let gr = Box::new(GaussianRegression {
        variable_index: variable_index,
        beta: beta_buffer,
        variable_beta: beta_slice[1],
        evidence_index: evidence_index_buffer,
        nparents: nparents,
        variance: variance,
    });

    let ptr_gr = Box::into_raw(gr);

    Box::into_raw(pro_que);
    *error = Error::NoError;
    ptr_gr
}

#[no_mangle]
pub unsafe extern "C" fn gaussian_regression_free(gr: *mut GaussianRegression) {
    println!("Free GaussianRegression");
    if gr.is_null() {
        return;
    }
    unsafe {
        Box::from_raw(gr);
    }
}

pub struct CKDE {
    kde: *mut GaussianKDE,
    precision: Buffer<f64>,
    precision_variable: f64,
    regressions: *mut *mut GaussianRegression,
    nregressions: c_uint,
}

#[no_mangle]
pub unsafe extern "C" fn ckde_init(
    pro_que: *mut ProQue,
    kde: *mut GaussianKDE,
    precision: *mut c_double,
    regressions: *mut *mut GaussianRegression,
    nregressions: c_uint,
    error: *mut Error,
) -> *mut CKDE {
    println!("Init CKDE");

    let d = (*kde).d;
    let mut pro_que = Box::from_raw(pro_que);

    let precision_slice = slice::from_raw_parts(precision, d * d);

    let (precision_buffer,) = copy_buffers!(pro_que, error, precision_slice => ptr::null_mut());

    let ckde = Box::new(CKDE {
        kde,
        precision: precision_buffer,
        precision_variable: precision_slice[0],
        regressions,
        nregressions,
    });

    Box::into_raw(pro_que);
    let ptr_ckde = Box::into_raw(ckde);
    ptr_ckde
}

#[no_mangle]
pub unsafe extern "C" fn ckde_free(ckde: *mut CKDE) {
    println!("Free CKDE");

    if ckde.is_null() {
        return;
    }

    let ckde = Box::from_raw(ckde);
    let kde_ptr = ckde.kde;
    let regressions = ckde.regressions;
    let nregressions = ckde.nregressions;

    mem::drop(ckde);

    gaussian_kde_free(kde_ptr);

    for i in 0..nregressions {
        gaussian_regression_free((*regressions).offset(i as isize));
    }
}

#[no_mangle]
pub unsafe extern "C" fn logdenominator_dataset_gaussian(
    ckde: *mut CKDE,
    pro_que: *mut ProQue,
    x: *const DoubleNumpyArray,
    result: *mut c_double,
    error: *mut Error,
) {
    let mut ckde_box = Box::from_raw(ckde);
    let mut pro_que = Box::from_raw(pro_que);
    let m = *(*x).shape;

    *error = Error::NoError;
    if (*ckde_box.kde).n >= m {
        logdenominator_iterate_test_gaussian(
            &mut ckde_box,
            &mut pro_que,
            x,
            result,
            error,
        );
    } else {
//        logdenominator_iterate_train_onlykde(
//            &mut kde_box,
//            &mut pro_que,
//            x,
//            precision,
//            result,
//            error,
//        );
    }

    Box::into_raw(ckde_box);
    Box::into_raw(pro_que);
}

unsafe fn s2(ckde: &Box<CKDE>) -> f64 {
    let mut s2 = 0.0f64;
    for i in 0..ckde.nregressions {
        let coeff = (*(*ckde.regressions).offset(i as isize)).variable_beta;
        s2 += (coeff*coeff) / (*(*ckde.regressions).offset(i as isize)).variance;
    }

    s2
}

fn s1_and_s3_constant_name(rowmajor: bool) -> &'static str {
    if rowmajor {
        "s1_and_s3_sum_constant_rowmajor"
    } else {
        "s1_and_s3_sum_constant_columnmajor"
    }
}

fn s1_and_s3_parents_name(rowmajor: bool) -> &'static str {
    if rowmajor {
        "s1_and_s3_sum_parents_rowmajor"
    } else {
        "s1_and_s3_sum_parents_columnmajor"
    }
}

unsafe fn logdenominator_iterate_test_gaussian(ckde: &Box<CKDE>,
                                                pro_que: &mut Box<ProQue>,
                                                x: *const DoubleNumpyArray,
                                                result: *mut c_double,
                                                error: *mut Error) {

    let kde = Box::from_raw(ckde.kde);
    let test_shape = slice::from_raw_parts((*x).shape, 2);

    let m = test_shape[0];
    let d_test = test_shape[1];

    let nparents_kde = d_test - 1;
    let n = kde.n;

    let test_slice = slice::from_raw_parts((*x).ptr, m * d_test);

    let (test_instances_buffer,) = copy_buffers!(pro_que, error, test_slice);

    let (s1, s3, final_result_buffer) =
        empty_buffers!(pro_que, error, f64, m, m, m);

    let a = 0.5*(ckde.precision_variable +  s2(ckde));

    buffer_fill_value(pro_que, &s1, m, 0.0f64);
    buffer_fill_value(pro_que, &s3, m, 0.0f64);

    let (test_rowmajor, test_leading_dimension) = is_rowmajor(x);

    let kernel_s1_and_s3_sum_constant = pro_que
        .kernel_builder(s1_and_s3_constant_name(test_rowmajor))
        .global_work_size(m)
        .arg(&test_instances_buffer)
        .arg(test_leading_dimension as u32)
        .arg_named("beta", None::<&Buffer<f64>>)
        .arg_named("variable_index", &0u32)
        .arg_named("inv_variance", &0.0f64)
        .arg(&s1)
        .arg(&s3)
        .build()
        .expect("Kernel s1_and_s3_sum_constant build failed.");

    let kernel_s1_and_s3_sum_parents = pro_que
        .kernel_builder(s1_and_s3_parents_name(test_rowmajor))
        .global_work_size(m)
        .arg(&test_instances_buffer)
        .arg(test_leading_dimension as u32)
        .arg_named("beta", None::<&Buffer<f64>>)
        .arg_named("variable_index", &0u32)
        .arg_named("evidence_index", None::<&Buffer<u32>>)
        .arg_named("len_evidence", &0u32)
        .arg_named("inv_variance", &0.0f64)
        .arg(&s1)
        .arg(&s3)
        .build()
        .expect("Kernel s1_and_s3_sum_parents build failed.");

    println!();
    print_buffers!(pro_que, test_instances_buffer);
    for i in 0..ckde.nregressions {
        println!("========================");
        println!("Iteration {}:", i);
        println!("========================");
        print_buffers!(pro_que, s1, s3);

        let gr = &**ckde.regressions.offset(i as isize);

        if gr.nparents > 1 {
            kernel_s1_and_s3_sum_parents.set_arg("beta", &gr.beta).unwrap();
            kernel_s1_and_s3_sum_parents.set_arg("variable_index", gr.variable_index).unwrap();
            kernel_s1_and_s3_sum_parents.set_arg("evidence_index", gr.evidence_index.as_ref().unwrap()).unwrap();
            kernel_s1_and_s3_sum_parents.set_arg("len_evidence", gr.nparents - 1).unwrap();
            kernel_s1_and_s3_sum_parents.set_arg("inv_variance", gr.variance.recip()).unwrap();
            kernel_s1_and_s3_sum_parents
                .enq()
                .expect("Error while executing kernel_s1_and_s3_sum_parents kernel.");
        }
        else {
            kernel_s1_and_s3_sum_constant.set_arg("beta", &gr.beta).unwrap();
            kernel_s1_and_s3_sum_constant.set_arg("variable_index", gr.variable_index).unwrap();
            kernel_s1_and_s3_sum_constant.set_arg("inv_variance", gr.variance.recip()).unwrap();
            kernel_s1_and_s3_sum_constant
                .enq()
                .expect("Error while executing kernel_s1_and_s3_sum_constant kernel.");
        }
    }

    print_buffers!(pro_que, s1, s3);

    Box::into_raw(kde);
}