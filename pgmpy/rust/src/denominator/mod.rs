use crate::{empty_buffers, copy_buffers, Error, GaussianKDE, buffer_fill_value};


use std::slice;
use std::ptr;

use ocl::{Buffer, ProQue};
use libc::{c_uint, c_double};

mod denominator_gaussian;
pub use denominator_gaussian::logdenominator_dataset_gaussian;

mod denominator_onlykde;
pub use denominator_onlykde::logdenominator_dataset_onlykde;

mod denominator_mix;
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
pub unsafe extern "C" fn gaussian_regression_init(
    pro_que: *mut ProQue,
    variable_index: c_uint,
    beta: *mut c_double,
    evidence_index: *mut c_uint,
    nparents: c_uint,
    variance: c_double,
    error: *mut Error,
) -> *mut GaussianRegression {
    let pro_que = Box::from_raw(pro_que);

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
    if gr.is_null() {
        return;
    }

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
    lognorm_factor: f64
}

#[no_mangle]
pub unsafe extern "C" fn ckde_init(
    pro_que: *mut ProQue,
    kde: *mut GaussianKDE,
    precision: *mut c_double,
    kde_indices: *mut c_uint,
    regressions: *mut *mut GaussianRegression,
    nregressions: c_uint,
    lognorm_factor: f64,
    error: *mut Error,
) -> *mut CKDE {

    let d = (*kde).d;
    let pro_que = Box::from_raw(pro_que);

    let precision_slice = slice::from_raw_parts(precision, d * d);

    let (precision_buffer,) = copy_buffers!(pro_que, error, precision_slice => ptr::null_mut());

    let (onlykde_precision,) = if d > 1 {
        empty_buffers!(pro_que, error, f64, (d-1)*(d-1) => ptr::null_mut())
    } else {
//        Create a dummy buffer.
        empty_buffers!(pro_que, error, f64, 1 => ptr::null_mut())
    };

    let kde_indices_slice = slice::from_raw_parts(kde_indices, d);
    let kde_indices = if d > 1 {
        let (b,) = copy_buffers!(pro_que, error, kde_indices_slice => ptr::null_mut());
        Some(b)
    } else {
        None
    };

    let kernel_precompute_onlykde_precision =  pro_que
        .kernel_builder("precompute_marginal_precision")
        .global_work_size((d-1)*(d-1))
        .arg(&precision_buffer)
        .arg(precision_slice[0].recip())
        .arg(d as u32)
        .arg(&onlykde_precision)
        .build()
        .expect("Kernel sum_constant build failed.");

    kernel_precompute_onlykde_precision
        .enq()
        .expect("Error while executing substract_without_origin kernel.");

    let ckde = Box::new(CKDE {
        kde,
        precision: precision_buffer,
        precision_variable: precision_slice[0],
        marginal_precision: onlykde_precision,
        kde_indices,
        regressions,
        nregressions,
        lognorm_factor,
    });

    Box::into_raw(pro_que);
    let ptr_ckde = Box::into_raw(ckde);
    ptr_ckde
}

#[no_mangle]
pub unsafe extern "C" fn ckde_free(ckde: *mut CKDE) {
    if ckde.is_null() {
        return;
    }

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

unsafe fn s1_s3_coefficients(
    ckde: &Box<CKDE>,
    pro_que: &mut Box<ProQue>,
    test_instances_buffer: &Buffer<f64>,
    test_leading_dimension: u32,
    test_rowmajor: bool,
    s1: &Buffer<f64>,
    s3: &Buffer<f64>,
    m: usize,
) {
    buffer_fill_value(pro_que, &s1, m, 0.0f64);
    buffer_fill_value(pro_que, &s3, m, 0.0f64);

    let kernel_s1_and_s3_sum_constant = pro_que
        .kernel_builder(s1_and_s3_constant_name(test_rowmajor))
        .global_work_size(m)
        .arg(test_instances_buffer)
        .arg(test_leading_dimension)
        .arg_named("beta", None::<&Buffer<f64>>)
        .arg_named("variable_index", &0u32)
        .arg_named("inv_variance", &0.0f64)
        .arg(s1)
        .arg(s3)
        .build()
        .expect("Kernel s1_and_s3_sum_constant build failed.");

    let kernel_s1_and_s3_sum_parents = pro_que
        .kernel_builder(s1_and_s3_parents_name(test_rowmajor))
        .global_work_size(m)
        .arg(test_instances_buffer)
        .arg(test_leading_dimension)
        .arg_named("beta", None::<&Buffer<f64>>)
        .arg_named("variable_index", &0u32)
        .arg_named("evidence_index", None::<&Buffer<u32>>)
        .arg_named("len_evidence", &0u32)
        .arg_named("inv_variance", &0.0f64)
        .arg(s1)
        .arg(s3)
        .build()
        .expect("Kernel s1_and_s3_sum_parents build failed.");

    for i in 0..ckde.nregressions {
        let gr = &**ckde.regressions.offset(i as isize);

        if gr.nparents > 1 {
            kernel_s1_and_s3_sum_parents
                .set_arg("beta", &gr.beta)
                .unwrap();
            kernel_s1_and_s3_sum_parents
                .set_arg("variable_index", gr.variable_index)
                .unwrap();
            kernel_s1_and_s3_sum_parents
                .set_arg("evidence_index", gr.evidence_index.as_ref().unwrap())
                .unwrap();
            kernel_s1_and_s3_sum_parents
                .set_arg("len_evidence", gr.nparents - 1)
                .unwrap();
            kernel_s1_and_s3_sum_parents
                .set_arg("inv_variance", gr.variance.recip())
                .unwrap();
            kernel_s1_and_s3_sum_parents
                .enq()
                .expect("Error while executing kernel_s1_and_s3_sum_parents kernel.");
        } else {
            kernel_s1_and_s3_sum_constant
                .set_arg("beta", &gr.beta)
                .unwrap();
            kernel_s1_and_s3_sum_constant
                .set_arg("variable_index", gr.variable_index)
                .unwrap();
            kernel_s1_and_s3_sum_constant
                .set_arg("inv_variance", gr.variance.recip())
                .unwrap();
            kernel_s1_and_s3_sum_constant
                .enq()
                .expect("Error while executing kernel_s1_and_s3_sum_constant kernel.");
        }
    }
}