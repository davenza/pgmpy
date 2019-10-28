use crate::{empty_buffers, copy_buffers, Error, GaussianKDE, gaussian_kde_free};


use std::slice;
use std::ptr;
use std::mem;

use ocl::{Buffer, ProQue};
use libc::{c_uint, c_double};

mod denominator_gaussian;
pub use denominator_gaussian::logdenominator_dataset_gaussian;

mod denominator_onlykde;
pub use denominator_onlykde::logdenominator_dataset_onlykde;

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
    marginal_precision: Buffer<f64>,
    regressions: *mut *mut GaussianRegression,
    nregressions: c_uint,
    lognorm_factor: f64
}

#[no_mangle]
pub unsafe extern "C" fn ckde_init(
    pro_que: *mut ProQue,
    kde: *mut GaussianKDE,
    precision: *mut c_double,
    regressions: *mut *mut GaussianRegression,
    nregressions: c_uint,
    lognorm_factor: f64,
    error: *mut Error,
) -> *mut CKDE {

    let d = (*kde).d;
    let mut pro_que = Box::from_raw(pro_que);

    let precision_slice = slice::from_raw_parts(precision, d * d);

    let (precision_buffer,) = copy_buffers!(pro_que, error, precision_slice => ptr::null_mut());

    let (onlykde_precision,) = if d > 1 {
        empty_buffers!(pro_que, error, f64, (d-1)*(d-1) => ptr::null_mut())
    } else {
//        Create a dummy buffer.
        empty_buffers!(pro_que, error, f64, 1 => ptr::null_mut())
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