use crate::{copy_buffers, empty_buffers};

use crate::{GaussianKDE, Error, DoubleNumpyArray, get_max_work_size, max_gpu_vec_copy,
            log_sum_gpu_vec, buffer_fill_value, max_gpu_mat, sum_gpu_mat};

use ocl::{Buffer, ProQue};
use libc::{c_double};

use std::slice;
use std::f64;

pub struct GaussianRegression {
    variable_index: c_int,
    beta: Buffer<f64>,
    evidence_index: Buffer<u32>,
    nparents: c_int,
    variance: c_double
}

#[no_mangle]
pub unsafe extern "C" fn gaussian_regression_init(pro_que: *mut ProQue,
                                                  variable_index: c_int,
                                                  beta: *mut c_double,
                                                  evidence_index: *mut c_int,
                                                  nparents: c_int,
                                                  variance: c_double,
                                                  error: *mut Error) -> *mut GaussianRegression {
    println!("Init GaussianRegression");

    let mut pro_que = Box::from_raw(pro_que);

    let beta_slice = slice::from_raw_parts(*beta, nparents+1);
    let beta_buffer = copy_buffers!(pro_que, error, beta_slice => ptr::null_mut());


    let evidence_index_buffer =
        if nparents > 1 {
            let evidence_index_slice = slice::from_raw_parts(*evidence_index, nparents-1);
            copy_buffers!(pro_que, error, evidence_index_slice => ptr::null_mut());
        } else {
            empty_buffers!(pro_que, error, u32, 0);
        };


    let gr = Box::new(GaussianRegression {
        variable_index: variable_index
        beta: beta_buffer,
        evidence_index: evidence_index_buffer,
        nparents: nparents,
        variance: variance
    });

    let ptr_gr = Box::into_raw(kde);

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
    regressions: *mut *mut GaussianRegression,
    nregressions: c_int,
}


#[no_mangle]
pub unsafe extern "C" fn ckde_init(pro_que: *mut ProQue,
                                   kde: *mut *GaussianKDE,
                                   precision: *mut c_double,
                                   regressions: *mut *mut GaussianRegression,
                                   nregressions: c_int,
                                   error: *mut Error) -> *mut CKDE {
    println!("Init CKDE");

    let d = (*kde).d;

    let precision_slice = slice::from_raw_parts(*precision, d*d);

    let precision_buffer = copy_buffers!(pro_que, error, precision_slice => ptr::null_mut());

    let ckde = Box::new(CKDE {
        kde,
        precision: precision_buffer,
        regressions,
        nregressions
    });

    let ptr_ckde = Box::into_raw(ckde);
    ptr_ckde
}

#[no_mangle]
pub unsafe extern "C" fn ckde_free(ckde: *mut CKDE) {
    println!("Free CKDE");
    if ckde.is_null() {
        return;
    }
    unsafe {
        Box::from_raw(ckde);
    }
}
