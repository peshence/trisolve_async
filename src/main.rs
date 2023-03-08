// mod async_test;

use ndarray::{Array2, Array1, ArrayBase,s, Data, prelude::*, DataMut};
// use async_mutex::Mutex;
// use async_dag::Graph;
// use futures::prelude::*;
// use tokio::prelude::*;
// use tokio::task;

// #[tokio::main]
fn main() {

}

/// assume L is upper triangular square matrix
async fn simple_triangular_solve<S1,S2>(L: &ArrayBase<S1, Ix2>, b: &mut ArrayBase<S2, Ix1>) -> ()
        where S1: Data<Elem=f64>, S2 : DataMut<Elem=f64> {
    for col in (0..L.dim().0).rev() {
        b[col] = b[col]/L[[col,col]];
        for row in (0..col).rev() {
            b[row] -= L[[row,col]]*b[col]
        }
    }
}

/// in a blocked upper triangular solver this handles the blocks above a diagonal block (which has been solved)
/// this needs the whole b vector and needs it to be mutex. Alternatively
async fn solve_above<S1>(L: &ArrayBase<S1, Ix2>, solved: &ArrayBase<S1, Ix1>)  -> Array1<f64> 
        where S1: Data<Elem=f64> {
    let mut b = Array::<f64,Ix1>::zeros(L.raw_dim()[0]);
    for col in (0..L.dim().0).rev() {        
        for row in (0..L.dim().0).rev() {
            b[row] -= L[[row,col]]*solved[col];
        }
    }
    b
}

/// Solve upper triangular matrix equation Lx = b by splitting L(nxn) into blocks of size mxm
// TODO: handle n not divisible by m
async fn blocked_triangular_solve(L: &Array2<f64>, b: &mut Array1<f64>, m:i32) -> () {
    let n = L.dim().0 as i32;
    let mut left_col_bound = n-m;//n-i*m;
    let mut right_col_bound = n;//n-(i+1)*m;
    for i in 0..(n/m) { // row cells
        let mut top_row_bound = left_col_bound;//n-(j+1)*m;
        let mut bottom_row_bound = right_col_bound;//n-j*m;
        for j in i..(n/m) { // col cells
            if i==j{
                simple_triangular_solve(&L.slice(s![top_row_bound..bottom_row_bound,
                                                    left_col_bound..right_col_bound]), 
                                        &mut b.slice_mut(s![top_row_bound..bottom_row_bound])).await;
            }
            else {
                // let mut a = b.slice_mut(s![lower_row_bound..upper_row_bound]);
                let temp = b.slice(s![top_row_bound..bottom_row_bound]);
                let sub_vec = solve_above(&L.slice(s![top_row_bound..bottom_row_bound,left_col_bound..right_col_bound]),
                                &b.slice(s![left_col_bound..right_col_bound])).await;
                let bminsub = &temp + &sub_vec;
                b.slice_mut(s![top_row_bound..bottom_row_bound]).assign(&bminsub);
            }
            top_row_bound -= m;
            bottom_row_bound -= m;
        }
        left_col_bound -= m;
        right_col_bound -= m;
    }
}


#[cfg(test)]
mod tests {
    use ndarray::{arr1,arr2};
    use test_case::test_case;

    use super::*;

    #[test_case(arr2(&[
        [1.,0.,0.],
        [0.,1.,0.],
        [0.,0.,1.]]),arr1(&[2.,5.,4.]),arr1(&[2.,5.,4.]) ; "it runs")]
    #[test_case(arr2(&[
        [1.,2.,2.],
        [0.,3.,1.],
        [0.,0.,7.]]),arr1(&[8.,5.,14.]),arr1(&[2.,1.,2.]) ; "it solves")]
    async fn test_simple_trisolve(L:Array2<f64>, mut b:Array1<f64>, solution:Array1<f64>){
        simple_triangular_solve(&L, &mut b).await;
        assert_eq!(b,solution);
    }

    #[test_case(arr2(&[
        [1.,0.,0.,0.],
        [0.,1.,0.,0.],
        [0.,0.,1.,0.],
        [0.,0.,0.,1.]]),arr1(&[2.,5.,4.,8.]),2,arr1(&[2.,5.,4.,8.]) ; "it runs")]
    #[test_case(arr2(&[
        [1.,2.,2.,0.],
        [0.,3.,1.,0.],
        [0.,0.,7.,0.],
        [0.,0.,0.,1.]]),arr1(&[8.,5.,14.,8.]),2,arr1(&[2.,1.,2.,8.]) ; "it solves")]
    async fn test_blocked_trisolve(L:Array2<f64>, mut b:Array1<f64>, m: i32, solution:Array1<f64>){
        blocked_triangular_solve(&L, &mut b, m).await;
        assert_eq!(b,solution);
    }

    #[test_case(arr2(&[
        [1.,2.,2.],
        [6.,3.,1.],
        [4.,10.,7.]]),arr1(&[8.,5.,14.]),arr1(&[2.,1.,2.]),arr1(&[0.,-12.,-18.]) ; "it solves")]
    async fn test_solve_above(L:Array2<f64>, mut b:Array1<f64>, solved: Array1<f64>, solution:Array1<f64>) -> ()
    {
        let test_solution = solve_above(&L, &solved).await;
        assert_eq!(b+test_solution,solution);
    }    
    // #[test_case(arr2(&[
    //     [1.,2.,2.],
    //     [6.,3.,1.],
    //     [4.,10.,7.]]),arr1(&[8.,5.,14.]),arr1(&[2.,1.,2.]),arr1(&[0.,-12.,-18.]) ; "it solves")]
    // fn test_solve_above_slice(L:Array2<f64>, mut b:Array1<f64>, solved: Array1<f64>, solution:Array1<f64>) -> ()
    // {
    //     solve_above(&L, &mut b, solved);
    //     assert_eq!(b,solution);
    // }
}