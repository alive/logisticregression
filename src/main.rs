extern crate nalgebra as na;
extern crate csv;
#[macro_use]
extern crate serde_derive;

use na::{DMatrix, DVector};
use rand::distributions::{Normal, Distribution};
use std::f64;
use csv::Reader;


// TODO change csv reading method to something more general
#[derive(Debug, Deserialize, PartialEq)]
struct Row {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64
}   

const testProportion: f64  = 0.25;

fn main() {

    println!("Reading data...");

    // data banknote authentication has 5 comma separated values
    // first 4 are inputs, 5th values is output (0 or 1) 

    // 1372 records

    let mut rdr = csv::ReaderBuilder::new().has_headers(false)
        .from_path("data_banknote_authentication.txt").expect("error opening data file");


    let mut inputVec = Vec::new(); 
    let mut outputVec = Vec::new();

    let mut testInputVec = Vec::new();
    let mut testOutputVec = Vec::new();

    let mut rows = 0;
    let testRatio = (1.0 / testProportion) as usize;
    let mut numTests = 0;

    for result in rdr.deserialize() {
        let record: Row = result.expect("error reading data");
        rows += 1;

        if rows % testRatio == 0 {
            numTests += 1;
            testInputVec.push(record.a);
            testInputVec.push(record.b);
            testInputVec.push(record.c);
            testInputVec.push(record.d);

            testOutputVec.push(record.e);

        } else {
            inputVec.push(record.a);
            inputVec.push(record.b);
            inputVec.push(record.c);
            inputVec.push(record.d);

            outputVec.push(record.e);
        }
    } 

    let cols = inputVec.len() / (rows - numTests);

    let inputs = DMatrix::from_row_slice(rows - numTests, cols, &inputVec[..]);
    let outputs = DMatrix::from_row_slice(rows - numTests, 1, &outputVec[..]);
    let testInputs = DMatrix::from_row_slice(numTests, cols, &testInputVec[..]);
    let testOutputs = DMatrix::from_row_slice(numTests, 1, &testOutputVec[..]);

    // println!("{:?}", inputs);
    // println!("{:?}", outputs);


    let theta_vector = fit(inputs, outputs);
    let predicted = predict(testInputs, &theta_vector);
    let accuracy = accuracy(&predicted, &testOutputs);
    println!("Accuracy: {}", accuracy);

}

const max_epochs: i32 = 5000;
const learning_rate: f64 = 0.1;

fn fit(x_: DMatrix<f64>, y_true: DMatrix<f64>) -> DVector<f64> {

    let x = add_bias_feature(x_);

    println!("{}", x);

    let training_set_size = x.nrows();
    let amount_of_features = x.ncols();

    let mut theta_vector = random_theta_guess(amount_of_features);
    for epoch in 0..max_epochs {

        let h = (x.clone() * theta_vector.clone()).map(sigmoid);
        let cost = (y_true.clone().transpose() * h.map(f64::ln) - 
            y_true.clone().scale(-1.0).add_scalar(1.0).transpose() *
            h.clone().scale(-1.0).add_scalar(1.0).map(f64::ln)) 
            .scale(-1.0 / training_set_size as f64);
        let temp = (x.clone().transpose() * (h - y_true.clone()))
        .scale(learning_rate / training_set_size as f64);
        // println!("{}", temp);
        theta_vector = theta_vector.clone() - temp;

        println!("{}|{}", epoch, cost.get(0).expect("aa"));

    }
    // println!("final theta vector: {}", theta_vector);
    // println!("decision boundary:");
    theta_vector
}

fn accuracy(y_pred: &DVector<f64>, y_true: &DMatrix<f64>) -> f64 {
    let rows = y_pred.nrows();
    let mut sum = 0;
    for i in 0..rows {
        if y_pred.get(i).expect("bad predicted vector") == y_true.get(i).expect("bad true vector") {
            sum += 1;
        }
    }
    sum as f64 / rows as f64 * 100.0
}

fn predict(x: DMatrix<f64>, theta: &DVector<f64>) -> DVector<f64> {
    let h = (add_bias_feature(x) * theta).map(sigmoid);
    println!("{:?}", h.clone());
    h.map(f64::round)
}

fn add_bias_feature(data: DMatrix<f64>) -> DMatrix<f64> {
    // adds the column of 1s as the bias feature
    data.insert_column(0, 1.0)
}

fn random_theta_guess(size: usize) -> DVector<f64> {
    let normal = Normal::new(0.0, 1.0);
    let theta = DVector::from_fn(size, |i, _| normal.sample(&mut rand::thread_rng()));
    theta
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + f64::consts::E.powf(-z))
}