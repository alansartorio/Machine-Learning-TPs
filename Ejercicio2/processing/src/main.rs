use anyhow::{anyhow, Result};
use statrs::statistics::Statistics;
use std::{
    collections::{HashMap, HashSet},
    iter,
};
mod models;
use itertools::Itertools;
use models::*;

fn simple_linear_regression(x: &Vec<f64>, y: &Vec<f64>) -> [f64; 2] {
    let (betas, beta_0) = multiple_linear_regression(Box::new([x]), y);

    [betas[0], beta_0]
}

fn multiple_linear_regression(variables: Box<[&Vec<f64>]>, y: &Vec<f64>) -> (Box<[f64]>, f64) {
    let n = variables.len();
    let mut betas = vec![0f64; n];
    let mut factors = vec![0f64; n];

    for (i, xi) in variables.into_iter().enumerate() {
        betas[i] = xi.covariance(y) / xi.variance();
        factors[i] = betas[i] * xi.mean();
    }

    let beta_0 = y.mean() - factors.iter().sum::<f64>();

    (betas.into_boxed_slice(), beta_0)
}

impl Metric {
    fn get(&self, real: &Vec<f64>, predicted: &Vec<f64>) -> Result<f64> {
        if (real.is_empty() || predicted.is_empty()) || real.len() != predicted.len() {
            return Err(anyhow!("The arrays must be fullfilled"));
        }

        let n = real.len();
        let mut aux: Vec<f64> = real
            .iter()
            .zip(predicted.into_iter())
            .map(|(real_elem, predicted_elem)| real_elem - predicted_elem)
            .collect();

        match self {
            Metric::MSE => aux.iter_mut().for_each(|e| *e = e.powi(2)),
            Metric::MAE => aux.iter_mut().for_each(|e| *e = e.abs()),
        }

        Ok(aux.iter().sum::<f64>() / n as f64)
    }
}

type Inputs = Box<[f64]>;
struct InputsOutput {
    inputs: Inputs,
    output: f64,
}

fn record_to_inputs_output(records: Vec<Record>, vars: Vec<Var>) -> Vec<InputsOutput> {
    records
        .into_iter()
        .map(|record| {
            let map: HashMap<_, _> = record.into();
            let inputs = vars.iter().map(|v| map[v]).collect_vec().into_boxed_slice();
            InputsOutput {
                inputs,
                output: map["sales"],
            }
        })
        .collect_vec()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum::<f64>()
}

fn rss(coef: Box<[f64]>, coef0: f64, data: &Vec<InputsOutput>) -> f64 {
    let f = |i: &[f64]| coef0 + dot(&coef, &i);

    data.iter()
        .map(|io| (io.output - f(&io.inputs)).powi(2))
        .sum()
}

fn r_squared(model_squared_sum: f64, total_squared_sum: f64) -> f64 {
    model_squared_sum / total_squared_sum
}

fn r_squared_adjusted(
    model_squared_sum: f64,
    total_squared_sum: f64,
    observarions_amount: i32,
    variables_amount: i32,
) -> f64 {
    (model_squared_sum / total_squared_sum)
        * (observarions_amount - 1 / (observarions_amount - variables_amount)) as f64
}

fn get_column(records: &Vec<Record>, column_name: &'static str) -> Vec<f64> {
    records
        .iter()
        .map(|r| {
            *HashMap::<&'static str, f64>::from(*r)
                .get(column_name)
                .unwrap()
        })
        .collect::<Vec<f64>>()
}

fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Clone,
{
    assert!(!v.is_empty());
    (0..v[0].len())
        .map(|i| v.iter().map(|inner| inner[i].clone()).collect::<Vec<T>>())
        .collect()
}

fn main() -> Result<()> {
    let records = read_input_data()?;

    // Data Analysis
    // covarianza entre variables input
    // covarianza de cada input con el output
    // let data_analysis_file = File::create("../data_analysis.json");

    // SLR
    // let res = simple_linear_regression(records.iter().map(|r| (r.tv, r.sales)).collect::<Vec<_>>());
    // dbg!(res);
    let sales: Vec<f64> = get_column(&records, "sales");

    let simple_lr: HashMap<_, _> = ["tv", "radio", "newspaper"]
        .map(|name| {
            (
                name,
                simple_linear_regression(&get_column(&records, name), &sales),
            )
        })
        .into_iter()
        .collect();

    write_simple_regression(&simple_lr)?;

    let result = mlr(records);

    write_file(result, "../data/data.json")?;

    Ok(())
}

fn mlr_rss(vars: &Vec<Var>, next_var: Var, inputs: &Vec<Record>) -> f64 {
    let mut vars = vars.clone();
    vars.push(next_var);
    let transposed = vars
        .iter()
        .map(|column_name| get_column(inputs, column_name))
        .collect_vec();
    let betas = multiple_linear_regression(
        transposed.iter().collect_vec().into_boxed_slice(),
        &get_column(inputs, "sales"),
    );
    let records = record_to_inputs_output(inputs.clone(), vars);

    rss(betas.0, betas.1, &records)
}

fn mlr(input: Vec<Record>) -> Vec<MlrCoeficient> {
    // MLR
    /*
       1. Calculas todos los modelos lineales simples
       2. Te quedás con el de menor RSS
       3. Vas agregando variables de a 1 y calculás el RSS
       4. Si te mejora el RSS, la dejás, si no la descartás
       5. Así continuas hasta que no puedas seguir mejorando el modelo agregando variables
    */
    let mut vars_left: HashSet<_> = ["tv", "radio", "newspaper"].into();
    let mut curr_vars = vec![];
    let mut current_rss = f64::INFINITY;

    while !vars_left.is_empty() {
        let (var, new_rss) = vars_left
            .iter()
            .map(|v| (*v, mlr_rss(&curr_vars, v, &input)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        if new_rss > current_rss {
            break;
        }

        current_rss = new_rss;
        vars_left.remove(var);
        curr_vars.push(var);
    }

    let transposed = curr_vars
        .iter()
        .map(|column_name| get_column(&input, column_name))
        .collect_vec();
    let betas = multiple_linear_regression(
        transposed.iter().collect_vec().into_boxed_slice(),
        &get_column(&input, "sales"),
    );
    curr_vars
        .into_iter()
        .zip(betas.0.into_iter())
        .map(|(name, value)| MlrCoeficient {
            var: name,
            beta: *value,
        })
        .chain(iter::once(MlrCoeficient {
            var: "beta_0",
            beta: betas.1,
        }))
        .collect_vec()
}
