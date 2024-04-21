use itertools::Itertools;
use polars::prelude::*;
use std::{collections::HashMap, sync::Arc};

use crate::decision_tree::{Node, RootData};

fn relative_freq(s: &Series) -> DataFrame {
    let mut res = s.value_counts(false, true).unwrap();

    res.with_column(
        (res.column("count")
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap()
            / s.len() as f64)
            .with_name("relative_freq"),
    )
    .unwrap();

    res
}

pub fn shannon_entropy(s: &Series) -> f64 {
    let rel = relative_freq(s);
    let freq = rel.column("relative_freq").unwrap();

    -freq.dot(&freq.log(2.0)).unwrap()
}

pub fn information_gain(df: &DataFrame, attribute: &str, output_col: &str) -> f64 {
    let rel = relative_freq(df.column(attribute).unwrap());
    let conditional_entropy = rel
        .column("relative_freq")
        .unwrap()
        .dot(
            &rel.column(attribute)
                .unwrap()
                .iter()
                .map(move |attribute_value| {
                    let attribute_value = attribute_value.try_extract::<f64>().unwrap();
                    let mask = df
                        .column(attribute)
                        .unwrap()
                        .equal(attribute_value)
                        .unwrap();
                    shannon_entropy(df.filter(&mask).unwrap().column(output_col).unwrap())
                })
                .collect(),
        )
        .unwrap();

    shannon_entropy(df.column(output_col).unwrap()) - conditional_entropy
}

fn find_highest_information_gain(df: &DataFrame, output_col: &str) -> String {
    let information_gain = |attr| information_gain(df, attr, output_col);

    df.get_columns()
        .iter()
        .map(|s| s.name())
        .filter(|attr| *attr != output_col)
        .max_by(|a, b| {
            information_gain(a)
                .partial_cmp(&information_gain(b))
                .unwrap()
        })
        .unwrap()
        .to_owned()
}

fn find_most_frequent(s: &Series) -> String {
    mode::mode(s)
        .unwrap()
        .get(0)
        .unwrap()
        .get_str()
        .unwrap()
        .to_owned()
}

fn train_inner(df: &DataFrame, output_col: &str, root_data: Arc<RootData>) -> Node {
    let attr = find_highest_information_gain(df, output_col);
    dbg!(&attr);
    let attr_values = df.column(&attr).unwrap().unique().unwrap().rechunk();
    let most_frequent = find_most_frequent(df.column(output_col).unwrap());
    let filter = |attr_value: i64| {
        let mask = df.column(&attr).unwrap().equal(attr_value).unwrap();
        df.filter(&mask).unwrap()
    };

    Node::new_split(
        root_data.clone(),
        &attr,
        HashMap::<_, _>::from_iter(attr_values.iter().map(|value| {
            let value = value.try_extract().unwrap();
            (value, {
                let filtered = filter(value);
                let unique = filtered.column(output_col).unwrap().unique().unwrap();

                match unique.len() {
                    0 => {
                        eprintln!("No rows with {attr} = {value}");
                        Node::new_classification(root_data.clone(), &most_frequent)
                    }
                    1 => {
                        let anyvalue = unique.get(0).unwrap();
                        let class = anyvalue.get_str().unwrap();
                        eprintln!("Can only be of class = {class} with {attr} = {value}");
                        Node::new_classification(root_data.clone(), class)
                    }
                    _ => {
                        eprintln!("Can be of many classes = {unique} with {attr} = {value}");
                        train_inner(&filtered, output_col, root_data.clone())
                    },
                }
            })
        })),
        &most_frequent,
    )
}

pub fn train(df: &DataFrame, output_col: &str) -> Node {
    let class_names = df
        .column(output_col)
        .unwrap()
        .unique()
        .unwrap()
        .rechunk()
        .iter()
        .map(|v| v.get_str().unwrap().to_owned())
        .collect_vec();
    let attribute_names = df
        .get_columns()
        .iter()
        .map(|c| c.name().to_owned())
        .collect_vec();

    let root_data = Arc::new(RootData {
        class_names,
        attribute_names,
    });

    train_inner(df, output_col, root_data)
}
