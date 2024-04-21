use std::sync::Arc;

use polars::frame::DataFrame;

use crate::decision_tree::{Node, RootData};

pub fn train(df: &DataFrame) -> Node {
    dbg!(df);

    let root_data = Arc::new(RootData {
        class_names: vec!["A".to_string()],
        attribute_names: vec![],
    });
    Node::new_classification(root_data, "A")
}
