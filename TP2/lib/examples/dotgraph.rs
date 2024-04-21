use std::{collections::HashMap, rc::Rc};

use tp2::decision_tree::{Node, RootData};

fn main() {
    let root_data = Rc::new(RootData {
        class_names: ["AA", "BB", "CC", "DD", "EE", "OO", "UU"]
            .map(ToOwned::to_owned)
            .to_vec(),
        attribute_names: ["A", "B"].map(ToOwned::to_owned).to_vec(),
    });

    let tree = Node::new_split(
        root_data.clone(),
        "A",
        HashMap::from([
            (0, Node::new_classification(root_data.clone(), "AA")),
            (1, Node::new_classification(root_data.clone(), "BB")),
            (
                2,
                Node::new_split(
                    root_data.clone(),
                    "B",
                    HashMap::from([
                        (0, Node::new_classification(root_data.clone(), "CC")),
                        (1, Node::new_classification(root_data.clone(), "DD")),
                        (2, Node::new_classification(root_data.clone(), "EE")),
                    ]),
                    "OO",
                ),
            ),
        ]),
        "UU",
    );

    println!("{}", tree.to_graphviz());
}
