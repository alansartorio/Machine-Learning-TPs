use itertools::Itertools;
use std::{collections::HashMap, fmt::Debug, rc::Rc, sync::Arc};

pub type Class = usize;
pub type Attribute = usize;
pub type Value = i64;

pub enum NodeType {
    Classification(Class),
    Split {
        attribute: Attribute,
        values: HashMap<Value, Node>,
        otherwise: Class,
    },
}

pub struct RootData {
    pub class_names: Vec<String>,
    pub attribute_names: Vec<String>,
}

impl RootData {
    fn get_class_index(&self, name: &str) -> Class {
        self.class_names.iter().position(|n| n == name).unwrap()
    }

    fn get_attribute_index(&self, name: &str) -> Attribute {
        self.attribute_names.iter().position(|n| n == name).unwrap()
    }
}

pub struct Node {
    root_data: Arc<RootData>,
    variant: NodeType,
}

impl Node {
    pub fn new_classification(root_data: Arc<RootData>, class_name: &str) -> Node {
        let class = root_data.get_class_index(class_name);
        Node {
            root_data,
            variant: NodeType::Classification(class),
        }
    }

    pub fn new_split(
        root_data: Arc<RootData>,
        attribute_name: &str,
        values: HashMap<Value, Node>,
        otherwise_class_name: &str,
    ) -> Node {
        let attribute = root_data.get_attribute_index(attribute_name);
        let otherwise = root_data.get_class_index(otherwise_class_name);
        Node {
            root_data,
            variant: NodeType::Split {
                attribute,
                values,
                otherwise,
            },
        }
    }

    pub fn classify(&self, entry: HashMap<Attribute, Value>) -> Class {
        match &self.variant {
            NodeType::Classification(class) => *class,
            NodeType::Split {
                attribute,
                values,
                otherwise,
            } => values
                .get(&entry[&attribute])
                .map(|child| child.classify(entry))
                .unwrap_or(*otherwise),
        }
    }

    pub fn classify_with_names(&self, entry: HashMap<&str, Value>) -> String {
        let entry = HashMap::from_iter(entry.into_iter().map(|(attribute_name, value)| {
            (self.root_data.get_attribute_index(attribute_name), value)
        }));
        self.root_data.class_names[self.classify(entry)].to_string()
    }

    pub fn to_graphviz_inner(&self, statements: &mut Vec<String>, counter: &mut usize) -> String {
        let classes = &self.root_data.class_names;
        let attributes = &self.root_data.attribute_names;

        *counter += 1;
        let self_name = format!("n{counter:03}");
        let self_label = match &self.variant {
            NodeType::Classification(class) => &classes[*class],
            NodeType::Split { attribute, .. } => &attributes[*attribute],
        };
        statements.push(format!("{self_name} [label=\"{self_label}\"]"));

        match &self.variant {
            NodeType::Classification(_) => (),
            NodeType::Split {
                attribute, values, ..
            } => {
                for (value, node) in values.iter() {
                    *counter += 1;
                    let variant_name = format!("n{counter:03}");
                    let variant_label = format!("{} = {value}", attributes[*attribute]);
                    statements.push(format!("{variant_name} [label=\"{variant_label}\"]"));
                    statements.push(format!("{self_name} -> {variant_name}"));
                    let child = node.to_graphviz_inner(statements, counter);
                    statements.push(format!("{variant_name} -> {child}"));
                }
            }
        }

        self_name
    }

    pub fn to_graphviz(&self) -> String {
        let mut statements = vec![];
        let mut counter = 0;
        self.to_graphviz_inner(&mut statements, &mut counter);
        format!("digraph {{rankdir=LR;\n{}}}", statements.join(";\n"))
    }
}

impl Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let classes = &self.root_data.class_names;
        let attributes = &self.root_data.attribute_names;

        match &self.variant {
            NodeType::Classification(class) => f.write_fmt(format_args!("{}", classes[*class])),
            NodeType::Split {
                attribute, values, ..
            } => {
                let attribute = &attributes[*attribute];
                f.write_fmt(format_args!(
                    "{} -> {{{}}}",
                    attribute,
                    values
                        .iter()
                        .map(|(value, node)| format!(
                            "\"{attribute} = {value}\" -> {{\n{node:?}\n}}"
                        ))
                        .join(";\n")
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use crate::decision_tree::RootData;

    use super::Node;

    #[test]
    fn test_classification() {
        let root_data = Arc::new(RootData {
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

        assert_eq!(tree.classify_with_names(HashMap::from([("A", 0)])), "AA");
        assert_eq!(tree.classify_with_names(HashMap::from([("A", 1)])), "BB");
        assert_eq!(
            tree.classify_with_names(HashMap::from([("A", 2), ("B", 0)])),
            "CC"
        );
        assert_eq!(
            tree.classify_with_names(HashMap::from([("A", 2), ("B", 1)])),
            "DD"
        );
        assert_eq!(
            tree.classify_with_names(HashMap::from([("A", 2), ("B", 2)])),
            "EE"
        );
        assert_eq!(
            tree.classify_with_names(HashMap::from([("A", 3), ("B", 2)])),
            "UU"
        );
        assert_eq!(
            tree.classify_with_names(HashMap::from([("A", 2), ("B", 3)])),
            "OO"
        );

        dbg!(tree);

        panic!();
    }
}
