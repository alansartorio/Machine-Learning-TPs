use std::collections::HashMap;

pub type Class<'a> = &'a str;
pub type Attribute<'a> = &'a str;
pub type Value = u8;

pub enum Node<'a> {
    Classification(Class<'a>),
    Split {
        attribute: Attribute<'a>,
        values: HashMap<Value, Node<'a>>,
        otherwise: Class<'a>,
    },
}

impl<'a> Node<'a> {
    pub fn classify(&self, entry: HashMap<Attribute, Value>) -> Class<'a> {
        match self {
            Node::Classification(class) => class,
            Node::Split {
                attribute,
                values,
                otherwise,
            } => values
                .get(&entry[attribute])
                .map(|child| child.classify(entry))
                .unwrap_or(otherwise),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::Node;

    #[test]
    fn test_classification() {
        let tree = Node::Split {
            attribute: "A",
            values: HashMap::from([
                (0, Node::Classification("AA")),
                (1, Node::Classification("BB")),
                (
                    2,
                    Node::Split {
                        attribute: "B",
                        values: HashMap::from([
                            (0, Node::Classification("CC")),
                            (1, Node::Classification("DD")),
                            (2, Node::Classification("EE")),
                        ]),
                        otherwise: "OO",
                    },
                ),
            ]),
            otherwise: "UU",
        };

        assert_eq!(tree.classify(HashMap::from([("A", 0)])), "AA");
        assert_eq!(tree.classify(HashMap::from([("A", 1)])), "BB");
        assert_eq!(tree.classify(HashMap::from([("A", 2), ("B", 0)])), "CC");
        assert_eq!(tree.classify(HashMap::from([("A", 2), ("B", 1)])), "DD");
        assert_eq!(tree.classify(HashMap::from([("A", 2), ("B", 2)])), "EE");
        assert_eq!(tree.classify(HashMap::from([("A", 3), ("B", 2)])), "UU");
        assert_eq!(tree.classify(HashMap::from([("A", 2), ("B", 3)])), "OO");
    }
}
