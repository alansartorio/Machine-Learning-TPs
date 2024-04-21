use std::collections::{HashMap, HashSet};

use crate::decision_tree::{Attribute, Class, Node, Value};

pub fn train<'a>(
    classes: HashSet<Class<'a>>,
    attribute_values: HashMap<Attribute<'a>, Value>,
) -> Node<'a> {
    todo!();
}
