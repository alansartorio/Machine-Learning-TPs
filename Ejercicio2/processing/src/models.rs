use std::{collections::HashMap, fs::File, io::BufWriter, io::BufReader};
use anyhow::Result;

#[derive(Debug, Clone, Copy, serde::Deserialize)]
pub struct Record {
    #[serde(rename = "TV")]
    tv: f64,
    #[serde(rename = "Radio")]
    radio: f64,
    #[serde(rename = "Newspaper")]
    newspaper: f64,
    #[serde(rename = "Sales")]
    sales: f64,
}

impl IntoIterator for Record {
    type Item = (&'static str, f64);
    type IntoIter = std::array::IntoIter<Self::Item, 4>;

    fn into_iter(self) -> Self::IntoIter {
        [
            ("tv", self.tv),
            ("radio", self.radio),
            ("newspaper", self.newspaper),
            ("sales", self.sales),
        ]
        .into_iter()
    }
}

impl From<Record> for HashMap<&'static str, f64> {
    fn from(value: Record) -> Self {
        value.into_iter().collect()
    }
}

/*
data_analysis
{
    "interCovariance": [{
        "variables": ["TV", "Radio"],
        "covariance": 0.4
    }, {
        "first": "TV",
        "second": "Radio",
        "covariance": 0.4
    }, {
        "TV_Radio"
    }, ]
    "covarianceWithOutput": {
        "TV": 0.8,
        "Radio": 0.3,
        "Newpaper": 0.1
    }
}
*/

#[derive(Debug, serde::Serialize)]
pub struct InterCovarianceResult {
    variables: [String; 2],
    covariance: f64,
}

#[derive(Debug, serde::Serialize)]
pub struct CovarianceWithOutputResult {
    tv: f64,
    radio: f64,
    newspaper: f64,
}

#[derive(Debug, serde::Serialize)]
pub struct DataAnalysisResult {
    inter_covariance: [InterCovarianceResult; 3],
    covariance_with_output: CovarianceWithOutputResult,
}

pub enum Metric {
    MSE,
    MAE,
}


pub fn read_input_data() -> Result<Vec<Record>>
{
    let file = File::open("../Advertising.csv")?;
    let file = BufReader::new(file);

    csv::Reader::from_reader(file)
        .deserialize()
        .collect::<Result<Vec<_>, _>>()
        .map_err(anyhow::Error::msg)
}

/*
simple_regression
    {
       "radio": [
            0.20249578339243968,
            9.311638095158283,
        ],
        "tv": [
            0.04753664043301976,
            7.032593549127699,
        ],
        "newspaper": [
            0.054693098472273355,
            12.351407069278164,
        ],
    }


*/
//

pub fn write_simple_regression(result: &HashMap<&'static str, [f64; 2]>) -> Result<()> {
   write_file(result, "../data/simple_regression.json")
}

pub fn write_file<T: serde::Serialize>(result: T, filename: &'static str) -> Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &result)?;
    Ok(())
}


/*

    [
        {
            "var": "tv",
            "beta": 343,
        },
        {
            "var": "radio"
            "beta": 23
        },
        
        {
            "var": "beta_0"
            "beta": 124
        }
    ],
*/

pub type Var = &'static str;

#[derive(Debug, serde::Serialize)]
pub struct MlrCoeficient {
    pub var: Var,
    pub beta: f64,
}