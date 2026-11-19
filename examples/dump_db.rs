//! Dump a v10 graph as sorted JSON for DB-parity comparisons.
use okfgraph::settings::Settings;
use okfgraph::store::Store;
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: dump_db <db-path>")?;
    let settings = Settings::default().with("db_path", Value::String(path))?;
    let store = Store::open(&settings)?;
    println!("{}", serde_json::to_string_pretty(&store.dump_all()?)?);
    Ok(())
}
