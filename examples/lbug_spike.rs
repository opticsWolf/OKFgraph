//! M0 go/no-go spike: open a Python-created DB with lbug 0.22.1,
//! load vector/fts, run index queries, and write a row.
use lbug::{Connection, Database, LogicalType, SystemConfig, Value};
use std::{env, error::Error};

fn dump(label: &str, rows: impl Iterator<Item = Vec<Value>>) {
    println!("-- {label} --");
    for row in rows {
        println!("{row:?}");
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).ok_or("usage: lbug_spike <db-path> <action>")?;
    let action = args.get(2).map(String::as_str).unwrap_or("inspect");
    println!(
        "lbug crate={} library={}",
        lbug::VERSION,
        lbug::get_library_source()
    );

    let db = Database::new(path, SystemConfig::default())?;
    let conn = Connection::new(&db)?;

    match action {
        "create" => {
            conn.query("CREATE NODE TABLE Spike(id STRING, text STRING, embedding FLOAT[3], PRIMARY KEY(id));")?;
            conn.query(
                "CREATE (:Spike {id: 'one', text: 'badger diet', embedding: [1.0, 0.0, 0.0]});",
            )?;
            println!("created Spike table and row");
        }
        "inspect" => {
            dump(
                "Concept",
                conn.query("CALL TABLE_INFO('Concept') RETURN *")?,
            );
        }
        "write" => {
            conn.query("CREATE (:Spike {id: 'rust-write', text: 'written from Rust', embedding: [0.0, 1.0, 0.0]});")?;
            println!("wrote spike row");
        }
        "extensions" => {
            for ext in ["vector", "fts"] {
                conn.query(&format!("INSTALL {ext};"))?;
                conn.query(&format!("LOAD {ext};"))?;
                println!("loaded {ext}");
            }
            conn.query("CALL CREATE_VECTOR_INDEX('Spike', 'spike_vec', 'embedding', mu := 30, ml := 60, metric := 'cosine', efc := 200)")?;
            conn.query("CALL CREATE_FTS_INDEX('Spike', 'spike_fts', ['text'])")?;

            let mut vec_stmt = conn.prepare(
                "CALL QUERY_VECTOR_INDEX('Spike', 'spike_vec', $vec, $k) RETURN node, distance",
            )?;
            let vec = Value::List(
                LogicalType::Float,
                vec![Value::Float(1.0), Value::Float(0.0), Value::Float(0.0)],
            );
            dump(
                "vector",
                conn.execute(&mut vec_stmt, vec![("vec", vec), ("k", Value::Int64(2))])?,
            );

            let mut fts_stmt = conn.prepare(
                "CALL QUERY_FTS_INDEX('Spike', 'spike_fts', $query, top := $k) RETURN node, score",
            )?;
            dump(
                "fts",
                conn.execute(
                    &mut fts_stmt,
                    vec![
                        ("query", Value::String("badger".into())),
                        ("k", Value::Int64(2)),
                    ],
                )?,
            );
        }
        other => return Err(format!("unknown action: {other}").into()),
    }
    Ok(())
}
