//! `clap` command tree generated from `SETTINGS`, plus renderers.
use crate::error::{Envelope, OkfError, Result};
use crate::ops::Router;
use crate::settings::{row, SettingKind, Settings, SETTINGS};
use clap::{Arg, ArgAction, ArgMatches, Command, Id};
use serde_json::{Map, Value};

/// Global settings flags shared by `okf` and `okf-mcp`.
///
/// Both binaries must resolve the same `--db-path`/`--bundle-root` grammar;
/// keeping one generator avoids the drift the Python CLI epilog warns about.
pub fn with_global_args(cmd: Command) -> Command {
    let mut cmd = cmd;
    for def in SETTINGS {
        if !def.cli {
            continue;
        }
        let dest = leak(def.dest_name());
        let long = leak(def.flag_name().trim_start_matches('-').to_string());
        let mut arg = Arg::new(Id::from(dest))
            .long(long)
            .global(true)
            .help(def.help);
        match def.kind {
            SettingKind::Bool => {
                arg = arg.action(ArgAction::SetTrue);
            }
            SettingKind::Int => {
                arg = arg.value_parser(clap::value_parser!(i64));
            }
            SettingKind::Float => {
                arg = arg.value_parser(clap::value_parser!(f64));
            }
            SettingKind::Str | SettingKind::Csv => {
                arg = arg.value_name(match def.kind {
                    SettingKind::Csv => "A,B",
                    _ => "VALUE",
                });
            }
        }
        cmd = cmd.arg(arg);
    }
    cmd.arg(
        Arg::new("root")
            .long("root")
            .global(true)
            .action(ArgAction::Append)
            .value_name("ALIAS=PATH")
            .help("Named bundle root (repeatable)"),
    )
}

pub fn command() -> Command {
    with_global_args(
        Command::new("okf")
            .version(super::version())
            .about("OKFgraph Rust core")
            .arg(
                Arg::new("json")
                    .long("json")
                    .global(true)
                    .action(ArgAction::SetTrue)
                    .help("Emit the uniform JSON envelope"),
            ),
    )
    .subcommand(Command::new("init").about("Initialize database and schema"))
    .subcommand(
        Command::new("read")
            .about("Read a concept")
            .arg(Arg::new("concept_id").required(true))
            .arg(
                Arg::new("include")
                    .long("include")
                    .default_value("body")
                    .value_parser(["body", "chunks", "document", "context"]),
            ),
    )
    .subcommand(
        Command::new("traverse")
            .about("Traverse relationships")
            .arg(Arg::new("start_id").required(false))
            .arg(
                Arg::new("relationship")
                    .long("relationship")
                    .default_value("CONTAINS"),
            )
            .arg(
                Arg::new("direction")
                    .long("direction")
                    .default_value("OUTGOING")
                    .value_parser(["OUTGOING", "INCOMING", "BOTH"]),
            )
            .arg(
                Arg::new("depth")
                    .long("depth")
                    .value_parser(clap::value_parser!(i64))
                    .default_value("1"),
            )
            .arg(Arg::new("target").long("target")),
    )
    .subcommand(
        Command::new("images")
            .about("List image assets attached to a concept")
            .arg(Arg::new("concept_id").required(true)),
    )
    .subcommand(
        Command::new("image")
            .about("Fetch one image asset")
            .arg(Arg::new("asset_id").required(true)),
    )
    .subcommand(Command::new("doctor").about("Health scan"))
    .subcommand(Command::new("model-info").about("Show configured model lanes"))
    .subcommand(Command::new("pin-info").about("Show graph model pins"))
    .subcommand(
        Command::new("list-directory")
            .about("List directory children")
            .arg(Arg::new("directory_id").required(false)),
    )
    .subcommand(
        Command::new("serve")
            .about("Own the database and serve the HTTP op plane")
            .arg(
                Arg::new("listen")
                    .long("listen")
                    .default_value("127.0.0.1:0")
                    .help("TCP address to bind (host:port; port 0 allocates a free port)"),
            ),
    )
}

pub fn run<I, T>(args: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let matches = match command().try_get_matches_from(args) {
        Ok(m) => m,
        Err(err) => {
            let code = if err.use_stderr() { 2 } else { 0 };
            err.print().expect("clap output");
            return code;
        }
    };
    match execute(&matches) {
        Ok((op, data)) => {
            if matches.get_flag("json") {
                println!(
                    "{}",
                    Envelope::success(op, data).to_json_pretty().expect("json")
                );
            } else {
                render(data);
            }
            0
        }
        Err(err) => {
            if matches.get_flag("json") {
                println!(
                    "{}",
                    Envelope::failure("cli", &err)
                        .to_json_pretty()
                        .expect("json")
                );
            } else {
                eprintln!("[ERROR] {}", err.titles());
            }
            err.exit_code()
        }
    }
}

fn execute(matches: &ArgMatches) -> Result<(&'static str, Value)> {
    let settings = settings_from(matches)?;
    let (name, sub) = matches
        .subcommand()
        .ok_or_else(|| OkfError::new("BAD_VALUE", "a command is required"))?;
    if name == "init" {
        let warnings = Router::init(&settings.db_path, settings.embedding_dim)?;
        return Ok((
            "init",
            serde_json::json!({"db_path": settings.db_path, "warnings": warnings}),
        ));
    }
    if name == "serve" {
        let listen = sub
            .get_one::<String>("listen")
            .map(String::as_str)
            .unwrap_or("127.0.0.1:0");
        let discovery = crate::server::serve(&settings, listen)?;
        return Ok((
            "serve",
            serde_json::json!({
                "discovery": discovery.to_string_lossy(),
                "db_path": settings.db_path,
            }),
        ));
    }
    let router = Router::open(&settings)?;
    match name {
        "read" => {
            let mut params = Map::new();
            params.insert(
                "concept_id".into(),
                Value::String(
                    sub.get_one::<String>("concept_id")
                        .expect("required")
                        .clone(),
                ),
            );
            params.insert(
                "include".into(),
                Value::String(sub.get_one::<String>("include").expect("default").clone()),
            );
            Ok(("read", router.call("read", &params)?))
        }
        "traverse" => {
            let mut params = Map::new();
            insert_str(&mut params, sub, "start_id");
            insert_str(&mut params, sub, "relationship");
            insert_str(&mut params, sub, "direction");
            insert_str(&mut params, sub, "target");
            if let Some(depth) = sub.get_one::<i64>("depth") {
                params.insert("depth".into(), Value::from(*depth));
            }
            Ok(("traverse", router.call("traverse", &params)?))
        }
        "images" => {
            let mut params = Map::new();
            params.insert(
                "concept_id".into(),
                Value::String(
                    sub.get_one::<String>("concept_id")
                        .expect("required")
                        .clone(),
                ),
            );
            Ok(("list_images", router.call("list_images", &params)?))
        }
        "image" => {
            let mut params = Map::new();
            params.insert(
                "asset_id".into(),
                Value::String(sub.get_one::<String>("asset_id").expect("required").clone()),
            );
            Ok(("get_image", router.call("get_image", &params)?))
        }
        "doctor" => Ok(("doctor", router.call("doctor", &Map::new())?)),
        "pin-info" => Ok(("pin_info", router.call("pin_info", &Map::new())?)),
        "list-directory" => {
            let mut params = Map::new();
            insert_str(&mut params, sub, "directory_id");
            Ok(("list_directory", router.call("list_directory", &params)?))
        }
        "model-info" => Ok((
            "model_info",
            serde_json::json!({
                "model_id": settings.model_id,
                "embedding_dim": settings.embedding_dim,
                "rerank_model_id": settings.rerank_model_id,
                "image_model_id": settings.image_model_id,
            }),
        )),
        _ => Err(OkfError::new(
            "BAD_VALUE",
            format!("unknown command: {name}"),
        )),
    }
}

fn leak(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn insert_str(params: &mut Map<String, Value>, matches: &ArgMatches, name: &str) {
    if let Some(value) = matches.get_one::<String>(name) {
        params.insert(name.into(), Value::String(value.clone()));
    }
}

pub fn settings_from(matches: &ArgMatches) -> Result<Settings> {
    let mut map = Map::new();
    for def in SETTINGS {
        if !def.cli {
            continue;
        }
        let dest = def.dest_name();
        let value = match def.kind {
            SettingKind::Bool => {
                if !matches.get_flag(&dest) {
                    continue;
                }
                Value::Bool(!def.invert)
            }
            SettingKind::Int => match matches.get_one::<i64>(&dest) {
                Some(v) => Value::from(*v),
                None => continue,
            },
            SettingKind::Float => match matches.get_one::<f64>(&dest) {
                Some(v) => serde_json::json!(v),
                None => continue,
            },
            SettingKind::Str | SettingKind::Csv => match matches.get_one::<String>(&dest) {
                Some(v) => Value::String(v.clone()),
                None => continue,
            },
        };
        let name = row(def.name).expect("table row").name;
        map.insert(name.to_string(), value);
    }
    if let Some(roots) = matches.get_many::<String>("root") {
        map.insert(
            "roots".into(),
            Value::Array(roots.cloned().map(Value::String).collect()),
        );
    }
    Settings::from_map(&map)
}

fn render(data: Value) {
    match data {
        Value::Array(items) => {
            for item in items {
                println!("{}", serde_json::to_string(&item).expect("json"));
            }
        }
        Value::Object(_) => println!("{}", serde_json::to_string_pretty(&data).expect("json")),
        Value::Null => {}
        other => println!("{other}"),
    }
}
