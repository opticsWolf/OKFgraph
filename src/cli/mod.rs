//! `clap` command tree generated from `SETTINGS`, plus renderers.
//!
//! The 29 verbs mirror `okfgraph/cli.py`; global settings flags come from the
//! generated table so the Rust and Python CLIs accept the same configuration
//! grammar. All business logic stays in `ops`/`diagnostics`/`producers`.

mod local;

use crate::error::{Envelope, OkfError, Result};
use crate::ops::{OpClient, Router};
use crate::server::client::OpHttpClient;
use crate::settings::{row, SettingKind, Settings, SETTINGS};
use clap::{Arg, ArgAction, ArgMatches, Command, Id};
use serde_json::{json, Map, Value};
use std::ffi::OsString;
use std::time::Duration;

/// All 29 public CLI verbs (the oracle's `_COMMANDS` keys).
pub const COMMANDS: &[&str] = &[
    "init",
    "model-info",
    "import",
    "ingest",
    "search",
    "read",
    "traverse",
    "export",
    "images",
    "image",
    "shell",
    "broken-links",
    "repair-links",
    "retitle-thoughts",
    "diff",
    "doctor",
    "lint",
    "produce",
    "reindex",
    "reembed",
    "deleted-list",
    "deleted-recover",
    "delete",
    "deleted-purge",
    "detach",
    "update",
    "supersede",
    "serve",
    "setup",
];

/// Global settings flags shared by `okf` and `okf-mcp`.
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
            SettingKind::Bool => arg = arg.action(ArgAction::SetTrue),
            SettingKind::Int => arg = arg.value_parser(clap::value_parser!(i64)),
            SettingKind::Float => arg = arg.value_parser(clap::value_parser!(f64)),
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

fn flag(name: &'static str, short: Option<char>, help: &'static str) -> Arg {
    let mut arg = Arg::new(name)
        .long(name)
        .global(true)
        .action(ArgAction::SetTrue)
        .help(help);
    if let Some(short) = short {
        arg = arg.short(short);
    }
    arg
}

pub fn command() -> Command {
    let cmd = with_global_args(
        Command::new("okf")
            .version(super::version())
            .about("OKFgraph Rust core")
            .arg(flag("json", None, "Emit the uniform JSON envelope"))
            .arg(flag("verbose", Some('v'), "Enable debug logging"))
            .arg(flag("quiet", Some('q'), "Suppress logging except errors"))
            .arg(
                Arg::new("log-file")
                    .long("log-file")
                    .global(true)
                    .default_value("")
                    .help("Write logs to a rotating file"),
            )
            .arg(flag("profile", None, "Profile the invocation")),
    );
    let mut cmd = cmd;
    for (name, help, args) in command_specs() {
        let mut sub = Command::new(name).about(help);
        for arg in args {
            sub = sub.arg(arg);
        }
        cmd = cmd.subcommand(sub);
    }
    cmd
}

type Spec = (&'static str, &'static str, Vec<Arg>);

fn text(name: &'static str) -> Arg {
    Arg::new(name).long(name).value_name("VALUE")
}

fn number(name: &'static str) -> Arg {
    Arg::new(name)
        .long(name)
        .value_parser(clap::value_parser!(i64))
}

fn decimal(name: &'static str) -> Arg {
    Arg::new(name)
        .long(name)
        .value_parser(clap::value_parser!(f64))
}

fn toggle(name: &'static str) -> Arg {
    Arg::new(name).long(name).action(ArgAction::SetTrue)
}

fn choices(name: &'static str, values: &'static [&'static str], default: &'static str) -> Arg {
    Arg::new(name)
        .long(name)
        .value_parser(clap::builder::PossibleValuesParser::new(
            values.iter().copied(),
        ))
        .default_value(default)
}

fn repeat(name: &'static str, value_name: &'static str) -> Arg {
    Arg::new(name)
        .long(name)
        .value_name(value_name)
        .action(ArgAction::Append)
}

fn command_specs() -> Vec<Spec> {
    vec![
        ("init", "Initialize database and schema", vec![]),
        (
            "model-info",
            "Show model cache status",
            vec![toggle("converters")],
        ),
        (
            "import",
            "Import OKF files (delta-aware)",
            vec![
                Arg::new("files").num_args(0..),
                toggle("all"),
                text("bundle-path"),
                number("batch-size"),
                choices("mode", &["text", "optional", "omni"], "text"),
                toggle("prune-missing"),
                toggle("force"),
                choices("rebuild", &["auto", "defer", "force"], "auto"),
            ],
        ),
        (
            "search",
            "Search concepts, chunks or images",
            vec![
                Arg::new("query").required(true),
                choices("target", &["concepts", "chunks", "images"], "concepts"),
                Arg::new("limit")
                    .long("limit")
                    .value_parser(clap::value_parser!(i64))
                    .default_value("10"),
                text("concept-type"),
                text("tags"),
                text("parent-id"),
                toggle("include-chunks"),
                Arg::new("max-chunks-per-doc")
                    .long("max-chunks-per-doc")
                    .value_parser(clap::value_parser!(i64))
                    .default_value("3"),
                toggle("expand"),
                Arg::new("context-hops")
                    .long("context-hops")
                    .value_parser(clap::value_parser!(i64))
                    .default_value("1"),
                toggle("hub-rerank"),
                Arg::new("hub-weight")
                    .long("hub-weight")
                    .value_parser(clap::value_parser!(f64))
                    .default_value("0.3"),
                choices("rank", &["none", "hub", "ppr", "cross"], "none"),
                number("efs"),
                toggle("dedupe"),
                toggle("rerank"),
                repeat("where", "KEY=VALUE"),
                text("since"),
                text("until"),
                text("exclude-tags"),
                text("link-kinds"),
                text("near"),
                number("hops"),
                Arg::new("offset")
                    .long("offset")
                    .value_parser(clap::value_parser!(i64))
                    .default_value("0"),
                decimal("recency"),
                decimal("recency-weight"),
            ],
        ),
        (
            "read",
            "Read a concept",
            vec![
                Arg::new("concept_id").required(true),
                choices(
                    "include",
                    &["body", "chunks", "document", "context"],
                    "body",
                ),
                text("output-path"),
                number("max-tokens"),
            ],
        ),
        (
            "traverse",
            "Traverse relationships",
            vec![
                Arg::new("start_id").default_value(""),
                choices(
                    "relationship",
                    &["CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET"],
                    "CONTAINS",
                ),
                choices("direction", &["OUTGOING", "INCOMING", "BOTH"], "OUTGOING"),
                Arg::new("depth")
                    .long("depth")
                    .value_parser(clap::value_parser!(i64))
                    .default_value("1"),
                text("node-type"),
                text("target"),
                Arg::new("max-path-length")
                    .long("max-path-length")
                    .value_parser(clap::value_parser!(i64))
                    .default_value("6"),
                text("link-kinds"),
            ],
        ),
        (
            "ingest",
            "Add content: markdown, PDF/Office, thoughts or BibTeX",
            vec![
                Arg::new("kind")
                    .long("kind")
                    .required(true)
                    .value_parser(["md", "pdf", "thoughts", "bib"]),
                text("md-path"),
                text("bib-path"),
                Arg::new("pdf-path").long("pdf-path").num_args(1..),
                text("thoughts"),
                text("topic"),
                text("concept-id"),
                text("title"),
                text("description"),
                text("tags"),
                Arg::new("auto-import")
                    .long("no-auto-import")
                    .action(ArgAction::SetFalse)
                    .default_value("true"),
                text("output-dir"),
                choices(
                    "routing-mode",
                    &["auto", "surgical", "always", "never"],
                    "auto",
                ),
                Arg::new("mode")
                    .long("mode")
                    .value_parser(["text", "optional", "omni"]),
                number("batch-size"),
                toggle("prune-missing"),
                toggle("force"),
                Arg::new("extract-images")
                    .long("no-extract-images")
                    .action(ArgAction::SetFalse)
                    .default_value("true"),
                text("type"),
                repeat("extra", "KEY=VALUE"),
                toggle("materialize"),
                repeat("related", "ID"),
                text("folder"),
            ],
        ),
        (
            "export",
            "Export the bundle or one concept",
            vec![
                toggle("all"),
                Arg::new("output-dir").long("output-dir").required(true),
                text("concept-id"),
                text("concept-type"),
                text("tags"),
                text("directory-id"),
                choices("flavor", &["okf", "obsidian"], "okf"),
            ],
        ),
        (
            "diff",
            "Structural diff",
            vec![Arg::new("old"), Arg::new("new")],
        ),
        (
            "doctor",
            "Health scan with optional safe fix",
            vec![
                toggle("strict"),
                toggle("fix"),
                Arg::new("stale-days")
                    .long("stale-days")
                    .value_parser(clap::value_parser!(i64))
                    .default_value("365"),
            ],
        ),
        (
            "lint",
            "Validate bundle frontmatter and links",
            vec![Arg::new("dir")],
        ),
        (
            "produce",
            "Generate a bundle from a data source",
            vec![
                Arg::new("from_").long("from").required(true),
                text("source"),
                toggle("sync"),
                text("output-dir"),
                text("prefix"),
                toggle("overwrite"),
            ],
        ),
        ("shell", "Interactive REPL", vec![]),
        ("broken-links", "List broken links", vec![]),
        ("repair-links", "Repair broken links", vec![]),
        (
            "retitle-thoughts",
            "Retitle legacy thoughts by first sentence",
            vec![text("topic"), toggle("apply")],
        ),
        (
            "reindex",
            "Rebuild vector and FTS indexes",
            vec![toggle("if-dirty")],
        ),
        ("reembed", "Rebuild every vector", vec![toggle("dry-run")]),
        ("deleted-list", "List soft-deleted concepts", vec![]),
        (
            "deleted-recover",
            "Recover a soft-deleted concept",
            vec![Arg::new("concept_id").required(true)],
        ),
        (
            "delete",
            "Soft-delete a concept",
            vec![Arg::new("concept_id").required(true), toggle("remove-file")],
        ),
        (
            "deleted-purge",
            "Permanently delete expired soft-deleted concepts",
            vec![number("older-than")],
        ),
        (
            "detach",
            "End the mirror relationship",
            vec![text("bundle-path"), toggle("no-verify"), toggle("force")],
        ),
        (
            "update",
            "Edit a concept's frontmatter via its bundle file",
            vec![
                Arg::new("concept_id").required(true),
                text("title"),
                text("description"),
                text("type"),
                text("tags"),
                text("add-tags"),
                text("remove-tags"),
                repeat("extra", "KEY=VALUE"),
                repeat("unset", "KEY"),
            ],
        ),
        (
            "supersede",
            "Record NEW supersedes OLD",
            vec![
                Arg::new("old_id").required(true),
                Arg::new("new_id").required(true),
            ],
        ),
        (
            "serve",
            "Own the database and serve the HTTP op plane",
            vec![Arg::new("listen")
                .long("listen")
                .default_value("127.0.0.1:8765")],
        ),
        (
            "setup",
            "One-time setup tasks",
            vec![
                Arg::new("what")
                    .required(true)
                    .value_parser(["extensions", "skills"]),
                Arg::new("from_dir").long("from"),
                text("to"),
                repeat("skill", "NAME"),
                toggle("no-download"),
            ],
        ),
        (
            "images",
            "List image assets attached to a concept",
            vec![Arg::new("concept_id").required(true)],
        ),
        (
            "image",
            "Fetch one image asset",
            vec![Arg::new("asset_id").required(true), text("output-path")],
        ),
    ]
}

pub fn run<I, T>(args: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let let_args: Vec<OsString> = args.into_iter().map(Into::into).collect();
    let matches = match command().try_get_matches_from(let_args.clone()) {
        Ok(m) => m,
        Err(err) => {
            let code = if err.use_stderr() { 2 } else { 0 };
            err.print().expect("clap output");
            return code;
        }
    };
    let (name, _) = match matches.subcommand() {
        Some(pair) => pair,
        None => {
            command().print_help().expect("help");
            return 0;
        }
    };
    let json = matches.get_flag("json");
    match execute(&matches, json, &let_args) {
        Ok((op, data)) => {
            if json {
                println!(
                    "{}",
                    Envelope::success(op, data).to_json_pretty().expect("json")
                );
            } else {
                render(data);
            }
            0
        }
        Err(mut err) => {
            let op = err.op.clone().unwrap_or_else(|| name.to_string());
            if err.code == "INTERNAL" {
                let argv: Vec<String> = let_args
                    .iter()
                    .map(|arg| arg.to_string_lossy().into_owned())
                    .collect();
                if let Some(path) = crate::util::crash::write_report(&err, name, &argv) {
                    let shown = path.to_string_lossy().into_owned();
                    err.remedy = Some(format!(
                        "this is a bug — full traceback: {shown}; report it with that file"
                    ));
                    err.fields
                        .insert("crash_report".into(), Value::String(shown));
                }
            }
            if json {
                println!(
                    "{}",
                    Envelope::failure(op, &err).to_json_pretty().expect("json")
                );
            } else {
                eprintln!("[ERROR] {}", err.titles());
            }
            err.exit_code()
        }
    }
}

/// The CLI either owns the router or borrows a live `okf serve` owner.
enum CliOpClient {
    Local(Box<Router>),
    Remote(OpHttpClient),
}

impl CliOpClient {
    fn local(&self) -> Option<&Router> {
        match self {
            Self::Local(router) => Some(router.as_ref()),
            Self::Remote(_) => None,
        }
    }
}

impl OpClient for CliOpClient {
    fn call(&self, op: &str, params: &Map<String, Value>) -> Result<Value> {
        match self {
            Self::Local(router) => router.call(op, params),
            Self::Remote(client) => client.call(op, params),
        }
    }
}

fn execute(matches: &ArgMatches, json: bool, argv: &[OsString]) -> Result<(&'static str, Value)> {
    let settings = settings_from(matches)?;
    let (name, sub) = matches
        .subcommand()
        .ok_or_else(|| OkfError::new("BAD_VALUE", "a command is required"))?;

    if name == "init" {
        let warnings = Router::init(&settings.db_path, settings.embedding_dim)?;
        return Ok((
            "init",
            json!({"db_path": settings.db_path, "warnings": warnings}),
        ));
    }
    if name == "serve" {
        let listen = sub
            .get_one::<String>("listen")
            .map(String::as_str)
            .unwrap_or("127.0.0.1:8765");
        let discovery = crate::server::serve(&settings, listen)?;
        return Ok((
            "serve",
            json!({"discovery": discovery.to_string_lossy(), "db_path": settings.db_path}),
        ));
    }
    if name == "model-info" {
        return Ok((
            "model_info",
            local::model_info(&settings, sub.get_flag("converters"))?,
        ));
    }
    if name == "lint" {
        let dir = sub
            .get_one::<String>("dir")
            .map(String::as_str)
            .unwrap_or(".");
        return Ok(("lint", crate::diagnostics::lint(dir, &settings)?));
    }
    if name == "setup" {
        let what = sub
            .get_one::<String>("what")
            .map(String::as_str)
            .unwrap_or("extensions");
        let skills = sub
            .get_many::<String>("skill")
            .map(|values| values.cloned().collect());
        let data = if what == "skills" {
            if sub.get_one::<String>("from_dir").is_some() || sub.get_flag("no-download") {
                return Err(OkfError::new(
                    "BAD_VALUE",
                    "--from/--no-download are for 'setup extensions'",
                )
                .op("setup_skills"));
            }
            local::setup_skills(sub.get_one::<String>("to").map(String::as_str), skills)?
        } else {
            if skills.is_some() {
                return Err(OkfError::new("BAD_VALUE", "--skill is for 'setup skills'")
                    .op("setup_extensions"));
            }
            local::setup_extensions(
                sub.get_one::<String>("to").map(String::as_str),
                sub.get_one::<String>("from_dir").map(String::as_str),
                !sub.get_flag("no-download"),
            )?
        };
        let op = if what == "skills" {
            "setup_skills"
        } else {
            "setup_extensions"
        };
        return Ok((op, data));
    }
    if name == "diff" && sub.get_one::<String>("new").is_some() {
        let old = sub
            .get_one::<String>("old")
            .map(String::as_str)
            .unwrap_or_default();
        let new = sub
            .get_one::<String>("new")
            .map(String::as_str)
            .unwrap_or_default();
        let report = crate::diagnostics::diff_dirs(old, new, &settings)?;
        return diff_result(report);
    }
    if name == "produce" && !sub.get_flag("sync") {
        let source = sub
            .get_one::<String>("source")
            .map(String::as_str)
            .ok_or_else(|| {
                OkfError::new("MISSING_PARAM", "produce needs --source (or --sync)")
                    .op("produce")
                    .field("source", Value::Null)
            })?;
        let from = sub.get_one::<String>("from_").expect("required").as_str();
        let output = sub
            .get_one::<String>("output-dir")
            .map(String::as_str)
            .or(settings.bundle_root.as_deref())
            .unwrap_or(".");
        let bundle = crate::producers::run(
            from,
            Some(source),
            std::path::Path::new(output),
            sub.get_one::<String>("prefix").map(String::as_str),
            sub.get_flag("overwrite"),
            None,
            "produce",
        )?;
        let root = bundle.root.join(&bundle.prefix);
        let lint = crate::diagnostics::lint(output, &settings)?;
        return Ok((
            "produce",
            json!({
                "producer": bundle.producer,
                "files": bundle.files,
                "root": crate::producers::native(&root),
                "prefix": bundle.prefix,
                "lint": lint,
            }),
        ));
    }
    if name == "shell" {
        shell(&settings, argv)?;
        return Ok(("shell", Value::Null));
    }

    // Python 0.15's CLI routes every op through a live `okf serve` owner so a
    // second process never opens the same Ladybug file. `OKFGRAPH_NO_SERVER=1`
    // opts out (useful for testing the direct path).
    let discovered = if std::env::var("OKFGRAPH_NO_SERVER").as_deref() != Ok("1") {
        crate::server::client::discover(&settings.db_path, Duration::from_secs(2))
            .and_then(|url| OpHttpClient::with_timeout(&url, Duration::from_secs(600)).ok())
    } else {
        None
    };
    let router = match discovered {
        Some(client) => CliOpClient::Remote(client),
        None => CliOpClient::Local(Box::new(Router::open(&settings)?)),
    };
    match name {
        "import" => import_cmd(&router, &settings, sub),
        "ingest" => ingest_cmd(&router, &settings, sub),
        "search" => {
            let mut params = Map::new();
            put_str(&mut params, sub, "query");
            put_str(&mut params, sub, "target");
            put_i64(&mut params, sub, "limit");
            put_str(&mut params, sub, "concept-type");
            put_csv(&mut params, sub, "tags");
            put_str(&mut params, sub, "parent-id");
            put_bool(&mut params, sub, "include-chunks");
            put_i64(&mut params, sub, "max-chunks-per-doc");
            put_bool(&mut params, sub, "expand");
            put_i64(&mut params, sub, "context-hops");
            put_bool(&mut params, sub, "hub-rerank");
            put_f64(&mut params, sub, "hub-weight");
            put_str(&mut params, sub, "rank");
            put_i64(&mut params, sub, "efs");
            put_bool(&mut params, sub, "dedupe");
            put_bool(&mut params, sub, "rerank");
            put_kv(&mut params, sub, "where")?;
            put_str(&mut params, sub, "since");
            put_str(&mut params, sub, "until");
            if let Some(value) = sub.get_one::<String>("exclude-tags") {
                params.insert("exclude_tags".into(), csv_array(value));
            }
            put_csv(&mut params, sub, "link-kinds");
            if let Some(value) = sub.get_one::<String>("near") {
                params.insert("near_id".into(), Value::String(value.clone()));
            }
            put_i64(&mut params, sub, "hops");
            put_i64(&mut params, sub, "offset");
            put_f64(&mut params, sub, "recency");
            put_f64(&mut params, sub, "recency-weight");
            Ok(("search", router.call("search", &params)?))
        }
        "read" => {
            let mut params = Map::new();
            put_str(&mut params, sub, "concept_id");
            put_str(&mut params, sub, "include");
            put_i64(&mut params, sub, "max-tokens");
            let data = router.call("read", &params)?;
            if !json {
                if let Some(path) = sub.get_one::<String>("output-path") {
                    let include = sub
                        .get_one::<String>("include")
                        .map(String::as_str)
                        .unwrap_or("body");
                    if include != "document" || params.contains_key("max_tokens") {
                        return Err(OkfError::new(
                            "BAD_VALUE",
                            "--output-path only applies to --include document",
                        )
                        .op("read")
                        .field("include", Value::String(include.into())));
                    }
                    let markdown =
                        data.get("markdown")
                            .and_then(Value::as_str)
                            .ok_or_else(|| {
                                OkfError::new("BAD_VALUE", "document read has no markdown")
                                    .op("read")
                            })?;
                    std::fs::write(path, markdown)
                        .map_err(|err| OkfError::internal(err.to_string(), "std::fs::write"))?;
                }
            }
            Ok(("read", data))
        }
        "traverse" => {
            let mut params = Map::new();
            put_str(&mut params, sub, "start_id");
            put_str(&mut params, sub, "relationship");
            put_str(&mut params, sub, "direction");
            put_i64(&mut params, sub, "depth");
            put_str(&mut params, sub, "node-type");
            put_str(&mut params, sub, "target");
            put_i64(&mut params, sub, "max-path-length");
            put_csv(&mut params, sub, "link-kinds");
            Ok(("traverse", router.call("traverse", &params)?))
        }
        "export" => {
            if sub.get_flag("all") && sub.get_one::<String>("concept-id").is_some() {
                return Err(OkfError::new(
                    "BAD_VALUE",
                    "export takes --all or --concept-id, not both",
                )
                .op("export_bundle"));
            }
            let mut params = Map::new();
            put_str(&mut params, sub, "output-dir");
            put_str(&mut params, sub, "flavor");
            let (op, data) = if sub.get_flag("all") {
                put_str(&mut params, sub, "directory-id");
                put_str(&mut params, sub, "concept-type");
                put_csv(&mut params, sub, "tags");
                ("export_bundle", router.call("export_bundle", &params)?)
            } else {
                let concept_id = sub.get_one::<String>("concept-id").ok_or_else(|| {
                    OkfError::new("MISSING_PARAM", "export needs --all or --concept-id ID")
                        .op("export_concept")
                })?;
                let ignored = [
                    ("--directory-id", "directory-id"),
                    ("--concept-type", "concept-type"),
                    ("--tags", "tags"),
                ]
                .into_iter()
                .filter_map(|(label, key)| sub.get_one::<String>(key).map(|_| label.to_string()))
                .collect::<Vec<_>>();
                if !ignored.is_empty() {
                    return Err(OkfError::new(
                        "BAD_VALUE",
                        format!("single-concept export ignores: {}", ignored.join(", ")),
                    )
                    .op("export_concept")
                    .field("ignored", json!(ignored))
                    .remedy("filters apply to --all only"));
                }
                params.insert("concept_id".into(), Value::String(concept_id.clone()));
                ("export_concept", router.call("export_concept", &params)?)
            };
            Ok((op, data))
        }
        "images" => {
            let mut params = Map::new();
            put_str(&mut params, sub, "concept_id");
            Ok(("list_images", router.call("list_images", &params)?))
        }
        "image" => {
            let mut params = Map::new();
            put_str(&mut params, sub, "asset_id");
            let data = router.call("get_image", &params)?;
            if !json {
                if let Some(path) = sub.get_one::<String>("output-path") {
                    let encoded = data.get("data").and_then(Value::as_str).ok_or_else(|| {
                        let asset = sub.get_one::<String>("asset_id").expect("required");
                        OkfError::new(
                            "UNKNOWN_ASSET",
                            format!("image asset '{asset}' has no stored bytes"),
                        )
                        .op("get_image")
                        .field("asset_id", Value::String(asset.clone()))
                    })?;
                    use base64::Engine;
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(encoded)
                        .map_err(|err| OkfError::internal(err.to_string(), "base64"))?;
                    std::fs::write(path, bytes)
                        .map_err(|err| OkfError::internal(err.to_string(), "std::fs::write"))?;
                }
            }
            Ok(("get_image", data))
        }
        "broken-links" => Ok((
            "list_broken_links",
            router.call("list_broken_links", &Map::new())?,
        )),
        "repair-links" => Ok(("repair_links", router.call("repair_links", &Map::new())?)),
        "retitle-thoughts" => {
            let local = router.local().ok_or_else(|| {
                OkfError::new(
                    "CONFIG_INVALID",
                    "retitle-thoughts needs the local graph; it is not a server op",
                )
                .op("retitle_thoughts")
            })?;
            Ok((
                "retitle_thoughts",
                local.retitle_thoughts(
                    sub.get_one::<String>("topic").map(String::as_str),
                    sub.get_flag("apply"),
                )?,
            ))
        }
        "diff" => {
            let mut params = Map::new();
            put_str(&mut params, sub, "old");
            put_str(&mut params, sub, "new");
            match router.call("diff", &params) {
                Ok(report) => diff_result(report),
                Err(err) => Err(err),
            }
        }
        "doctor" => {
            let mut params = Map::new();
            put_i64(&mut params, sub, "stale-days");
            put_bool(&mut params, sub, "fix");
            put_bool(&mut params, sub, "strict");
            if let Some(cache) = settings.cache_dir.as_deref() {
                params.insert("cache_dir".into(), Value::String(cache.into()));
            }
            Ok(("doctor", router.call("doctor", &params)?))
        }
        "produce" | "sync" => {
            let mut params = Map::new();
            let (op, producer_key) = if sub.get_flag("sync") {
                ("sync", "from_")
            } else {
                ("produce", "from_")
            };
            put_str(&mut params, sub, producer_key);
            if op == "sync" {
                if let Some(value) = params
                    .get("from_")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                {
                    params.remove("from_");
                    params.insert("producer".into(), Value::String(value));
                }
            } else if let Some(value) = params
                .get("from_")
                .and_then(Value::as_str)
                .map(str::to_string)
            {
                params.remove("from_");
                params.insert("source_type".into(), Value::String(value));
            }
            put_str(&mut params, sub, "source");
            if op == "sync" && params.contains_key("source") {
                let source = params.remove("source").expect("checked");
                params.insert("source".into(), source);
            } else if params.contains_key("source") {
                let source = params.remove("source").expect("checked");
                params.insert("source_path".into(), source);
            }
            put_str(&mut params, sub, "output-dir");
            put_str(&mut params, sub, "prefix");
            put_bool(&mut params, sub, "overwrite");
            Ok((op, router.call(op, &params)?))
        }
        "reindex" => {
            let mut params = Map::new();
            put_bool(&mut params, sub, "if-dirty");
            Ok(("reindex", router.call("reindex", &params)?))
        }
        "reembed" => {
            let mut params = Map::new();
            put_bool(&mut params, sub, "dry-run");
            Ok(("reembed", router.call("reembed", &params)?))
        }
        "deleted-list" => Ok(("list_deleted", router.call("list_deleted", &Map::new())?)),
        "deleted-recover" => {
            let mut params = Map::new();
            put_str(&mut params, sub, "concept_id");
            Ok(("recover_deleted", router.call("recover_deleted", &params)?))
        }
        "delete" => {
            let mut params = Map::new();
            put_str(&mut params, sub, "concept_id");
            put_bool(&mut params, sub, "remove-file");
            Ok(("delete", router.call("delete", &params)?))
        }
        "deleted-purge" => {
            let mut params = Map::new();
            put_i64(&mut params, sub, "older-than");
            Ok(("purge_deleted", router.call("purge_deleted", &params)?))
        }
        "detach" => {
            let mut params = Map::new();
            put_str(&mut params, sub, "bundle-path");
            if sub.get_flag("no-verify") {
                params.insert("verify".into(), Value::Bool(false));
            }
            put_bool(&mut params, sub, "force");
            Ok(("detach", router.call("detach", &params)?))
        }
        "update" => {
            let mut params = Map::new();
            put_str(&mut params, sub, "concept_id");
            for key in [
                "title",
                "description",
                "type",
                "tags",
                "add-tags",
                "remove-tags",
            ] {
                put_str(&mut params, sub, key);
            }
            let mut extra = kv_pairs(sub, "extra", "--extra")?;
            for key in sub
                .get_many::<String>("unset")
                .map(|v| v.cloned().collect::<Vec<_>>())
                .unwrap_or_default()
            {
                extra.insert(key, Value::Null);
            }
            if !extra.is_empty() {
                params.insert("extra".into(), Value::Object(extra));
            }
            Ok(("update", router.call("update", &params)?))
        }
        "supersede" => {
            let mut params = Map::new();
            put_str(&mut params, sub, "old_id");
            put_str(&mut params, sub, "new_id");
            Ok(("supersede", router.call("supersede", &params)?))
        }
        _ => Err(OkfError::new(
            "BAD_VALUE",
            format!("unknown command: {name}"),
        )),
    }
}

fn diff_result(report: Value) -> Result<(&'static str, Value)> {
    match report.get("identical").and_then(Value::as_bool) {
        Some(false) => Err(OkfError::outcome(
            "DIFF_DIFFERENT",
            "structures differ (report in data)",
            report,
        )
        .op("diff")),
        _ => Ok(("diff", report)),
    }
}

fn import_cmd(
    router: &dyn OpClient,
    settings: &Settings,
    sub: &ArgMatches,
) -> Result<(&'static str, Value)> {
    let mode = sub
        .get_one::<String>("mode")
        .map(String::as_str)
        .unwrap_or(&settings.mode);
    let mut files: Vec<String> = sub
        .get_many::<String>("files")
        .map(|values| values.cloned().collect())
        .unwrap_or_default();
    let all = sub.get_flag("all");
    if all && !files.is_empty() {
        return Err(OkfError::new(
            "BAD_VALUE",
            "import --all takes no FILES (use --bundle-path DIR to pin one tree)",
        )
        .op("import_bundle")
        .field("files", json!(files)));
    }
    let rebuild = sub
        .get_one::<String>("rebuild")
        .map(String::as_str)
        .unwrap_or("auto");
    if !matches!(rebuild, "auto" | "defer" | "force") {
        return Err(OkfError::new(
            "BAD_VALUE",
            format!("rebuild must be 'auto', 'defer' or 'force', got '{rebuild}'"),
        )
        .op("import_file")
        .field("rebuild", json!(rebuild)));
    }
    let mut params = Map::new();
    put_i64(&mut params, sub, "batch-size");
    params.insert("mode".into(), Value::String(mode.into()));
    params.insert("rebuild".into(), Value::String(rebuild.into()));
    put_bool(&mut params, sub, "prune-missing");
    put_bool(&mut params, sub, "force");
    if all {
        put_str(&mut params, sub, "bundle-path");
        return Ok(("import_bundle", router.call("import_bundle", &params)?));
    }
    if files.is_empty() {
        return Err(
            OkfError::new("MISSING_PARAM", "import needs FILES or --all").op("import_file"),
        );
    }
    if sub.get_flag("prune-missing") {
        return Err(
            OkfError::new("BAD_VALUE", "--prune-missing only applies with --all")
                .op("import_file")
                .field("flag", Value::String("--prune-missing".into())),
        );
    }
    let missing: Vec<String> = files
        .iter()
        .filter(|path| !std::path::Path::new(path).is_file())
        .cloned()
        .collect();
    if !missing.is_empty() {
        return Err(OkfError::new(
            "FILE_NOT_FOUND",
            format!("file(s) not found: {}", missing.join(", ")),
        )
        .op("import_file")
        .field("files", json!(missing)));
    }
    if files.len() > 1 {
        params.insert(
            "paths".into(),
            Value::Array(files.into_iter().map(Value::String).collect()),
        );
        return Ok(("import_files", router.call("import_files", &params)?));
    }
    params.insert("file_path".into(), Value::String(files.remove(0)));
    let data = router.call("import_file", &params)?;
    Ok(("import_file", json!([data])))
}

fn ingest_cmd(
    router: &dyn OpClient,
    settings: &Settings,
    sub: &ArgMatches,
) -> Result<(&'static str, Value)> {
    let mut params = Map::new();
    put_str(&mut params, sub, "kind");
    for key in [
        "md-path",
        "bib-path",
        "thoughts",
        "topic",
        "concept-id",
        "title",
        "description",
        "type",
        "output-dir",
        "routing-mode",
        "folder",
    ] {
        put_str(&mut params, sub, key);
    }
    if let Some(paths) = sub.get_many::<String>("pdf-path") {
        params.insert(
            "pdf_path".into(),
            Value::Array(paths.cloned().map(Value::String).collect()),
        );
    }
    put_csv(&mut params, sub, "tags");
    if let Some(values) = sub.get_many::<String>("related") {
        params.insert(
            "related".into(),
            Value::Array(values.cloned().map(Value::String).collect()),
        );
    }
    let extra = kv_pairs(sub, "extra", "--extra")?;
    if !extra.is_empty() {
        params.insert("extra".into(), Value::Object(extra));
    }
    params.insert(
        "mode".into(),
        Value::String(
            sub.get_one::<String>("mode")
                .map(String::as_str)
                .unwrap_or(&settings.mode)
                .into(),
        ),
    );
    params.insert(
        "auto_import".into(),
        Value::Bool(sub.get_flag("auto-import")),
    );
    params.insert(
        "extract_images".into(),
        Value::Bool(sub.get_flag("extract-images")),
    );
    put_i64(&mut params, sub, "batch-size");
    put_bool(&mut params, sub, "prune-missing");
    put_bool(&mut params, sub, "force");
    put_bool(&mut params, sub, "materialize");
    Ok(("ingest", router.call("ingest", &params)?))
}

fn diff_params_key(name: &str) -> String {
    match name {
        "batch-size" => "batch_size",
        "prune-missing" => "prune_missing",
        "node-type" => "node_type",
        "max-path-length" => "max_path_length",
        "link-kinds" => "link_kinds",
        "include-chunks" => "include_chunks",
        "max-chunks-per-doc" => "max_chunks_per_doc",
        "context-hops" => "context_hops",
        "hub-rerank" => "hub_rerank",
        "hub-weight" => "hub_weight",
        "recency-weight" => "recency_weight",
        "if-dirty" => "if_dirty",
        "dry-run" => "dry_run",
        "remove-file" => "remove_file",
        "older-than" => "older_than",
        "no-verify" => "no_verify",
        "stale-days" => "stale_days",
        "add-tags" => "add_tags",
        "remove-tags" => "remove_tags",
        "bundle-path" => "bundle_path",
        "md-path" => "md_path",
        "bib-path" => "bib_path",
        "pdf-path" => "pdf_path",
        "concept-id" => "concept_id",
        "output-dir" => "output_dir",
        "routing-mode" => "routing_mode",
        "output-path" => "output_path",
        "source" => "source",
        "from_" => "from",
        other => other,
    }
    .to_string()
}

fn put_str(params: &mut Map<String, Value>, matches: &ArgMatches, name: &str) {
    if let Some(value) = matches.get_one::<String>(name) {
        params.insert(diff_params_key(name), Value::String(value.clone()));
    }
}

fn put_bool(params: &mut Map<String, Value>, matches: &ArgMatches, name: &str) {
    if matches.try_get_one::<bool>(name).ok().flatten().copied() == Some(true) {
        params.insert(diff_params_key(name), Value::Bool(true));
    }
}

fn put_i64(params: &mut Map<String, Value>, matches: &ArgMatches, name: &str) {
    if let Some(value) = matches.try_get_one::<i64>(name).ok().flatten().copied() {
        params.insert(diff_params_key(name), Value::from(value));
    }
}

fn put_f64(params: &mut Map<String, Value>, matches: &ArgMatches, name: &str) {
    if let Some(value) = matches.try_get_one::<f64>(name).ok().flatten().copied() {
        params.insert(
            diff_params_key(name),
            serde_json::Number::from_f64(value)
                .map(Value::Number)
                .unwrap_or(Value::Null),
        );
    }
}

fn put_csv(params: &mut Map<String, Value>, matches: &ArgMatches, name: &str) {
    if let Some(value) = matches.get_one::<String>(name) {
        params.insert(diff_params_key(name), csv_array(value));
    }
}

fn csv_array(value: &str) -> Value {
    Value::Array(
        value
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(|item| Value::String(item.into()))
            .collect(),
    )
}

fn kv_pairs(matches: &ArgMatches, name: &str, flag: &str) -> Result<Map<String, Value>> {
    let mut out = Map::new();
    for item in matches
        .get_many::<String>(name)
        .map(|values| values.cloned().collect::<Vec<_>>())
        .unwrap_or_default()
    {
        let (key, value) = item.split_once('=').ok_or_else(|| {
            OkfError::new(
                "BAD_VALUE",
                format!("{flag} must be KEY=VALUE, got '{item}'"),
            )
            .field(flag, Value::String(item.clone()))
        })?;
        if key.is_empty() {
            return Err(OkfError::new(
                "BAD_VALUE",
                format!("{flag} must be KEY=VALUE, got '{item}'"),
            )
            .field(flag, Value::String(item)));
        }
        out.insert(key.into(), Value::String(value.into()));
    }
    Ok(out)
}

fn put_kv(params: &mut Map<String, Value>, matches: &ArgMatches, name: &str) -> Result<()> {
    let value = kv_pairs(matches, name, &format!("--{name}"))?;
    if !value.is_empty() {
        params.insert(name.into(), Value::Object(value));
    }
    Ok(())
}

fn shell(settings: &Settings, argv: &[OsString]) -> Result<()> {
    use std::io::BufRead;
    println!(
        "OKF Interactive Shell\ntype 'help' for commands, 'quit' to exit (http ops via okf serve)"
    );
    let prefix: Vec<OsString> = argv
        .iter()
        .take_while(|arg| arg.to_string_lossy() != "shell")
        .cloned()
        .collect();
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line.trim().to_string(),
            Err(_) => break,
        };
        if line.is_empty() {
            continue;
        }
        if matches!(line.as_str(), "quit" | "exit") {
            break;
        }
        if line == "help" {
            println!("import, search, read, traverse, images, delete, model-info, broken-links, repair-links, quit");
            continue;
        }
        let tokens: Vec<String> = line.split_whitespace().map(str::to_string).collect();
        let mut args = prefix.clone();
        args.extend(tokens.iter().map(OsString::from));
        let _ = super::cli::run(args);
        // The DB owner is per command; keep the shell usable even when one
        // line fails. Settings are resolved again from the same global flags.
        let _ = settings;
    }
    Ok(())
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
                Some(v) => json!(v),
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

fn leak(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_tree_has_all_29_verbs() {
        let cmd = command();
        let actual: Vec<String> = cmd
            .get_subcommands()
            .map(|sub| sub.get_name().to_string())
            .collect();
        let mut actual = actual;
        let mut expected = COMMANDS
            .iter()
            .map(|name| name.to_string())
            .collect::<Vec<_>>();
        actual.sort();
        expected.sort();
        assert_eq!(actual, expected);
    }

    #[test]
    fn kv_pairs_rejects_malformed_items() {
        let matches = command()
            .try_get_matches_from(["okf", "update", "alpha", "--extra", "oops"])
            .unwrap();
        let sub = matches.subcommand().unwrap().1;
        let err = kv_pairs(sub, "extra", "--extra").unwrap_err();
        assert_eq!(err.code, "BAD_VALUE");
    }
}
