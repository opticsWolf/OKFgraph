//! Deterministic SQLite → markdown bundle producer, matching Python 0.15.

use crate::error::{OkfError, Result};
use crate::export::fs_write;
use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub const PRODUCERS: &[&str] = &["sqlite"];

#[derive(Debug, Clone)]
struct Column {
    name: String,
    kind: String,
    nullable: bool,
    default: String,
    pk: bool,
}

#[derive(Debug, Clone)]
struct ForeignKey {
    from_col: String,
    ref_table: String,
    ref_col: String,
}

#[derive(Debug, Clone)]
struct Table {
    name: String,
    stem: String,
    columns: Vec<Column>,
    fks: Vec<ForeignKey>,
    row_count: i64,
}

#[derive(Debug, Clone)]
pub struct ProducedBundle {
    pub root: PathBuf,
    pub prefix: String,
    pub files: usize,
    pub producer: &'static str,
    pub source: String,
    pub cursor: String,
    pub written: Vec<PathBuf>,
}

fn qi(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn safe_filename(name: &str) -> String {
    let safe: String = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    let safe = safe.trim_matches(|ch| ch == '.' || ch == '_');
    if safe.is_empty() {
        "table".to_string()
    } else {
        safe.to_string()
    }
}

pub(crate) fn native(path: &Path) -> String {
    let text = path.to_string_lossy().into_owned();
    #[cfg(windows)]
    let text = text.replace('/', "\\");
    text
}

fn read_schema(src: &Path) -> Result<Vec<Table>> {
    let con =
        rusqlite::Connection::open_with_flags(src, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(|err| {
                OkfError::new(
                    "BAD_VALUE",
                    format!("not a SQLite database: {} ({err})", native(src)),
                )
            })?;
    let names: Vec<String> = con
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        .and_then(|mut stmt| {
            stmt.query_map([], |row| row.get::<_, String>(0))?
                .collect::<std::result::Result<Vec<_>, _>>()
        })
        .map_err(|err| OkfError::new("BAD_VALUE", format!("not a SQLite database: {} ({err})", native(src))))?;
    let mut used = BTreeSet::new();
    let mut tables = Vec::new();
    for name in names {
        let base = safe_filename(&name);
        let mut stem = base.clone();
        let mut n = 2;
        while used.contains(&stem) {
            stem = format!("{base}_{n}");
            n += 1;
        }
        used.insert(stem.clone());
        let mut stmt = con
            .prepare(&format!("PRAGMA table_info({})", qi(&name)))
            .map_err(|err| OkfError::internal(err.to_string(), "sqlite"))?;
        let columns = stmt
            .query_map([], |row| {
                Ok(Column {
                    name: row.get::<_, String>(1)?,
                    kind: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    nullable: row.get::<_, i64>(3)? == 0,
                    default: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                    pk: row.get::<_, i64>(5)? != 0,
                })
            })
            .and_then(|rows| rows.collect::<std::result::Result<Vec<_>, _>>())
            .map_err(|err| OkfError::internal(err.to_string(), "sqlite"))?;
        drop(stmt);
        let mut stmt = con
            .prepare(&format!("PRAGMA foreign_key_list({})", qi(&name)))
            .map_err(|err| OkfError::internal(err.to_string(), "sqlite"))?;
        let mut fks = stmt
            .query_map([], |row| {
                Ok(ForeignKey {
                    ref_table: row.get::<_, String>(2)?,
                    from_col: row.get::<_, String>(3)?,
                    ref_col: row.get::<_, String>(4)?,
                })
            })
            .and_then(|rows| rows.collect::<std::result::Result<Vec<_>, _>>())
            .map_err(|err| OkfError::internal(err.to_string(), "sqlite"))?;
        drop(stmt);
        fks.sort_by(|a, b| {
            (&a.from_col, &a.ref_table, &a.ref_col).cmp(&(&b.from_col, &b.ref_table, &b.ref_col))
        });
        let quoted = qi(&name);
        let row_count: i64 = con
            .query_row(&format!("SELECT COUNT(*) FROM {quoted}"), [], |row| {
                row.get(0)
            })
            .map_err(|err| {
                OkfError::new(
                    "BAD_VALUE",
                    format!("cannot read table '{name}' in {} ({err})", native(src)),
                )
            })?;
        tables.push(Table {
            name,
            stem,
            columns,
            fks,
            row_count,
        });
    }
    Ok(tables)
}

fn percent(rate: f64) -> String {
    format!("{:.0}%", rate * 100.0)
}

fn observe(con: &rusqlite::Connection, table: &Table) -> Result<Vec<String>> {
    if table.row_count == 0 {
        return Ok(vec![format!("Table `{}` is empty (0 rows).", table.name)]);
    }
    let quoted = qi(&table.name);
    let mut findings = Vec::new();
    let mut checks = Vec::new();
    for column in &table.columns {
        checks.push(format!(
            "COUNT({}), COUNT(DISTINCT {})",
            qi(&column.name),
            qi(&column.name)
        ));
    }
    if !checks.is_empty() {
        let sql = format!("SELECT {} FROM {quoted}", checks.join(", "));
        let mut stmt = con
            .prepare(&sql)
            .map_err(|err| OkfError::internal(err.to_string(), "sqlite"))?;
        let counts: Vec<i64> = stmt
            .query_map([], |row| {
                let mut values = Vec::new();
                for index in 0..table.columns.len() {
                    values.push(row.get::<_, i64>(index * 2)?);
                    values.push(row.get::<_, i64>(index * 2 + 1)?);
                }
                Ok(values)
            })
            .and_then(|mut rows| rows.next().unwrap_or_else(|| Ok(Vec::new())))
            .map_err(|err| OkfError::internal(err.to_string(), "sqlite"))?;
        drop(stmt);
        let holder_cols: BTreeSet<&str> = table.fks.iter().map(|fk| fk.from_col.as_str()).collect();
        for (index, column) in table.columns.iter().enumerate() {
            let present = counts.get(index * 2).copied().unwrap_or(0);
            let distinct = counts.get(index * 2 + 1).copied().unwrap_or(0);
            let null_rate = 1.0 - present as f64 / table.row_count as f64;
            if null_rate >= 0.5 {
                findings.push(format!(
                    "`{}.{}` is NULL in {} of rows ({}/{}) — column may be deprecated or unpopulated.",
                    table.name, column.name, percent(null_rate), table.row_count - present, table.row_count
                ));
            }
            if present > 0 && !holder_cols.contains(column.name.as_str()) {
                let dup_rate = 1.0 - distinct as f64 / present as f64;
                if dup_rate >= 0.1 {
                    findings.push(format!(
                        "`{}.{}` has {} duplicate values ({} distinct / {} present) — uniqueness constraint missing?",
                        table.name, column.name, percent(dup_rate), distinct, present
                    ));
                }
            }
            let mut stmt = con
                .prepare(&format!(
                    "SELECT DISTINCT typeof({}) FROM {quoted}",
                    qi(&column.name)
                ))
                .map_err(|err| OkfError::internal(err.to_string(), "sqlite"))?;
            let mut types: Vec<String> = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .and_then(|rows| rows.collect::<std::result::Result<Vec<_>, _>>())
                .map_err(|err| OkfError::internal(err.to_string(), "sqlite"))?;
            drop(stmt);
            types.sort();
            types.dedup();
            let real: Vec<String> = types.into_iter().filter(|kind| kind != "null").collect();
            if real.len() >= 2 {
                findings.push(format!(
                    "`{}.{}` stores multiple types: {} — schema variance, check producers.",
                    table.name,
                    column.name,
                    real.join(", ")
                ));
            }
        }
    }
    for fk in &table.fks {
        let sql = format!(
            "SELECT COUNT(*) FROM {quoted} WHERE {} IS NOT NULL AND NOT EXISTS \
             (SELECT 1 FROM {} WHERE {}.{} = {}.{})",
            qi(&fk.from_col),
            qi(&fk.ref_table),
            qi(&fk.ref_table),
            qi(&fk.ref_col),
            quoted,
            qi(&fk.from_col)
        );
        let orphans: i64 = con
            .query_row(&sql, [], |row| row.get(0))
            .map_err(|err| OkfError::internal(err.to_string(), "sqlite"))?;
        if orphans > 0 {
            findings.push(format!(
                "`{}.{}` has {} value(s) with no match in `{}.{}` — orphaned references.",
                table.name, fk.from_col, orphans, fk.ref_table, fk.ref_col
            ));
        }
    }
    Ok(findings)
}

fn frontmatter(entries: Vec<(&str, Json)>) -> String {
    let mut map = Map::new();
    for (key, value) in entries {
        map.insert(key.to_string(), value);
    }
    crate::export::dump_yaml(&Json::Object(map), 0)
}

fn table_doc(
    db_name: &str,
    prefix: &str,
    table: &Table,
    incoming: &[ForeignKey],
    by_name: &BTreeMap<String, &Table>,
    observations: &[String],
) -> String {
    let pks: Vec<&str> = table
        .columns
        .iter()
        .filter(|c| c.pk)
        .map(|c| c.name.as_str())
        .collect();
    let yaml = frontmatter(vec![
        ("title", Json::String(format!("Table {}", table.name))),
        ("type", Json::String("table".into())),
        ("database", Json::String(db_name.into())),
        ("table", Json::String(table.name.clone())),
        ("column_count", Json::from(table.columns.len())),
        (
            "primary_key",
            Json::String(if pks.len() == 1 {
                pks[0].to_string()
            } else {
                String::new()
            }),
        ),
        (
            "composite_primary_key",
            Json::Array(if pks.len() > 1 {
                pks.into_iter().map(|v| Json::String(v.into())).collect()
            } else {
                Vec::new()
            }),
        ),
        ("row_count", Json::from(table.row_count)),
    ]);
    let mut lines = vec![
        "---".to_string(),
        yaml.trim_end().to_string(),
        "---".to_string(),
        String::new(),
        format!("# Table {}", table.name),
        String::new(),
        "## Schema".to_string(),
        String::new(),
        "| column | type | nullable | default | key |".to_string(),
        "| --- | --- | --- | --- | --- |".to_string(),
    ];
    for column in &table.columns {
        let default = if column.default.is_empty() {
            "—".to_string()
        } else {
            format!("`{}`", column.default)
        };
        lines.push(format!(
            "| `{}` | {} | {} | {} | {} |",
            column.name,
            if column.kind.is_empty() {
                "—"
            } else {
                &column.kind
            },
            if column.nullable { "yes" } else { "no" },
            default,
            if column.pk { "PK" } else { "" }
        ));
    }
    lines.extend([
        "".to_string(),
        "## Relationships".to_string(),
        String::new(),
    ]);
    if table.fks.is_empty() && incoming.is_empty() {
        lines.push("None.".to_string());
    } else {
        for fk in &table.fks {
            if let Some(target) = by_name.get(&fk.ref_table) {
                lines.push(format!(
                    "- [{}]({}/tables/{}.md) — `{}.{}` → `{}.{}`",
                    fk.ref_table,
                    prefix,
                    target.stem,
                    table.name,
                    fk.from_col,
                    fk.ref_table,
                    fk.ref_col
                ));
            } else {
                lines.push(format!(
                    "- `{}.{}` → `{}.{}` (external table, no concept)",
                    table.name, fk.from_col, fk.ref_table, fk.ref_col
                ));
            }
        }
        let mut sorted = incoming.to_vec();
        sorted.sort_by(|a, b| (&a.from_col, &a.ref_table).cmp(&(&b.from_col, &b.ref_table)));
        for inc in sorted {
            if inc.ref_table != table.name {
                lines.push(format!(
                    "- Referenced by `{}.{}`",
                    inc.ref_table, inc.from_col
                ));
            }
        }
    }
    if !observations.is_empty() {
        lines.extend(["".to_string(), "## Observations".to_string(), String::new()]);
        for finding in observations {
            lines.push(format!("- {finding}"));
        }
    }
    lines.join("\n") + "\n"
}

fn overview_doc(
    db_name: &str,
    prefix: &str,
    tables: &[Table],
    observations: &BTreeMap<String, Vec<String>>,
) -> String {
    let yaml = frontmatter(vec![
        ("title", Json::String(format!("Database {db_name}"))),
        ("type", Json::String("database".into())),
        ("database", Json::String(db_name.into())),
        ("table_count", Json::from(tables.len())),
    ]);
    let mut lines = vec![
        "---".to_string(),
        yaml.trim_end().to_string(),
        "---".to_string(),
        String::new(),
        format!("# Database {db_name}"),
        String::new(),
        "## Tables".to_string(),
        String::new(),
    ];
    for table in tables {
        let count = observations.get(&table.name).map_or(0, Vec::len);
        let marker = if count == 0 {
            String::new()
        } else {
            format!(", {count} observation{}", if count == 1 { "" } else { "s" })
        };
        lines.push(format!(
            "- [{}]({}/tables/{}.md) — {} columns, {} rows{}",
            table.name,
            prefix,
            table.stem,
            table.columns.len(),
            table.row_count,
            marker
        ));
    }
    lines.join("\n") + "\n"
}

fn append_log(
    prefix_dir: &Path,
    db_name: &str,
    tables: &[Table],
    observations: &BTreeMap<String, Vec<String>>,
) -> Result<()> {
    let total: usize = observations.values().map(Vec::len).sum();
    let names = tables
        .iter()
        .map(|t| t.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let word = if total == 1 {
        "observation"
    } else {
        "observations"
    };
    let entry = format!(
        "produce sqlite {db_name} → {} files ({} tables [{}] + overview, {total} {word})\n",
        tables.len() + 1,
        tables.len(),
        names
    );
    let log = prefix_dir.join("log.md");
    let existing = fs::read_to_string(&log).unwrap_or_default();
    let same_last = existing
        .trim_end_matches(['\r', '\n'])
        .ends_with(entry.trim_end_matches('\n'));
    if same_last {
        return Ok(());
    }
    let text = if existing.is_empty() {
        format!("# Producer log\n\n{entry}")
    } else {
        format!("{existing}{entry}")
    };
    fs_write(&log, &text)
}

fn markdown_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            markdown_files(&path, out);
        } else if path.extension().and_then(|v| v.to_str()) == Some("md") {
            out.push(path);
        }
    }
}

pub fn run(
    producer: &str,
    source: Option<&str>,
    output_dir: &Path,
    prefix: Option<&str>,
    overwrite: bool,
    cursor: Option<&str>,
    op: &str,
) -> Result<ProducedBundle> {
    if producer != "sqlite" {
        return Err(
            OkfError::new("BAD_VALUE", format!("unknown producer '{producer}'"))
                .op(op)
                .field("source_type", Json::String(producer.into()))
                .field(
                    "available",
                    Json::Array(
                        PRODUCERS
                            .iter()
                            .map(|v| Json::String((*v).into()))
                            .collect(),
                    ),
                ),
        );
    }
    let source =
        source.ok_or_else(|| OkfError::new("MISSING_PARAM", "source_path is required").op(op))?;
    let src = PathBuf::from(source);
    if !src.is_file() {
        return Err(OkfError::new(
            "FILE_NOT_FOUND",
            format!("SQLite source not found: {}", native(&src)),
        )
        .op(op)
        .field("source_path", Json::String(source.into())));
    }
    let prefix = prefix.unwrap_or("database").to_string();
    if prefix.contains('/') || prefix.contains('\\') || prefix == "." || prefix == ".." {
        return Err(OkfError::new(
            "BAD_VALUE",
            format!("prefix must be a single directory name, got '{prefix}'"),
        )
        .op(op));
    }
    let bytes =
        fs::read(&src).map_err(|err| OkfError::new("FILE_NOT_FOUND", err.to_string()).op(op))?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if cursor == Some(digest.as_str()) {
        let mut existing = Vec::new();
        markdown_files(&output_dir.join(&prefix), &mut existing);
        existing.sort();
        return Ok(ProducedBundle {
            root: output_dir.to_path_buf(),
            prefix,
            files: existing.len(),
            producer: "sqlite",
            source: src
                .file_name()
                .map(|v| v.to_string_lossy().into_owned())
                .unwrap_or_default(),
            cursor: digest,
            written: Vec::new(),
        });
    }
    let tables = read_schema(&src)?;
    if tables.is_empty() {
        return Err(OkfError::new(
            "BAD_VALUE",
            format!("no user tables found in SQLite database: {}", native(&src)),
        )
        .op(op));
    }
    let con =
        rusqlite::Connection::open_with_flags(&src, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(|err| OkfError::internal(err.to_string(), "sqlite"))?;
    let mut observations = BTreeMap::new();
    for table in &tables {
        observations.insert(table.name.clone(), observe(&con, table)?);
    }
    let prefix_dir = output_dir.join(&prefix);
    let dest = prefix_dir.join("tables");
    fs::create_dir_all(&dest).map_err(|err| OkfError::internal(err.to_string(), "fs"))?;
    let mut paths: Vec<PathBuf> = tables
        .iter()
        .map(|t| dest.join(format!("{}.md", t.stem)))
        .collect();
    paths.push(prefix_dir.join("overview.md"));
    if !overwrite {
        let existing: Vec<String> = paths
            .iter()
            .filter(|p| p.exists())
            .map(|p| native(p))
            .collect();
        if !existing.is_empty() {
            return Err(OkfError::new(
                "INTERNAL",
                format!(
                    "refusing to clobber existing producer output (pass overwrite=True): {}",
                    existing.join(", ")
                ),
            )
            .op(op));
        }
    }
    let by_name: BTreeMap<String, &Table> = tables.iter().map(|t| (t.name.clone(), t)).collect();
    let mut referenced_by: BTreeMap<String, Vec<ForeignKey>> = BTreeMap::new();
    for table in &tables {
        referenced_by.entry(table.name.clone()).or_default();
    }
    for table in &tables {
        for fk in &table.fks {
            if let Some(list) = referenced_by.get_mut(&fk.ref_table) {
                list.push(ForeignKey {
                    from_col: table.name.clone(),
                    ref_table: fk.from_col.clone(),
                    ref_col: fk.ref_col.clone(),
                });
            }
        }
    }
    let db_name = src
        .file_name()
        .map(|v| v.to_string_lossy().into_owned())
        .unwrap_or_default();
    for table in &tables {
        let incoming = referenced_by.get(&table.name).cloned().unwrap_or_default();
        let text = table_doc(
            &db_name,
            &prefix,
            table,
            &incoming,
            &by_name,
            observations.get(&table.name).map_or(&[][..], Vec::as_slice),
        );
        fs_write(&dest.join(format!("{}.md", table.stem)), &text)?;
    }
    fs_write(
        &prefix_dir.join("overview.md"),
        &overview_doc(&db_name, &prefix, &tables, &observations),
    )?;
    append_log(&prefix_dir, &db_name, &tables, &observations)?;
    Ok(ProducedBundle {
        root: output_dir.to_path_buf(),
        prefix,
        files: paths.len(),
        producer: "sqlite",
        source: db_name,
        cursor: digest,
        written: paths,
    })
}
