//! Pure ranking algorithms shared by concept/chunk search.
//!
//! The oracle's RRF, hub blend and PPR are deliberately deterministic; all
//! maps are `BTreeMap`/`BTreeSet` and tie-breaks are by id.

use serde_json::Value as Json;
use std::collections::{BTreeMap, BTreeSet};

pub const FETCH_CAP: i64 = 1000;
pub const PAGE_WINDOW: usize = 25;
pub const DEDUPE_COSINE: f64 = 0.97;

const STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from", "how", "what",
    "when", "where", "which", "does", "did", "can", "could", "should", "would", "will", "has",
    "have", "had", "not", "its", "our", "your", "their", "about", "into", "over", "under", "why",
    "who", "whom",
];

pub fn fetch_k(limit: usize, filtered: bool) -> i64 {
    let limit = limit.max(PAGE_WINDOW) as i64;
    if !filtered {
        limit * 3
    } else {
        (limit * 10).max(50).min(FETCH_CAP.max(limit * 3))
    }
}

pub fn deeper_k(k: i64, found: usize, limit: usize, full: bool) -> Option<i64> {
    if found >= limit || !full || k >= FETCH_CAP {
        return None;
    }
    Some((k * 4).min(FETCH_CAP))
}

/// Reciprocal Rank Fusion (k=60) of two score lists, best first, ties by id.
pub fn rrf_fuse(vec_scores: &[(String, f64)], fts_scores: &[(String, f64)]) -> Vec<(String, f64)> {
    let ranks = |scores: &[(String, f64)]| -> BTreeMap<String, usize> {
        let mut ordered = scores.to_vec();
        ordered.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ordered
            .into_iter()
            .enumerate()
            .map(|(i, (id, _))| (id, i + 1))
            .collect()
    };
    let vec_rank = ranks(vec_scores);
    let fts_rank = ranks(fts_scores);
    let mut combined: BTreeMap<String, f64> = BTreeMap::new();
    for (id, rank) in &vec_rank {
        combined.insert(id.clone(), 1.0 / (60.0 + *rank as f64));
    }
    for (id, rank) in &fts_rank {
        *combined.entry(id.clone()).or_default() += 1.0 / (60.0 + *rank as f64);
    }
    let mut out: Vec<(String, f64)> = combined.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

/// Lowercase alphanumeric tokens, length >= 3, minus stopwords (sorted).
pub fn query_tokens(query: &str) -> Vec<String> {
    let mut tokens = BTreeSet::new();
    let mut current = String::new();
    for ch in query
        .to_ascii_lowercase()
        .chars()
        .chain(std::iter::once(' '))
    {
        if ch.is_ascii_alphanumeric() {
            current.push(ch);
        } else if !current.is_empty() {
            if current.len() >= 3 && !STOPWORDS.contains(&current.as_str()) {
                tokens.insert(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        }
    }
    tokens.into_iter().collect()
}

/// Lexical seeds: +3 title, +2 description/tags, +1 body per distinct token.
pub fn seeds(concepts: &[Json], query: &str, k: usize) -> Vec<(String, f64)> {
    let tokens = query_tokens(query);
    if tokens.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for concept in concepts {
        let Some(id) = concept.get("id").and_then(Json::as_str) else {
            continue;
        };
        let title = concept
            .get("title")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_ascii_lowercase();
        let desc = concept
            .get("description")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_ascii_lowercase();
        let body = concept
            .get("body")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_ascii_lowercase();
        let tags = concept
            .get("tags")
            .and_then(Json::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Json::as_str)
                    .collect::<Vec<_>>()
                    .join(" ")
                    .to_ascii_lowercase()
            })
            .unwrap_or_default();
        let mut score = 0.0_f64;
        for token in &tokens {
            if title.contains(token) {
                score += 3.0;
            }
            if desc.contains(token) || tags.contains(token) {
                score += 2.0;
            }
            if body.contains(token) {
                score += 1.0;
            }
        }
        if score > 0.0 {
            out.push((id.to_string(), score));
        }
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out.truncate(k);
    out
}

/// Python's `round(x, 10)` (banker's rounding at the half).
fn py_round10(value: f64) -> f64 {
    const FACTOR: f64 = 10_000_000_000.0;
    let scaled = value * FACTOR;
    let floor = scaled.floor();
    let fraction = scaled - floor;
    let rounded = if fraction > 0.5 {
        floor + 1.0
    } else if fraction < 0.5 || (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    };
    rounded / FACTOR
}

/// Exact personalized PageRank over the symmetrized link graph. Dangling
/// nodes keep their mass, matching the oracle's isolated-seed behavior.
pub fn ppr(
    nodes: &[String],
    edges: &[(String, String)],
    starts: &[(String, f64)],
    damping: f64,
    tol: f64,
    max_iter: usize,
    k: Option<usize>,
) -> Vec<(String, f64)> {
    let ids: Vec<String> = nodes
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let index: BTreeMap<&str, usize> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();
    let mut seed = vec![0.0; ids.len()];
    for (id, weight) in starts {
        if let Some(&i) = index.get(id.as_str()) {
            seed[i] += *weight;
        }
    }
    let total: f64 = seed.iter().sum();
    if total <= 0.0 {
        return Vec::new();
    }
    for value in &mut seed {
        *value /= total;
    }
    let mut edges_set: BTreeSet<(usize, usize)> = BTreeSet::new();
    for (src, dst) in edges {
        if let (Some(&s), Some(&d)) = (index.get(src.as_str()), index.get(dst.as_str())) {
            if s != d {
                edges_set.insert((s, d));
                edges_set.insert((d, s));
            }
        }
    }
    let ordered: Vec<(usize, usize)> = edges_set.into_iter().collect();
    let mut degree = vec![0usize; ids.len()];
    for (s, _) in &ordered {
        degree[*s] += 1;
    }
    let mut p = seed.clone();
    for _ in 0..max_iter {
        let mut contrib = vec![0.0; ids.len()];
        for (s, d) in &ordered {
            if p[*s] != 0.0 {
                contrib[*d] += p[*s] / degree[*s] as f64;
            }
        }
        let next: Vec<f64> = (0..ids.len())
            .map(|i| {
                (1.0 - damping) * seed[i]
                    + damping * (contrib[i] + if degree[i] == 0 { p[i] } else { 0.0 })
            })
            .collect();
        let delta: f64 = next.iter().zip(&p).map(|(a, b)| (a - b).abs()).sum();
        p = next;
        if delta < tol {
            break;
        }
    }
    let mut rows: Vec<(String, f64)> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.clone(), py_round10(p[i])))
        .filter(|(_, score)| *score > 0.0)
        .collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if let Some(k) = k {
        rows.truncate(k);
    }
    rows
}

pub fn hub_scores(candidates: &[String], edges: &[(String, String)]) -> BTreeMap<String, i64> {
    let wanted: BTreeSet<&str> = candidates.iter().map(String::as_str).collect();
    let mut counts: BTreeMap<String, i64> = BTreeMap::new();
    for (src, dst) in edges {
        if wanted.contains(dst.as_str()) {
            *counts.entry(dst.clone()).or_default() += 1;
        }
        let _ = src;
    }
    counts
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
    let nb: f64 = b.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_uses_rank_not_score_magnitude() {
        let vec = vec![("a".into(), 100.0), ("b".into(), 1.0)];
        let fts = vec![("b".into(), 0.9), ("c".into(), 0.1)];
        let fused = rrf_fuse(&vec, &fts);
        assert_eq!(fused[0].0, "b");
        assert!((fused[0].1 - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-12);
    }

    #[test]
    fn ppr_keeps_isolated_seed_mass() {
        let nodes = vec!["a".into(), "b".into()];
        let rows = ppr(&nodes, &[], &[("a".into(), 1.0)], 0.85, 1e-12, 200, None);
        assert_eq!(rows[0].0, "a");
        assert!((rows[0].1 - 1.0).abs() < 1e-9);
    }
}
