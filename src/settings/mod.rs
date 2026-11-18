//! Settings table and precedence (52 entries land in M1).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingKind {
    Bool,
    Int,
    Float,
    String,
    Path,
    Choice,
    Roots,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingDef {
    pub name: &'static str,
    pub kind: SettingKind,
    pub default: &'static str,
    pub required_mcp: bool,
}

pub const SETTINGS: &[SettingDef] = &[];
