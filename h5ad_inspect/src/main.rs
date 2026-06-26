use std::collections::HashMap;
use std::env;
use std::ffi::CString;
use std::io::Write;
use std::process;

use hdf5_metno::types::{FloatSize, IntSize, TypeDescriptor};
use hdf5_metno::Datatype;
use hdf5_metno_sys::h5d::H5Dread;
use hdf5_metno_sys::h5p::H5P_DEFAULT;
use hdf5_metno_sys::h5s::H5S_ALL;
use hdf5_metno_sys::h5t::{H5T_class_t, H5Tclose, H5Tcreate, H5Tinsert};
use ndarray::s;

// Used to read index from old-format h5ad where obs/var is a compound dataset
#[derive(hdf5_metno::H5Type, Clone)]
#[repr(C)]
struct CompoundIndex {
    index: hdf5_metno::types::VarLenUnicode,
}

fn die(msg: &str) -> ! {
    eprintln!("Error: {}", msg);
    process::exit(1);
}

// Read obs or var index as a Vec<String> in original (unsorted) order.
fn read_group_index(file: &hdf5_metno::File, group_name: &str) -> Vec<String> {
    let loc_type = file
        .loc_type_by_name(group_name)
        .unwrap_or_else(|e| die(&format!("cannot locate '{}': {}", group_name, e)));
    if loc_type == hdf5_metno::LocationType::Group {
        let group = file
            .group(group_name)
            .unwrap_or_else(|e| die(&format!("cannot open group '{}': {}", group_name, e)));
        let index_name = match group.attr("_index") {
            Ok(attr) => match attr.read_scalar::<hdf5_metno::types::VarLenUnicode>() {
                Ok(v) => v.as_str().to_string(),
                Err(_) => "_index".to_string(),
            },
            Err(_) => "_index".to_string(),
        };
        let ds_path = format!("{}/{}", group_name, index_name);
        let ds = file
            .dataset(&ds_path)
            .unwrap_or_else(|e| die(&format!("cannot open '{}': {}", ds_path, e)));
        match ds.read_1d::<hdf5_metno::types::VarLenUnicode>() {
            Ok(arr) => arr.iter().map(|v| v.as_str().to_string()).collect(),
            Err(e) => die(&format!("cannot read index '{}': {}", ds_path, e)),
        }
    } else {
        let ds = file
            .dataset(group_name)
            .unwrap_or_else(|e| die(&format!("cannot open '{}': {}", group_name, e)));
        match ds.read_1d::<CompoundIndex>() {
            Ok(arr) => arr.iter().map(|r| r.index.as_str().to_string()).collect(),
            Err(e) => die(&format!(
                "cannot read compound index '{}': {}",
                group_name, e
            )),
        }
    }
}

fn find_index_position(index: &[String], value: &str) -> usize {
    index
        .iter()
        .position(|v| v == value)
        .unwrap_or_else(|| die(&format!("'{}' not found in index", value)))
}

type StringIter = Box<dyn Iterator<Item = String>>;

// Lazy iterator over the string representations of a typed 1-D dataset.
fn dataset_values(ds: &hdf5_metno::Dataset) -> StringIter {
    let desc = ds
        .dtype()
        .and_then(|d| d.to_descriptor())
        .unwrap_or_else(|e| die(&format!("dtype: {}", e)));
    macro_rules! iter_typed {
        ($T:ty) => {
            Box::new(
                ds.read_1d::<$T>()
                    .unwrap_or_else(|e| die(&format!("read: {}", e)))
                    .into_iter()
                    .map(|v| format!("{}", v)),
            )
        };
    }
    match desc {
        TypeDescriptor::Integer(IntSize::U1) => iter_typed!(i8),
        TypeDescriptor::Integer(IntSize::U2) => iter_typed!(i16),
        TypeDescriptor::Integer(IntSize::U4) => iter_typed!(i32),
        TypeDescriptor::Integer(IntSize::U8) => iter_typed!(i64),
        TypeDescriptor::Unsigned(IntSize::U1) => iter_typed!(u8),
        TypeDescriptor::Unsigned(IntSize::U2) => iter_typed!(u16),
        TypeDescriptor::Unsigned(IntSize::U4) => iter_typed!(u32),
        TypeDescriptor::Unsigned(IntSize::U8) => iter_typed!(u64),
        TypeDescriptor::Float(FloatSize::U4) => iter_typed!(f32),
        TypeDescriptor::Float(FloatSize::U8) => iter_typed!(f64),
        TypeDescriptor::Boolean => iter_typed!(bool),
        TypeDescriptor::VarLenUnicode => Box::new(
            ds.read_1d::<hdf5_metno::types::VarLenUnicode>()
                .unwrap_or_else(|e| die(&format!("read: {}", e)))
                .into_iter()
                .map(|v| v.as_str().to_string()),
        ),
        TypeDescriptor::VarLenAscii => Box::new(
            ds.read_1d::<hdf5_metno::types::VarLenAscii>()
                .unwrap_or_else(|e| die(&format!("read: {}", e)))
                .into_iter()
                .map(|v| v.as_str().to_string()),
        ),
        _ => die("unsupported dtype for export"),
    }
}

// Stream values through: print directly, or count and print sorted by frequency.
fn output_iter(iter: impl Iterator<Item = String>, value_count: bool) {
    if !value_count {
        for v in iter {
            println!("{}", v);
        }
    } else {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for v in iter {
            *counts.entry(v).or_insert(0) += 1;
        }
        let mut pairs: Vec<(String, usize)> = counts.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for (v, c) in pairs {
            println!("{}\t{}", v, c);
        }
    }
}

fn read_string_dataset(ds: &hdf5_metno::Dataset) -> Vec<String> {
    let desc = ds
        .dtype()
        .and_then(|d| d.to_descriptor())
        .unwrap_or_else(|e| die(&format!("dtype: {}", e)));
    match desc {
        TypeDescriptor::VarLenUnicode => ds
            .read_1d::<hdf5_metno::types::VarLenUnicode>()
            .unwrap_or_else(|e| die(&format!("read: {}", e)))
            .iter()
            .map(|v| v.as_str().to_string())
            .collect(),
        TypeDescriptor::VarLenAscii => ds
            .read_1d::<hdf5_metno::types::VarLenAscii>()
            .unwrap_or_else(|e| die(&format!("read: {}", e)))
            .iter()
            .map(|v| v.as_str().to_string())
            .collect(),
        _ => die("unsupported string dataset type"),
    }
}

// Read a boolean mask dataset (stored as bool or 0/1 integer).
fn read_bool_mask(ds: &hdf5_metno::Dataset) -> Vec<bool> {
    let desc = ds
        .dtype()
        .and_then(|d| d.to_descriptor())
        .unwrap_or_else(|e| die(&format!("mask dtype: {}", e)));
    macro_rules! mask_from {
        ($T:ty, $map:expr) => {
            ds.read_1d::<$T>()
                .unwrap_or_else(|e| die(&format!("mask read: {}", e)))
                .iter()
                .map($map)
                .collect()
        };
    }
    match desc {
        TypeDescriptor::Boolean => mask_from!(bool, |&v| v),
        TypeDescriptor::Unsigned(IntSize::U1) => mask_from!(u8, |&v| v != 0),
        TypeDescriptor::Integer(IntSize::U1) => mask_from!(i8, |&v| v != 0),
        _ => die("unsupported mask dtype"),
    }
}

fn collect_dataset_values(ds: &hdf5_metno::Dataset) -> Vec<String> {
    dataset_values(ds).collect()
}

// Replace masked entries with "NA" (pandas NA sentinel).
fn apply_mask(values: Vec<String>, mask: Option<Vec<bool>>) -> Vec<String> {
    match mask {
        Some(m) => values
            .into_iter()
            .zip(m.into_iter())
            .map(|(v, na)| if na { "NA".to_string() } else { v })
            .collect(),
        None => values,
    }
}

// Read a "categories" element that may be a plain string dataset or a
// nullable-string-array group (values + mask).
fn read_categories(file: &hdf5_metno::File, cats_path: &str) -> Option<Vec<String>> {
    if let Ok(ds) = file.dataset(cats_path) {
        return Some(read_string_dataset(&ds));
    }
    let values_path = format!("{}/values", cats_path);
    let values_ds = file.dataset(&values_path).ok()?;
    let mut cats = collect_dataset_values(&values_ds);
    let mask_path = format!("{}/mask", cats_path);
    if let Ok(mask_ds) = file.dataset(&mask_path) {
        cats = apply_mask(cats, Some(read_bool_mask(&mask_ds)));
    }
    Some(cats)
}

// Decode categorical codes to their string labels.
// Negative codes (pandas NA sentinel) become "NA".
fn collect_categorical(codes_ds: &hdf5_metno::Dataset, categories: &[String]) -> Vec<String> {
    let desc = codes_ds
        .dtype()
        .and_then(|d| d.to_descriptor())
        .unwrap_or_else(|e| die(&format!("codes dtype: {}", e)));
    macro_rules! collect_signed {
        ($T:ty) => {
            codes_ds
                .read_1d::<$T>()
                .unwrap_or_else(|e| die(&format!("codes read: {}", e)))
                .iter()
                .map(|&c| {
                    if c < 0 {
                        "NA".to_string()
                    } else {
                        let idx = c as usize;
                        if idx < categories.len() {
                            categories[idx].clone()
                        } else {
                            die(&format!("code {} out of range", c))
                        }
                    }
                })
                .collect()
        };
    }
    macro_rules! collect_unsigned {
        ($T:ty) => {
            codes_ds
                .read_1d::<$T>()
                .unwrap_or_else(|e| die(&format!("codes read: {}", e)))
                .iter()
                .map(|&c| {
                    let idx = c as usize;
                    if idx < categories.len() {
                        categories[idx].clone()
                    } else {
                        die(&format!("code {} out of range", c))
                    }
                })
                .collect()
        };
    }
    match desc {
        TypeDescriptor::Integer(IntSize::U1) => collect_signed!(i8),
        TypeDescriptor::Integer(IntSize::U2) => collect_signed!(i16),
        TypeDescriptor::Integer(IntSize::U4) => collect_signed!(i32),
        TypeDescriptor::Integer(IntSize::U8) => collect_signed!(i64),
        TypeDescriptor::Unsigned(IntSize::U1) => collect_unsigned!(u8),
        TypeDescriptor::Unsigned(IntSize::U2) => collect_unsigned!(u16),
        TypeDescriptor::Unsigned(IntSize::U4) => collect_unsigned!(u32),
        TypeDescriptor::Unsigned(IntSize::U8) => collect_unsigned!(u64),
        _ => die("unsupported categorical codes type"),
    }
}

// Export a named field from a compound dataset (old-style obs/var).
// Uses raw HDF5 partial compound read to avoid needing compile-time field names.
// Categorical integer fields are decoded via uns/<col>_categories if present.
fn export_compound_dataset_field(
    file: &hdf5_metno::File,
    ds: &hdf5_metno::Dataset,
    col_name: &str,
) {
    let file_dtype = ds.dtype().unwrap_or_else(|e| die(&format!("dtype: {}", e)));
    let desc = file_dtype
        .to_descriptor()
        .unwrap_or_else(|e| die(&format!("desc: {}", e)));
    let ct = match desc {
        TypeDescriptor::Compound(ref c) => c,
        _ => die("not a compound dataset"),
    };

    let field = ct
        .fields
        .iter()
        .find(|f| f.name == col_name)
        .unwrap_or_else(|| {
            die(&format!(
                "field '{}' not found in compound dataset",
                col_name
            ))
        });

    let n_rows = ds.shape()[0];

    // Build a single-field compound memory type using the field's type.
    let field_dt = Datatype::from_descriptor(&field.ty)
        .unwrap_or_else(|e| die(&format!("field dtype: {}", e)));
    let field_size = field_dt.size();

    let field_name_c =
        CString::new(col_name).unwrap_or_else(|_| die("field name contains NUL byte"));

    let mem_type_id = unsafe {
        let t = H5Tcreate(H5T_class_t::H5T_COMPOUND, field_size);
        H5Tinsert(t, field_name_c.as_ptr(), 0, field_dt.id());
        t
    };

    let mut buf = vec![0u8; n_rows * field_size];

    unsafe {
        H5Dread(
            ds.id(),
            mem_type_id,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            buf.as_mut_ptr().cast(),
        );
        H5Tclose(mem_type_id);
    }

    // Check for old-style categories in uns/<col>_categories.
    let cats_uns_path = format!("uns/{}_categories", col_name);
    let categories: Option<Vec<String>> = file
        .dataset(&cats_uns_path)
        .ok()
        .map(|cats_ds| read_string_dataset(&cats_ds));

    // Helper: print an integer code, resolving via categories if available.
    macro_rules! print_int_or_cat {
        ($val:expr) => {
            if let Some(ref cats) = categories {
                let idx = $val as usize;
                if idx < cats.len() {
                    println!("{}", cats[idx]);
                } else {
                    println!("NA");
                }
            } else {
                println!("{}", $val);
            }
        };
    }

    // Parse and print field values.
    match &field.ty {
        TypeDescriptor::Integer(IntSize::U1) => {
            for i in 0..n_rows {
                let v = i8::from_ne_bytes([buf[i]]);
                if categories.is_some() && v < 0 {
                    println!("NA");
                } else {
                    print_int_or_cat!(v);
                }
            }
        }
        TypeDescriptor::Integer(IntSize::U2) => {
            for i in 0..n_rows {
                let v = i16::from_ne_bytes(buf[i * 2..(i + 1) * 2].try_into().unwrap());
                if categories.is_some() && v < 0 {
                    println!("NA");
                } else {
                    print_int_or_cat!(v);
                }
            }
        }
        TypeDescriptor::Integer(IntSize::U4) => {
            for i in 0..n_rows {
                let v = i32::from_ne_bytes(buf[i * 4..(i + 1) * 4].try_into().unwrap());
                if categories.is_some() && v < 0 {
                    println!("NA");
                } else {
                    print_int_or_cat!(v);
                }
            }
        }
        TypeDescriptor::Integer(IntSize::U8) => {
            for i in 0..n_rows {
                let v = i64::from_ne_bytes(buf[i * 8..(i + 1) * 8].try_into().unwrap());
                if categories.is_some() && v < 0 {
                    println!("NA");
                } else {
                    print_int_or_cat!(v);
                }
            }
        }
        TypeDescriptor::Unsigned(IntSize::U1) => {
            for i in 0..n_rows {
                print_int_or_cat!(buf[i] as u8);
            }
        }
        TypeDescriptor::Unsigned(IntSize::U2) => {
            for i in 0..n_rows {
                let v = u16::from_ne_bytes(buf[i * 2..(i + 1) * 2].try_into().unwrap());
                print_int_or_cat!(v);
            }
        }
        TypeDescriptor::Unsigned(IntSize::U4) => {
            for i in 0..n_rows {
                let v = u32::from_ne_bytes(buf[i * 4..(i + 1) * 4].try_into().unwrap());
                print_int_or_cat!(v);
            }
        }
        TypeDescriptor::Unsigned(IntSize::U8) => {
            for i in 0..n_rows {
                let v = u64::from_ne_bytes(buf[i * 8..(i + 1) * 8].try_into().unwrap());
                print_int_or_cat!(v);
            }
        }
        TypeDescriptor::Float(FloatSize::U4) => {
            for i in 0..n_rows {
                println!(
                    "{}",
                    f32::from_ne_bytes(buf[i * 4..(i + 1) * 4].try_into().unwrap())
                );
            }
        }
        TypeDescriptor::Float(FloatSize::U8) => {
            for i in 0..n_rows {
                println!(
                    "{}",
                    f64::from_ne_bytes(buf[i * 8..(i + 1) * 8].try_into().unwrap())
                );
            }
        }
        _ => die(&format!(
            "unsupported compound field type for '{}': {:?}",
            col_name, field.ty
        )),
    }
}

fn export_obs_var_categories(file: &hdf5_metno::File, group_name: &str, col_name: &str) {
    let loc_type = file
        .loc_type_by_name(group_name)
        .unwrap_or_else(|e| die(&format!("cannot locate '{}': {}", group_name, e)));

    if loc_type != hdf5_metno::LocationType::Group {
        // Old compound dataset — categories live in uns/<col>_categories.
        let cats_path = format!("uns/{}_categories", col_name);
        let cats_ds = file.dataset(&cats_path).unwrap_or_else(|_| {
            die(&format!(
                "no categories found for '{}' (tried {})",
                col_name, cats_path
            ))
        });
        for v in read_string_dataset(&cats_ds) {
            println!("{}", v);
        }
        return;
    }

    let col_path = format!("{}/{}", group_name, col_name);
    let col_loc = file.loc_type_by_name(&col_path).unwrap_or_else(|_| {
        die(&format!(
            "column '{}' not found in '{}'",
            col_name, group_name
        ))
    });

    if col_loc == hdf5_metno::LocationType::Group {
        // New-style: group contains categories dataset (or nullable-string-array group).
        let cats_path = format!("{}/categories", col_path);
        match read_categories(file, &cats_path) {
            Some(cats) => {
                for v in cats {
                    println!("{}", v);
                }
            }
            None => die(&format!(
                "'{}' is not categorical (no categories dataset)",
                col_path
            )),
        }
        return;
    }

    // Old-style: flat codes dataset + <group>/__categories/<col>.
    let cats_path = format!("{}/__categories/{}", group_name, col_name);
    let cats_ds = file.dataset(&cats_path).unwrap_or_else(|_| {
        die(&format!(
            "'{}' is not categorical (no __categories entry)",
            col_name
        ))
    });
    for v in read_string_dataset(&cats_ds) {
        println!("{}", v);
    }
}

fn export_obs_var_column(file: &hdf5_metno::File, group_name: &str, col_name: &str) {
    let loc_type = file
        .loc_type_by_name(group_name)
        .unwrap_or_else(|e| die(&format!("cannot locate '{}': {}", group_name, e)));

    if loc_type != hdf5_metno::LocationType::Group {
        // Old-style: obs/var is a single compound dataset.
        let ds = file
            .dataset(group_name)
            .unwrap_or_else(|e| die(&format!("cannot open '{}': {}", group_name, e)));
        export_compound_dataset_field(file, &ds, col_name);
        return;
    }

    let col_path = format!("{}/{}", group_name, col_name);

    // Check if the column is itself a group (new-style h5ad categorical/string storage).
    let col_loc = file.loc_type_by_name(&col_path).unwrap_or_else(|_| {
        die(&format!(
            "column '{}' not found in '{}'",
            col_name, group_name
        ))
    });

    if col_loc == hdf5_metno::LocationType::Group {
        // New-style: group contains `categories` + `codes` (categorical),
        // or `values` + `mask` (nullable integer/string/boolean).
        let codes_path = format!("{}/codes", col_path);
        let cats_path = format!("{}/categories", col_path);
        let values_path = format!("{}/values", col_path);
        let mask_path = format!("{}/mask", col_path);

        // Categorical column: codes + categories (categories may be a plain
        // string dataset or a nullable-string-array group).
        if let Ok(codes_ds) = file.dataset(&codes_path) {
            let categories = read_categories(file, &cats_path).unwrap_or_else(|| {
                die(&format!(
                    "categorical column '{}' has no categories",
                    col_name
                ))
            });
            output_iter(
                collect_categorical(&codes_ds, &categories).into_iter(),
                false,
            );
            return;
        }

        // Nullable column (integer/string/boolean): values + mask.
        if let Ok(values_ds) = file.dataset(&values_path) {
            let mask = file.dataset(&mask_path).ok().map(|m| read_bool_mask(&m));
            output_iter(
                apply_mask(collect_dataset_values(&values_ds), mask).into_iter(),
                false,
            );
            return;
        }

        die(&format!(
            "unrecognised group structure for column '{}' in '{}'",
            col_name, group_name
        ));
    }

    // Column is a direct dataset.
    let col_ds = file
        .dataset(&col_path)
        .unwrap_or_else(|e| die(&format!("cannot open '{}': {}", col_path, e)));

    // Old-style categorical: flat codes dataset + __categories/<col>
    let cats_path_old = format!("{}/__categories/{}", group_name, col_name);
    if let Ok(cats_ds) = file.dataset(&cats_path_old) {
        let categories = read_string_dataset(&cats_ds);
        output_iter(collect_categorical(&col_ds, &categories).into_iter(), false);
    } else {
        output_iter(dataset_values(&col_ds), false);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Column encoding classification
// ──────────────────────────────────────────────────────────────────────────────

// Classify a column's h5ad encoding as ``"categorical"``, ``"bool"``, or
// ``"numeric"``, matching the h5py-based reference: a child group whose
// ``encoding-type`` attribute is ``"categorical"`` is categorical; a child
// bool dataset is bool; everything else is numeric. For categorical columns
// the in-order category labels are returned alongside.
fn col_encoding(
    file: &hdf5_metno::File,
    group_name: &str,
    col_name: &str,
) -> (String, Option<Vec<String>>) {
    let group_loc = match file.loc_type_by_name(group_name) {
        Ok(t) => t,
        Err(_) => return ("numeric".to_string(), None),
    };

    if group_loc == hdf5_metno::LocationType::Group {
        let col_path = format!("{}/{}", group_name, col_name);
        match file.loc_type_by_name(&col_path) {
            Ok(hdf5_metno::LocationType::Group) => {
                let is_cat = file
                    .group(&col_path)
                    .ok()
                    .and_then(|g| g.attr("encoding-type").ok())
                    .and_then(|a| a.read_scalar::<hdf5_metno::types::VarLenUnicode>().ok())
                    .map(|v| v.as_str() == "categorical")
                    .unwrap_or(false);
                if is_cat {
                    let cats = read_categories(file, &format!("{}/categories", col_path))
                        .unwrap_or_default();
                    ("categorical".to_string(), Some(cats))
                } else {
                    ("numeric".to_string(), None)
                }
            }
            Ok(hdf5_metno::LocationType::Dataset) => {
                let is_bool = file
                    .dataset(&col_path)
                    .ok()
                    .and_then(|d| d.dtype().ok())
                    .and_then(|t| t.to_descriptor().ok())
                    .map(|d| matches!(d, TypeDescriptor::Boolean))
                    .unwrap_or(false);
                if is_bool {
                    ("bool".to_string(), None)
                } else {
                    ("numeric".to_string(), None)
                }
            }
            _ => ("numeric".to_string(), None),
        }
    } else {
        // Old-style compound dataset: classify via the field's type.
        let is_bool = file
            .dataset(group_name)
            .ok()
            .and_then(|d| d.dtype().ok())
            .and_then(|t| t.to_descriptor().ok())
            .and_then(|desc| match desc {
                TypeDescriptor::Compound(ct) => ct
                    .fields
                    .iter()
                    .find(|f| f.name == col_name)
                    .map(|f| matches!(f.ty, TypeDescriptor::Boolean)),
                _ => None,
            })
            .unwrap_or(false);
        if is_bool {
            ("bool".to_string(), None)
        } else {
            ("numeric".to_string(), None)
        }
    }
}

fn json_escape_into(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn print_encoding_json(encoding: &str, categories: Option<&[String]>) {
    let mut s = String::from("{\"encoding\":");
    json_escape_into(encoding, &mut s);
    if let Some(cats) = categories {
        s.push_str(",\"categories\":[");
        for (i, c) in cats.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            json_escape_into(c, &mut s);
        }
        s.push(']');
    }
    s.push('}');
    println!("{}", s);
}

// ──────────────────────────────────────────────────────────────────────────────
// X matrix helpers
// ──────────────────────────────────────────────────────────────────────────────

fn x_sparse_format(file: &hdf5_metno::File, x_path: &str) -> Option<String> {
    let group = file.group(x_path).ok()?;
    for attr_name in &["encoding-type", "h5sparse_format"] {
        if let Ok(attr) = group.attr(attr_name) {
            if let Ok(v) = attr.read_scalar::<hdf5_metno::types::VarLenUnicode>() {
                let s = v.as_str().to_lowercase();
                if s.contains("csr") {
                    return Some("csr".to_string());
                }
                if s.contains("csc") {
                    return Some("csc".to_string());
                }
            }
        }
    }
    None
}

fn read_indptr(file: &hdf5_metno::File, x_path: &str) -> Vec<usize> {
    let path = format!("{}/indptr", x_path);
    let ds = file
        .dataset(&path)
        .unwrap_or_else(|e| die(&format!("{}: {}", path, e)));
    let desc = ds
        .dtype()
        .and_then(|d| d.to_descriptor())
        .unwrap_or_else(|e| die(&format!("{} dtype: {}", path, e)));
    macro_rules! read_as_usize {
        ($T:ty) => {
            ds.read_1d::<$T>()
                .unwrap_or_else(|e| die(&format!("{} read: {}", path, e)))
                .iter()
                .map(|&v| v as usize)
                .collect()
        };
    }
    match desc {
        TypeDescriptor::Integer(IntSize::U4) => read_as_usize!(i32),
        TypeDescriptor::Integer(IntSize::U8) => read_as_usize!(i64),
        TypeDescriptor::Unsigned(IntSize::U4) => read_as_usize!(u32),
        TypeDescriptor::Unsigned(IntSize::U8) => read_as_usize!(u64),
        _ => die(&format!("unsupported {} dtype", path)),
    }
}

fn read_indices_slice(
    file: &hdf5_metno::File,
    x_path: &str,
    start: usize,
    end: usize,
) -> Vec<usize> {
    let path = format!("{}/indices", x_path);
    let ds = file
        .dataset(&path)
        .unwrap_or_else(|e| die(&format!("{}: {}", path, e)));
    let desc = ds
        .dtype()
        .and_then(|d| d.to_descriptor())
        .unwrap_or_else(|e| die(&format!("{} dtype: {}", path, e)));
    macro_rules! read_slice_usize {
        ($T:ty) => {
            ds.read_slice_1d::<$T, _>(s![start..end])
                .unwrap_or_else(|e| die(&format!("{} read: {}", path, e)))
                .iter()
                .map(|&v| v as usize)
                .collect()
        };
    }
    match desc {
        TypeDescriptor::Integer(IntSize::U4) => read_slice_usize!(i32),
        TypeDescriptor::Integer(IntSize::U8) => read_slice_usize!(i64),
        TypeDescriptor::Unsigned(IntSize::U4) => read_slice_usize!(u32),
        TypeDescriptor::Unsigned(IntSize::U8) => read_slice_usize!(u64),
        _ => die(&format!("unsupported {} dtype", path)),
    }
}

fn read_data_slice(file: &hdf5_metno::File, x_path: &str, start: usize, end: usize) -> Vec<f64> {
    let path = format!("{}/data", x_path);
    let ds = file
        .dataset(&path)
        .unwrap_or_else(|e| die(&format!("{}: {}", path, e)));
    let desc = ds
        .dtype()
        .and_then(|d| d.to_descriptor())
        .unwrap_or_else(|e| die(&format!("{} dtype: {}", path, e)));
    macro_rules! read_slice_f64 {
        ($T:ty) => {
            ds.read_slice_1d::<$T, _>(s![start..end])
                .unwrap_or_else(|e| die(&format!("{} read: {}", path, e)))
                .iter()
                .map(|&v| v as f64)
                .collect()
        };
    }
    match desc {
        TypeDescriptor::Float(FloatSize::U4) => read_slice_f64!(f32),
        TypeDescriptor::Float(FloatSize::U8) => read_slice_f64!(f64),
        TypeDescriptor::Integer(IntSize::U4) => read_slice_f64!(i32),
        TypeDescriptor::Integer(IntSize::U8) => read_slice_f64!(i64),
        TypeDescriptor::Unsigned(IntSize::U4) => read_slice_f64!(u32),
        _ => die(&format!("unsupported {} dtype", path)),
    }
}

fn write_f64_values(vals: &[f64], binary: bool) {
    if binary {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        for &v in vals {
            out.write_all(&v.to_le_bytes())
                .unwrap_or_else(|e| die(&format!("write: {}", e)));
        }
    } else {
        for &v in vals {
            println!("{}", v);
        }
    }
}

// Export a CSR row: O(nnz_row) efficient
fn export_x_csr_row(
    file: &hdf5_metno::File,
    x_path: &str,
    row_idx: usize,
    n_cols: usize,
    binary: bool,
) {
    let indptr = read_indptr(file, x_path);
    let start = indptr[row_idx];
    let end = indptr[row_idx + 1];
    let indices = read_indices_slice(file, x_path, start, end);
    let data = read_data_slice(file, x_path, start, end);
    let mut vals = vec![0.0f64; n_cols];
    for (k, &col) in indices.iter().enumerate() {
        vals[col] = data[k];
    }
    write_f64_values(&vals, binary);
}

// Export a CSC column: O(nnz_col) efficient
fn export_x_csc_col(
    file: &hdf5_metno::File,
    x_path: &str,
    col_idx: usize,
    n_rows: usize,
    binary: bool,
) {
    let indptr = read_indptr(file, x_path);
    let start = indptr[col_idx];
    let end = indptr[col_idx + 1];
    let indices = read_indices_slice(file, x_path, start, end);
    let data = read_data_slice(file, x_path, start, end);
    let mut vals = vec![0.0f64; n_rows];
    for (k, &row) in indices.iter().enumerate() {
        vals[row] = data[k];
    }
    write_f64_values(&vals, binary);
}

// Export a CSR column (suboptimal: scans all rows)
fn export_x_csr_col(
    file: &hdf5_metno::File,
    x_path: &str,
    col_idx: usize,
    n_rows: usize,
    binary: bool,
) {
    let indptr = read_indptr(file, x_path);
    // Read full indices + data — no way to avoid it for CSR column access
    let indices = read_indices_slice(file, x_path, 0, *indptr.last().unwrap_or(&0));
    let data = read_data_slice(file, x_path, 0, *indptr.last().unwrap_or(&0));
    let mut vals = vec![0.0f64; n_rows];
    for row in 0..n_rows {
        let start = indptr[row];
        let end = indptr[row + 1];
        for k in start..end {
            if indices[k] == col_idx {
                vals[row] = data[k];
                break;
            }
        }
    }
    write_f64_values(&vals, binary);
}

fn export_x_obssum(file: &hdf5_metno::File, x_path: &str, binary: bool) {
    let n_obs = read_group_index(file, "obs").len();
    let x_loc = file
        .loc_type_by_name(x_path)
        .unwrap_or_else(|e| die(&format!("cannot locate {}: {}", x_path, e)));
    if x_loc == hdf5_metno::LocationType::Group {
        let fmt = x_sparse_format(file, x_path)
            .unwrap_or_else(|| die(&format!("cannot determine {} sparse format", x_path)));
        let indptr = read_indptr(file, x_path);
        let nnz = *indptr.last().unwrap_or(&0);
        let sums = match fmt.as_str() {
            "csr" => {
                let data = read_data_slice(file, x_path, 0, nnz);
                (0..n_obs)
                    .map(|i| data[indptr[i]..indptr[i + 1]].iter().sum::<f64>())
                    .collect::<Vec<f64>>()
            }
            "csc" => {
                let data = read_data_slice(file, x_path, 0, nnz);
                let indices = read_indices_slice(file, x_path, 0, nnz);
                let mut sums = vec![0.0f64; n_obs];
                for (k, &row) in indices.iter().enumerate() {
                    sums[row] += data[k];
                }
                sums
            }
            _ => die(&format!("unknown sparse format: {}", fmt)),
        };
        write_f64_values(&sums, binary);
    } else {
        let ds = file
            .dataset(x_path)
            .unwrap_or_else(|e| die(&format!("{} dataset: {}", x_path, e)));
        let desc = ds
            .dtype()
            .and_then(|d| d.to_descriptor())
            .unwrap_or_else(|e| die(&format!("{} dtype: {}", x_path, e)));
        macro_rules! dense_obssum {
            ($T:ty) => {{
                (0..n_obs)
                    .map(|i| {
                        ds.read_slice_1d::<$T, _>(s![i, ..])
                            .unwrap_or_else(|e| die(&format!("X row read: {}", e)))
                            .iter()
                            .map(|&v| v as f64)
                            .sum::<f64>()
                    })
                    .collect::<Vec<f64>>()
            }};
        }
        let sums = match desc {
            TypeDescriptor::Float(FloatSize::U4) => dense_obssum!(f32),
            TypeDescriptor::Float(FloatSize::U8) => dense_obssum!(f64),
            TypeDescriptor::Integer(IntSize::U4) => dense_obssum!(i32),
            TypeDescriptor::Unsigned(IntSize::U4) => dense_obssum!(u32),
            _ => die("unsupported dense X dtype"),
        };
        write_f64_values(&sums, binary);
    }
}

fn export_x_varsum(file: &hdf5_metno::File, x_path: &str, binary: bool) {
    let n_var = read_group_index(file, "var").len();
    let x_loc = file
        .loc_type_by_name(x_path)
        .unwrap_or_else(|e| die(&format!("cannot locate {}: {}", x_path, e)));
    if x_loc == hdf5_metno::LocationType::Group {
        let fmt = x_sparse_format(file, x_path)
            .unwrap_or_else(|| die(&format!("cannot determine {} sparse format", x_path)));
        let indptr = read_indptr(file, x_path);
        let nnz = *indptr.last().unwrap_or(&0);
        let sums = match fmt.as_str() {
            "csr" => {
                let data = read_data_slice(file, x_path, 0, nnz);
                let indices = read_indices_slice(file, x_path, 0, nnz);
                let mut sums = vec![0.0f64; n_var];
                for (k, &col) in indices.iter().enumerate() {
                    sums[col] += data[k];
                }
                sums
            }
            "csc" => {
                let data = read_data_slice(file, x_path, 0, nnz);
                (0..n_var)
                    .map(|j| data[indptr[j]..indptr[j + 1]].iter().sum::<f64>())
                    .collect::<Vec<f64>>()
            }
            _ => die(&format!("unknown sparse format: {}", fmt)),
        };
        write_f64_values(&sums, binary);
    } else {
        let ds = file
            .dataset(x_path)
            .unwrap_or_else(|e| die(&format!("{} dataset: {}", x_path, e)));
        let desc = ds
            .dtype()
            .and_then(|d| d.to_descriptor())
            .unwrap_or_else(|e| die(&format!("{} dtype: {}", x_path, e)));
        macro_rules! dense_varsum {
            ($T:ty) => {{
                (0..n_var)
                    .map(|j| {
                        ds.read_slice_1d::<$T, _>(s![.., j])
                            .unwrap_or_else(|e| die(&format!("X col read: {}", e)))
                            .iter()
                            .map(|&v| v as f64)
                            .sum::<f64>()
                    })
                    .collect::<Vec<f64>>()
            }};
        }
        let sums = match desc {
            TypeDescriptor::Float(FloatSize::U4) => dense_varsum!(f32),
            TypeDescriptor::Float(FloatSize::U8) => dense_varsum!(f64),
            TypeDescriptor::Integer(IntSize::U4) => dense_varsum!(i32),
            TypeDescriptor::Unsigned(IntSize::U4) => dense_varsum!(u32),
            _ => die("unsupported dense X dtype"),
        };
        write_f64_values(&sums, binary);
    }
}

// Export a CSC row (suboptimal: scans all columns)
fn export_x_csc_row(
    file: &hdf5_metno::File,
    x_path: &str,
    row_idx: usize,
    n_cols: usize,
    binary: bool,
) {
    let indptr = read_indptr(file, x_path);
    let indices = read_indices_slice(file, x_path, 0, *indptr.last().unwrap_or(&0));
    let data = read_data_slice(file, x_path, 0, *indptr.last().unwrap_or(&0));
    let mut vals = vec![0.0f64; n_cols];
    for col in 0..n_cols {
        let start = indptr[col];
        let end = indptr[col + 1];
        for k in start..end {
            if indices[k] == row_idx {
                vals[col] = data[k];
                break;
            }
        }
    }
    write_f64_values(&vals, binary);
}

fn export_x_row(file: &hdf5_metno::File, x_path: &str, obs_idx_val: &str, binary: bool) {
    let obs_index = read_group_index(file, "obs");
    let var_index = read_group_index(file, "var");
    let obs_pos = find_index_position(&obs_index, obs_idx_val);
    let n_var = var_index.len();

    let x_loc = file
        .loc_type_by_name(x_path)
        .unwrap_or_else(|e| die(&format!("cannot locate {}: {}", x_path, e)));

    if x_loc == hdf5_metno::LocationType::Group {
        let fmt = x_sparse_format(file, x_path)
            .unwrap_or_else(|| die(&format!("cannot determine {} sparse format", x_path)));
        match fmt.as_str() {
            "csr" => export_x_csr_row(file, x_path, obs_pos, n_var, binary),
            "csc" => export_x_csc_row(file, x_path, obs_pos, n_var, binary),
            _ => die(&format!("unknown sparse format: {}", fmt)),
        }
    } else {
        // Dense 2-D dataset
        let ds = file
            .dataset(x_path)
            .unwrap_or_else(|e| die(&format!("{} dataset: {}", x_path, e)));
        let desc = ds
            .dtype()
            .and_then(|d| d.to_descriptor())
            .unwrap_or_else(|e| die(&format!("{} dtype: {}", x_path, e)));
        macro_rules! collect_dense_row {
            ($T:ty) => {
                ds.read_slice_1d::<$T, _>(s![obs_pos, ..])
                    .unwrap_or_else(|e| die(&format!("X row read: {}", e)))
                    .iter()
                    .map(|&v| v as f64)
                    .collect::<Vec<f64>>()
            };
        }
        let vals = match desc {
            TypeDescriptor::Float(FloatSize::U4) => collect_dense_row!(f32),
            TypeDescriptor::Float(FloatSize::U8) => collect_dense_row!(f64),
            TypeDescriptor::Integer(IntSize::U4) => collect_dense_row!(i32),
            TypeDescriptor::Unsigned(IntSize::U4) => collect_dense_row!(u32),
            _ => die("unsupported dense X dtype"),
        };
        write_f64_values(&vals, binary);
    }
}

fn export_x_column(file: &hdf5_metno::File, x_path: &str, var_idx_val: &str, binary: bool) {
    let obs_index = read_group_index(file, "obs");
    let var_index = read_group_index(file, "var");
    let var_pos = find_index_position(&var_index, var_idx_val);
    let n_obs = obs_index.len();

    let x_loc = file
        .loc_type_by_name(x_path)
        .unwrap_or_else(|e| die(&format!("cannot locate {}: {}", x_path, e)));

    if x_loc == hdf5_metno::LocationType::Group {
        let fmt = x_sparse_format(file, x_path)
            .unwrap_or_else(|| die(&format!("cannot determine {} sparse format", x_path)));
        match fmt.as_str() {
            "csr" => export_x_csr_col(file, x_path, var_pos, n_obs, binary),
            "csc" => export_x_csc_col(file, x_path, var_pos, n_obs, binary),
            _ => die(&format!("unknown sparse format: {}", fmt)),
        }
    } else {
        let ds = file
            .dataset(x_path)
            .unwrap_or_else(|e| die(&format!("{} dataset: {}", x_path, e)));
        let desc = ds
            .dtype()
            .and_then(|d| d.to_descriptor())
            .unwrap_or_else(|e| die(&format!("{} dtype: {}", x_path, e)));
        macro_rules! collect_dense_col {
            ($T:ty) => {
                ds.read_slice_1d::<$T, _>(s![.., var_pos])
                    .unwrap_or_else(|e| die(&format!("X col read: {}", e)))
                    .iter()
                    .map(|&v| v as f64)
                    .collect::<Vec<f64>>()
            };
        }
        let vals = match desc {
            TypeDescriptor::Float(FloatSize::U4) => collect_dense_col!(f32),
            TypeDescriptor::Float(FloatSize::U8) => collect_dense_col!(f64),
            TypeDescriptor::Integer(IntSize::U4) => collect_dense_col!(i32),
            TypeDescriptor::Unsigned(IntSize::U4) => collect_dense_col!(u32),
            _ => die("unsupported dense X dtype"),
        };
        write_f64_values(&vals, binary);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// obsm (multi-dimensional obs annotations)
// ──────────────────────────────────────────────────────────────────────────────

// Export a 2-D obsm dataset (e.g. X_pca, X_umap) in obs order.
// Text: one tab-separated line per obs, n_components columns per line.
// --binary: row-major little-endian float64 (n_obs * n_components values).
fn export_obsm(file: &hdf5_metno::File, name: &str, binary: bool) {
    let ds_path = format!("obsm/{}", name);
    let ds = file
        .dataset(&ds_path)
        .unwrap_or_else(|e| die(&format!("cannot open '{}': {}", ds_path, e)));

    let shape = ds.shape();
    if shape.len() != 2 {
        die(&format!(
            "'{}' is not 2-D (got {} dimension{})",
            ds_path,
            shape.len(),
            if shape.len() == 1 { "" } else { "s" }
        ));
    }
    let n_obs = shape[0];
    let n_components = shape[1];

    let desc = ds
        .dtype()
        .and_then(|d| d.to_descriptor())
        .unwrap_or_else(|e| die(&format!("obsm dtype: {}", e)));

    macro_rules! emit {
        ($T:ty) => {{
            let arr = ds
                .read_2d::<$T>()
                .unwrap_or_else(|e| die(&format!("obsm read: {}", e)));
            if binary {
                let stdout = std::io::stdout();
                let mut out = stdout.lock();
                for i in 0..n_obs {
                    for j in 0..n_components {
                        let v = arr[[i, j]] as f64;
                        out.write_all(&v.to_le_bytes())
                            .unwrap_or_else(|e| die(&format!("write: {}", e)));
                    }
                }
            } else {
                for i in 0..n_obs {
                    let mut line = String::with_capacity(n_components * 8);
                    for j in 0..n_components {
                        if j > 0 {
                            line.push('\t');
                        }
                        line.push_str(&format!("{}", arr[[i, j]]));
                    }
                    println!("{}", line);
                }
            }
        }};
    }

    match desc {
        TypeDescriptor::Float(FloatSize::U4) => emit!(f32),
        TypeDescriptor::Float(FloatSize::U8) => emit!(f64),
        TypeDescriptor::Integer(IntSize::U1) => emit!(i8),
        TypeDescriptor::Integer(IntSize::U2) => emit!(i16),
        TypeDescriptor::Integer(IntSize::U4) => emit!(i32),
        TypeDescriptor::Integer(IntSize::U8) => emit!(i64),
        TypeDescriptor::Unsigned(IntSize::U1) => emit!(u8),
        TypeDescriptor::Unsigned(IntSize::U2) => emit!(u16),
        TypeDescriptor::Unsigned(IntSize::U4) => emit!(u32),
        TypeDescriptor::Unsigned(IntSize::U8) => emit!(u64),
        _ => die(&format!("unsupported dtype for obsm '{}'", name)),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Full X matrix export — CSR / CSC as streamed NumPy .npz archives
// ──────────────────────────────────────────────────────────────────────────────

// Minimal CRC-32 (IEEE 802.3, reflected poly 0xEDB88320) for ZIP entries.
fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for i in 0..256u32 {
        let mut c = i;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB88320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        table[i as usize] = c;
    }
    let mut crc = 0xFFFFFFFFu32;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFFFFFF
}

struct ZipEntry {
    name: String,
    bytes: Vec<u8>,
}

// Stream a STORED (uncompressed) ZIP archive to `w`. Each entry's payload is a
// complete .npy file. Uses 32-bit offsets/sizes (≤4 GiB total archive).
fn write_zip<W: Write>(w: &mut W, entries: &[ZipEntry]) {
    // (name, crc, size, local-header offset) per entry, for the central dir.
    let mut central: Vec<(Vec<u8>, u32, u32, u32)> = Vec::new();
    let mut offset: u32 = 0;

    for e in entries {
        let crc = crc32(&e.bytes);
        let size = e.bytes.len() as u32;
        let name = e.name.as_bytes();
        let mut lh = Vec::with_capacity(30 + name.len());
        lh.extend_from_slice(&0x04034b50u32.to_le_bytes()); // local file header sig
        lh.extend_from_slice(&20u16.to_le_bytes()); // version needed
        lh.extend_from_slice(&0u16.to_le_bytes()); // flags
        lh.extend_from_slice(&0u16.to_le_bytes()); // method (stored)
        lh.extend_from_slice(&0u16.to_le_bytes()); // mod time
        lh.extend_from_slice(&0u16.to_le_bytes()); // mod date
        lh.extend_from_slice(&crc.to_le_bytes());
        lh.extend_from_slice(&size.to_le_bytes()); // compressed size
        lh.extend_from_slice(&size.to_le_bytes()); // uncompressed size
        lh.extend_from_slice(&(name.len() as u16).to_le_bytes());
        lh.extend_from_slice(&0u16.to_le_bytes()); // extra field len
        lh.extend_from_slice(name);
        w.write_all(&lh)
            .unwrap_or_else(|e| die(&format!("write: {}", e)));
        w.write_all(&e.bytes)
            .unwrap_or_else(|e| die(&format!("write: {}", e)));
        central.push((name.to_vec(), crc, size, offset));
        offset = offset.saturating_add(lh.len() as u32).saturating_add(size);
    }

    let cd_start = offset;
    let mut cd_size: u32 = 0;
    for (name, crc, size, lh_offset) in &central {
        let mut h = Vec::with_capacity(46 + name.len());
        h.extend_from_slice(&0x02014b50u32.to_le_bytes()); // central dir header sig
        h.extend_from_slice(&20u16.to_le_bytes()); // version made by
        h.extend_from_slice(&20u16.to_le_bytes()); // version needed
        h.extend_from_slice(&0u16.to_le_bytes()); // flags
        h.extend_from_slice(&0u16.to_le_bytes()); // method
        h.extend_from_slice(&0u16.to_le_bytes()); // mod time
        h.extend_from_slice(&0u16.to_le_bytes()); // mod date
        h.extend_from_slice(&crc.to_le_bytes());
        h.extend_from_slice(&size.to_le_bytes()); // compressed size
        h.extend_from_slice(&size.to_le_bytes()); // uncompressed size
        h.extend_from_slice(&(name.len() as u16).to_le_bytes());
        h.extend_from_slice(&0u16.to_le_bytes()); // extra
        h.extend_from_slice(&0u16.to_le_bytes()); // comment len
        h.extend_from_slice(&0u16.to_le_bytes()); // disk start
        h.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        h.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        h.extend_from_slice(&lh_offset.to_le_bytes()); // local header offset
        h.extend_from_slice(name);
        w.write_all(&h)
            .unwrap_or_else(|e| die(&format!("write: {}", e)));
        cd_size = cd_size.saturating_add(h.len() as u32);
    }

    let mut eocd = Vec::with_capacity(22);
    eocd.extend_from_slice(&0x06054b50u32.to_le_bytes()); // EOCD sig
    eocd.extend_from_slice(&0u16.to_le_bytes()); // this disk
    eocd.extend_from_slice(&0u16.to_le_bytes()); // disk with cd
    eocd.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    eocd.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    eocd.extend_from_slice(&cd_size.to_le_bytes());
    eocd.extend_from_slice(&cd_start.to_le_bytes());
    eocd.extend_from_slice(&0u16.to_le_bytes()); // comment len
    w.write_all(&eocd)
        .unwrap_or_else(|e| die(&format!("write: {}", e)));
}

// Build a NumPy v1.0 .npy header (magic + version + length + padded dict),
// without the data payload.
fn npy_header(descr: &str, shape: &[u64]) -> Vec<u8> {
    let shape_str = match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => {
            let parts: Vec<String> = shape.iter().map(|x| x.to_string()).collect();
            format!("({})", parts.join(", "))
        }
    };
    let dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr, shape_str
    );
    let mut h = dict.into_bytes();
    let mut pad = 1; // reserve the trailing newline
    while (10 + h.len() + pad) % 64 != 0 {
        pad += 1;
    }
    h.resize(h.len() + pad - 1, b' ');
    h.push(b'\n');

    let mut out = Vec::with_capacity(10 + h.len());
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(&h);
    out
}

fn npy_f64(vals: &[f64]) -> Vec<u8> {
    let mut out = npy_header("<f8", &[vals.len() as u64]);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

// Integer indices/indptr are emitted as little-endian int64.
fn npy_i64(vals: &[usize]) -> Vec<u8> {
    let mut out = npy_header("<i8", &[vals.len() as u64]);
    for &v in vals {
        out.extend_from_slice(&(v as i64).to_le_bytes());
    }
    out
}

fn npy_shape2(n_rows: usize, n_cols: usize) -> Vec<u8> {
    let mut out = npy_header("<i8", &[2]);
    out.extend_from_slice(&(n_rows as i64).to_le_bytes());
    out.extend_from_slice(&(n_cols as i64).to_le_bytes());
    out
}

// Read a dense 2-D X dataset into row-major f64.
fn read_dense_x_full(file: &hdf5_metno::File, x_path: &str) -> Vec<f64> {
    let ds = file
        .dataset(x_path)
        .unwrap_or_else(|e| die(&format!("{} dataset: {}", x_path, e)));
    let shape = ds.shape();
    if shape.len() != 2 {
        die(&format!("{} is not a 2-D dataset", x_path));
    }
    let (n_obs, n_var) = (shape[0], shape[1]);
    let desc = ds
        .dtype()
        .and_then(|d| d.to_descriptor())
        .unwrap_or_else(|e| die(&format!("X dtype: {}", e)));
    macro_rules! rd {
        ($T:ty) => {{
            let a = ds
                .read_2d::<$T>()
                .unwrap_or_else(|e| die(&format!("X read: {}", e)));
            let mut v = Vec::with_capacity(n_obs * n_var);
            for i in 0..n_obs {
                for j in 0..n_var {
                    v.push(a[[i, j]] as f64);
                }
            }
            v
        }};
    }
    match desc {
        TypeDescriptor::Float(FloatSize::U4) => rd!(f32),
        TypeDescriptor::Float(FloatSize::U8) => rd!(f64),
        TypeDescriptor::Integer(IntSize::U4) => rd!(i32),
        TypeDescriptor::Unsigned(IntSize::U4) => rd!(u32),
        _ => die("unsupported dense X dtype"),
    }
}

// Transpose CSR arrays of an (n_rows × n_cols) matrix into CSR arrays of its
// (n_cols × n_rows) transpose, via counting sort. The output indices within
// each column are in ascending row order (canonical). Used to derive CSC from
// CSR and vice-versa.
fn transpose_csr(
    n_rows: usize,
    n_cols: usize,
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut counts = vec![0usize; n_cols];
    for &c in indices {
        counts[c] += 1;
    }
    let mut out_indptr = vec![0usize; n_cols + 1];
    for c in 0..n_cols {
        out_indptr[c + 1] = out_indptr[c] + counts[c];
    }
    let nnz = indices.len();
    let mut out_indices = vec![0usize; nnz];
    let mut out_data = vec![0.0f64; nnz];
    let mut pos = out_indptr[..n_cols].to_vec();
    for r in 0..n_rows {
        for k in indptr[r]..indptr[r + 1] {
            let c = indices[k];
            let dest = pos[c];
            out_indices[dest] = r;
            out_data[dest] = data[k];
            pos[c] += 1;
        }
    }
    (out_indptr, out_indices, out_data)
}

// Read X as CSR arrays: indptr has n_obs+1 entries, indices holds column
// indices. A dense X is expanded to a fully-populated CSR; a CSC-stored X is
// transposed once via counting sort.
fn read_x_csr(
    file: &hdf5_metno::File,
    x_path: &str,
    n_obs: usize,
    n_var: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let x_loc = file
        .loc_type_by_name(x_path)
        .unwrap_or_else(|e| die(&format!("cannot locate {}: {}", x_path, e)));
    if x_loc == hdf5_metno::LocationType::Group {
        let fmt = x_sparse_format(file, x_path)
            .unwrap_or_else(|| die(&format!("cannot determine {} sparse format", x_path)));
        let indptr = read_indptr(file, x_path);
        let nnz = *indptr.last().unwrap_or(&0);
        let indices = read_indices_slice(file, x_path, 0, nnz);
        let data = read_data_slice(file, x_path, 0, nnz);
        match fmt.as_str() {
            "csr" => (indptr, indices, data),
            "csc" => transpose_csr(n_var, n_obs, &indptr, &indices, &data),
            _ => die(&format!("unknown sparse format: {}", fmt)),
        }
    } else {
        let dense = read_dense_x_full(file, x_path);
        let mut indptr = Vec::with_capacity(n_obs + 1);
        let mut indices = Vec::with_capacity(n_obs * n_var);
        let mut data = Vec::with_capacity(n_obs * n_var);
        for i in 0..n_obs {
            indptr.push(i * n_var);
            for j in 0..n_var {
                indices.push(j);
                data.push(dense[i * n_var + j]);
            }
        }
        indptr.push(n_obs * n_var);
        (indptr, indices, data)
    }
}

// Read X as CSC arrays: indptr has n_var+1 entries, indices holds row indices.
// CSC arrays of M are exactly the CSR arrays of M^T, so a CSR-stored X is
// transposed once; a dense X is expanded column-major.
fn read_x_csc(
    file: &hdf5_metno::File,
    x_path: &str,
    n_obs: usize,
    n_var: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let x_loc = file
        .loc_type_by_name(x_path)
        .unwrap_or_else(|e| die(&format!("cannot locate {}: {}", x_path, e)));
    if x_loc == hdf5_metno::LocationType::Group {
        let fmt = x_sparse_format(file, x_path)
            .unwrap_or_else(|| die(&format!("cannot determine {} sparse format", x_path)));
        let indptr = read_indptr(file, x_path);
        let nnz = *indptr.last().unwrap_or(&0);
        let indices = read_indices_slice(file, x_path, 0, nnz);
        let data = read_data_slice(file, x_path, 0, nnz);
        match fmt.as_str() {
            "csc" => (indptr, indices, data),
            "csr" => transpose_csr(n_obs, n_var, &indptr, &indices, &data),
            _ => die(&format!("unknown sparse format: {}", fmt)),
        }
    } else {
        let dense = read_dense_x_full(file, x_path);
        let mut indptr = Vec::with_capacity(n_var + 1);
        let mut indices = Vec::with_capacity(n_obs * n_var);
        let mut data = Vec::with_capacity(n_obs * n_var);
        for j in 0..n_var {
            indptr.push(j * n_obs);
            for i in 0..n_obs {
                indices.push(i);
                data.push(dense[i * n_var + j]);
            }
        }
        indptr.push(n_var * n_obs);
        (indptr, indices, data)
    }
}

// Export the full X matrix as a NumPy .npz stream holding a single CSR
// sparse representation: data/indices/indptr/shape. Data is float64; indices
// and indptr are int64. Works whether X is dense, CSR, or CSC on disk.
fn export_matrix_csr(file: &hdf5_metno::File, x_path: &str) {
    let n_obs = read_group_index(file, "obs").len();
    let n_var = read_group_index(file, "var").len();
    let (indptr, indices, data) = read_x_csr(file, x_path, n_obs, n_var);
    let entries = vec![
        ZipEntry {
            name: "csr_data.npy".into(),
            bytes: npy_f64(&data),
        },
        ZipEntry {
            name: "csr_indices.npy".into(),
            bytes: npy_i64(&indices),
        },
        ZipEntry {
            name: "csr_indptr.npy".into(),
            bytes: npy_i64(&indptr),
        },
        ZipEntry {
            name: "csr_shape.npy".into(),
            bytes: npy_shape2(n_obs, n_var),
        },
    ];
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    write_zip(&mut out, &entries);
}

// Export the full X matrix as a NumPy .npz stream holding a single CSC
// sparse representation: data/indices/indptr/shape. See export_matrix_csr.
fn export_matrix_csc(file: &hdf5_metno::File, x_path: &str) {
    let n_obs = read_group_index(file, "obs").len();
    let n_var = read_group_index(file, "var").len();
    let (indptr, indices, data) = read_x_csc(file, x_path, n_obs, n_var);
    let entries = vec![
        ZipEntry {
            name: "csc_data.npy".into(),
            bytes: npy_f64(&data),
        },
        ZipEntry {
            name: "csc_indices.npy".into(),
            bytes: npy_i64(&indices),
        },
        ZipEntry {
            name: "csc_indptr.npy".into(),
            bytes: npy_i64(&indptr),
        },
        ZipEntry {
            name: "csc_shape.npy".into(),
            bytes: npy_shape2(n_obs, n_var),
        },
    ];
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    write_zip(&mut out, &entries);
}

// Write a 1-D dataset of a fixed-size H5Type (f64, i32, …) into `group`.
fn write_h5_1d<T: hdf5_metno::H5Type>(group: &hdf5_metno::Group, name: &str, data: &[T]) {
    group
        .new_dataset::<T>()
        .shape([data.len()])
        .create(name)
        .and_then(|ds| ds.write(data))
        .unwrap_or_else(|e| die(&format!("write {}: {}", name, e)));
}

// Write a 1-D variable-length UTF-8 string dataset into `group`.
fn write_h5_strings(group: &hdf5_metno::Group, name: &str, vals: &[String]) {
    use std::str::FromStr;
    let conv: Vec<hdf5_metno::types::VarLenUnicode> = vals
        .iter()
        .map(|s| {
            hdf5_metno::types::VarLenUnicode::from_str(s)
                .unwrap_or_else(|e| die(&format!("{}: bad string {:?}: {}", name, s, e)))
        })
        .collect();
    group
        .new_dataset::<hdf5_metno::types::VarLenUnicode>()
        .shape([conv.len()])
        .create(name)
        .and_then(|ds| ds.write(&conv[..]))
        .unwrap_or_else(|e| die(&format!("write {}: {}", name, e)));
}

// Export X as a 10x CellRanger v3 .h5 file at `out_path`, readable natively by
// Seurat::Read10X_h5 (no Python/reticulate). 10x stores the matrix as
// features × barcodes in CSC; that layout is byte-for-byte the CSR arrays of
// AnnData's (cells × genes) X (per-cell indptr, gene row-indices), so we reuse
// read_x_csr directly with no extra transpose. obs become barcodes (columns),
// var become features (rows). Index slots are int32 (the dgCMatrix limit).
fn export_matrix_cellranger_v3_hdf5(file: &hdf5_metno::File, x_path: &str, out_path: &str) {
    let barcodes = read_group_index(file, "obs");
    let features = read_group_index(file, "var");
    let n_obs = barcodes.len();
    let n_var = features.len();

    let (indptr, indices, data) = read_x_csr(file, x_path, n_obs, n_var);
    let nnz = data.len();

    let max = i32::MAX as usize;
    if n_obs > max || n_var > max || nnz > max {
        die("matrix too large for CellRanger v3 (int32) layout: dims/nnz exceed 2^31-1");
    }
    let indptr_i32: Vec<i32> = indptr.iter().map(|&v| v as i32).collect();
    let indices_i32: Vec<i32> = indices.iter().map(|&v| v as i32).collect();
    let shape_i32: [i32; 2] = [n_var as i32, n_obs as i32]; // (features, barcodes)

    let out = hdf5_metno::File::create(out_path)
        .unwrap_or_else(|e| die(&format!("create {}: {}", out_path, e)));
    let matrix = out
        .create_group("matrix")
        .unwrap_or_else(|e| die(&format!("create /matrix: {}", e)));

    write_h5_1d(&matrix, "data", &data);
    write_h5_1d(&matrix, "indices", &indices_i32);
    write_h5_1d(&matrix, "indptr", &indptr_i32);
    write_h5_1d(&matrix, "shape", &shape_i32[..]);
    write_h5_strings(&matrix, "barcodes", &barcodes);

    let feat = matrix
        .create_group("features")
        .unwrap_or_else(|e| die(&format!("create /matrix/features: {}", e)));
    // No separate gene-ID column in AnnData var_names, so id mirrors name.
    write_h5_strings(&feat, "id", &features);
    write_h5_strings(&feat, "name", &features);
    write_h5_strings(&feat, "feature_type", &vec!["Gene Expression".to_string(); n_var]);
    write_h5_strings(&feat, "genome", &vec![String::new(); n_var]);
    write_h5_strings(&feat, "_all_tag_keys", &["genome".to_string()]);
}

// ──────────────────────────────────────────────────────────────────────────────
// main
// ──────────────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();

    // Export subcommand: h5ad-inspect <filename> export obs|var|row|column <name>
    // "export" and the filename may appear in any order; sub-cmd and name follow export.
    if let Some(export_pos) = args[1..].iter().position(|a| a == "export") {
        let export_pos = export_pos + 1; // index into args

        // Parse flags from the args after "export". --binary is a bare flag;
        // --layer <name> (or --layer=<name>) selects a matrix source under
        // /layers/<name> instead of /X for the matrix subcommands.
        let mut binary = false;
        let mut layer: Option<String> = None;
        let mut export_args: Vec<&str> = Vec::new();
        let mut rest = args[export_pos + 1..].iter();
        while let Some(a) = rest.next() {
            match a.as_str() {
                "--binary" => binary = true,
                "--layer" => {
                    let v = rest.next().unwrap_or_else(|| {
                        eprintln!("Error: --layer requires a layer name");
                        process::exit(1);
                    });
                    layer = Some(v.clone());
                }
                s if s.starts_with("--layer=") => {
                    layer = Some(s["--layer=".len()..].to_string());
                }
                s => export_args.push(s),
            }
        }

        if export_args.is_empty() {
            eprintln!(
                "Usage: h5ad-inspect <filename> export obs_index|var_index|obssum|varsum|matrix_csr|matrix_csc"
            );
            eprintln!(
                "       h5ad-inspect <filename> export [--binary] obs|var|row|column|obsm <name>"
            );
            eprintln!(
                "       h5ad-inspect <filename> export matrix_cellranger_v3_hdf5 <out.h5>"
            );
            eprintln!(
                "       --layer <name> targets layers/<name> instead of X (row, column, obssum, varsum, matrix_*)"
            );
            process::exit(1);
        }
        let sub_cmd = export_args[0];

        // obs_index / var_index / obssum / varsum / matrix_csr / matrix_csc take no <name> argument.
        let no_name_cmds = [
            "obs_index",
            "var_index",
            "obssum",
            "varsum",
            "matrix_csr",
            "matrix_csc",
        ];
        let needs_name = !no_name_cmds.contains(&sub_cmd);

        if needs_name && export_args.len() < 2 {
            eprintln!("Error: export {} requires a <name> argument", sub_cmd);
            process::exit(1);
        }

        let name = if needs_name { export_args[1] } else { "" };

        // --layer only applies to the matrix subcommands (those that read X).
        // Layers share obs/var axes with X, so only the matrix source changes.
        let layer_cmds = [
            "row",
            "column",
            "obssum",
            "varsum",
            "matrix_csr",
            "matrix_csc",
            "matrix_cellranger_v3_hdf5",
        ];
        if layer.is_some() && !layer_cmds.contains(&sub_cmd) {
            eprintln!(
                "Error: --layer only applies to row, column, obssum, varsum, matrix_csr, matrix_csc, and matrix_cellranger_v3_hdf5"
            );
            process::exit(1);
        }
        let x_path: String = match &layer {
            Some(l) => format!("layers/{}", l),
            None => "X".to_string(),
        };

        // filename is the arg in args[1..] before "export" (or after the export block).
        // Since --binary has been stripped, find the first arg before export_pos that
        // isn't "export" itself, i.e. the positional before "export".
        let filename = args[1..export_pos]
            .iter()
            .map(|a| a.as_str())
            .next()
            .unwrap_or_else(|| {
                eprintln!("Error: could not determine filename");
                process::exit(1);
            });

        let file = hdf5_metno::File::open(filename).unwrap_or_else(|e| {
            eprintln!("Error opening {}: {}", filename, e);
            process::exit(1);
        });

        match sub_cmd {
            "obs_index" => {
                for v in read_group_index(&file, "obs") {
                    println!("{}", v);
                }
            }
            "var_index" => {
                for v in read_group_index(&file, "var") {
                    println!("{}", v);
                }
            }
            "obs" => export_obs_var_column(&file, "obs", name),
            "var" => export_obs_var_column(&file, "var", name),
            "obs_categories" => export_obs_var_categories(&file, "obs", name),
            "var_categories" => export_obs_var_categories(&file, "var", name),
            "obs_encoding" => {
                let (enc, cats) = col_encoding(&file, "obs", name);
                print_encoding_json(&enc, cats.as_deref());
            }
            "var_encoding" => {
                let (enc, cats) = col_encoding(&file, "var", name);
                print_encoding_json(&enc, cats.as_deref());
            }
            "row" => export_x_row(&file, &x_path, name, binary),
            "column" => export_x_column(&file, &x_path, name, binary),
            "obssum" => export_x_obssum(&file, &x_path, binary),
            "varsum" => export_x_varsum(&file, &x_path, binary),
            "obsm" => export_obsm(&file, name, binary),
            "matrix_csr" => export_matrix_csr(&file, &x_path),
            "matrix_csc" => export_matrix_csc(&file, &x_path),
            "matrix_cellranger_v3_hdf5" => export_matrix_cellranger_v3_hdf5(&file, &x_path, name),
            _ => {
                eprintln!(
                    "Error: export subcommand must be obs_index, var_index, obs, var, obs_categories, var_categories, obs_encoding, var_encoding, row, column, obssum, varsum, obsm, matrix_csr, matrix_csc, or matrix_cellranger_v3_hdf5"
                );
                process::exit(1);
            }
        }
        return;
    }

    // Original inspect mode: h5ad-inspect <filename> obs|var|uns|obsm|layers|obs_index|var_index|shape
    if args.len() != 3 {
        eprintln!(
            "Usage: h5ad-inspect <filename> obs|var|uns|obsm|layers|obs_index|var_index|shape"
        );
        eprintln!("       h5ad-inspect <filename> export obs_index|var_index|obssum|varsum|matrix_csr|matrix_csc");
        eprintln!(
            "       h5ad-inspect <filename> export [--binary] obs|var|row|column|obsm <name>"
        );
        process::exit(1);
    }

    let sections = [
        "obs",
        "var",
        "uns",
        "obsm",
        "layers",
        "obs_index",
        "var_index",
        "shape",
    ];
    let section_arg = args[1..].iter().find(|a| sections.contains(&a.as_str()));
    let section = match section_arg {
        Some(s) => s.as_str(),
        None => {
            eprintln!(
                "Error: section must be one of obs, var, uns, obsm, layers, obs_index, var_index"
            );
            process::exit(1);
        }
    };
    let filename = args[1..].iter().find(|a| a.as_str() != section).unwrap();

    let file = match hdf5_metno::File::open(filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error opening {}: {}", filename, e);
            process::exit(1);
        }
    };

    match section {
        "obs" | "var" | "uns" | "obsm" | "layers" => {
            let loc_type = match file.loc_type_by_name(section) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: cannot locate '{}': {}", section, e);
                    process::exit(1);
                }
            };
            if loc_type == hdf5_metno::LocationType::Group {
                let group = match file.group(section) {
                    Ok(g) => g,
                    Err(e) => {
                        eprintln!("Error: group '{}' not found: {}", section, e);
                        process::exit(1);
                    }
                };
                let mut names = match group.member_names() {
                    Ok(n) => n,
                    Err(e) => {
                        eprintln!("Error reading members of '{}': {}", section, e);
                        process::exit(1);
                    }
                };
                if section == "obs" || section == "var" {
                    // Exclude the index dataset (named in the "_index" attr,
                    // defaulting to "_index") and legacy __categories.
                    let index_name = match group.attr("_index") {
                        Ok(attr) => match attr.read_scalar::<hdf5_metno::types::VarLenUnicode>() {
                            Ok(v) => v.as_str().to_string(),
                            Err(_) => "_index".to_string(),
                        },
                        Err(_) => "_index".to_string(),
                    };
                    names.retain(|n| n != &index_name && n != "__categories");
                }
                names.sort_unstable();
                for name in names {
                    println!("{}", name);
                }
            } else {
                let ds = match file.dataset(section) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("Error: dataset '{}' not found: {}", section, e);
                        process::exit(1);
                    }
                };
                let dtype = match ds.dtype() {
                    Ok(dt) => dt,
                    Err(e) => {
                        eprintln!("Error reading dtype of '{}': {}", section, e);
                        process::exit(1);
                    }
                };
                let desc = match dtype.to_descriptor() {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("Error converting dtype of '{}': {}", section, e);
                        process::exit(1);
                    }
                };
                match desc {
                    TypeDescriptor::Compound(ct) => {
                        let mut names: Vec<String> = ct
                            .fields
                            .iter()
                            .map(|f| f.name.clone())
                            .filter(|n| n != "index")
                            .collect();
                        names.sort_unstable();
                        for name in names {
                            println!("{}", name);
                        }
                    }
                    _ => {
                        eprintln!("Error: '{}' is not a compound dataset", section);
                        process::exit(1);
                    }
                }
            }
        }
        "obs_index" | "var_index" => {
            let group_name = if section == "obs_index" { "obs" } else { "var" };
            let loc_type = match file.loc_type_by_name(group_name) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: cannot locate '{}': {}", group_name, e);
                    process::exit(1);
                }
            };
            if loc_type == hdf5_metno::LocationType::Group {
                let group = match file.group(group_name) {
                    Ok(g) => g,
                    Err(e) => {
                        eprintln!("Error opening group '{}': {}", group_name, e);
                        process::exit(1);
                    }
                };
                let index_name = match group.attr("_index") {
                    Ok(attr) => match attr.read_scalar::<hdf5_metno::types::VarLenUnicode>() {
                        Ok(v) => v.as_str().to_string(),
                        Err(_) => "_index".to_string(),
                    },
                    Err(_) => "_index".to_string(),
                };
                let ds = match file.dataset(&format!("{}/{}", group_name, index_name)) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("Error: {}/{}: {}", group_name, index_name, e);
                        process::exit(1);
                    }
                };
                let arr = match ds.read_1d::<hdf5_metno::types::VarLenUnicode>() {
                    Ok(a) => a,
                    Err(e) => {
                        eprintln!("Error reading {group_name}/{index_name}: {e}");
                        process::exit(1);
                    }
                };
                let mut values: Vec<&str> = arr.iter().map(|v| v.as_str()).collect();
                values.sort_unstable();
                for v in values {
                    println!("{}", v);
                }
            } else {
                let ds = match file.dataset(group_name) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("Error: dataset '{}' not found: {}", group_name, e);
                        process::exit(1);
                    }
                };
                let arr = match ds.read_1d::<CompoundIndex>() {
                    Ok(a) => a,
                    Err(e) => {
                        eprintln!("Error reading compound index '{}': {}", group_name, e);
                        process::exit(1);
                    }
                };
                let mut values: Vec<String> = arr
                    .iter()
                    .map(|row| row.index.as_str().to_string())
                    .collect();
                values.sort_unstable();
                for v in values {
                    println!("{}", v);
                }
            }
        }
        "shape" => {
            let n_obs = read_group_index(&file, "obs").len();
            let n_var = read_group_index(&file, "var").len();
            println!("n_obs\t{}", n_obs);
            println!("n_var\t{}", n_var);
        }
        _ => {
            eprintln!(
                "Error: section must be one of obs, var, uns, obsm, layers, obs_index, var_index, shape"
            );
            process::exit(1);
        }
    }
}
