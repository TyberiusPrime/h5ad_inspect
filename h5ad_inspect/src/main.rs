use std::env;
use std::ffi::CString;
use std::process;

use hdf5_metno::types::{FloatSize, IntSize, TypeDescriptor};
use hdf5_metno::Datatype;
use hdf5_metno_sys::h5d::H5Dread;
use hdf5_metno_sys::h5p::H5P_DEFAULT;
use hdf5_metno_sys::h5s::H5S_ALL;
use hdf5_metno_sys::h5t::{H5Tclose, H5Tcreate, H5Tinsert, H5T_class_t};
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
            Err(e) => die(&format!("cannot read compound index '{}': {}", group_name, e)),
        }
    }
}

fn find_index_position(index: &[String], value: &str) -> usize {
    index
        .iter()
        .position(|v| v == value)
        .unwrap_or_else(|| die(&format!("'{}' not found in index", value)))
}

// Print each element of a typed 1-D dataset as a newline-separated string.
fn print_dataset_values(ds: &hdf5_metno::Dataset) {
    let desc = ds
        .dtype()
        .and_then(|d| d.to_descriptor())
        .unwrap_or_else(|e| die(&format!("dtype: {}", e)));
    macro_rules! print_typed {
        ($T:ty) => {{
            for v in ds
                .read_1d::<$T>()
                .unwrap_or_else(|e| die(&format!("read: {}", e)))
                .iter()
            {
                println!("{}", v);
            }
        }};
    }
    match desc {
        TypeDescriptor::Integer(IntSize::U1) => print_typed!(i8),
        TypeDescriptor::Integer(IntSize::U2) => print_typed!(i16),
        TypeDescriptor::Integer(IntSize::U4) => print_typed!(i32),
        TypeDescriptor::Integer(IntSize::U8) => print_typed!(i64),
        TypeDescriptor::Unsigned(IntSize::U1) => print_typed!(u8),
        TypeDescriptor::Unsigned(IntSize::U2) => print_typed!(u16),
        TypeDescriptor::Unsigned(IntSize::U4) => print_typed!(u32),
        TypeDescriptor::Unsigned(IntSize::U8) => print_typed!(u64),
        TypeDescriptor::Float(FloatSize::U4) => print_typed!(f32),
        TypeDescriptor::Float(FloatSize::U8) => print_typed!(f64),
        TypeDescriptor::Boolean => print_typed!(bool),
        TypeDescriptor::VarLenUnicode => {
            for v in ds
                .read_1d::<hdf5_metno::types::VarLenUnicode>()
                .unwrap_or_else(|e| die(&format!("read: {}", e)))
                .iter()
            {
                println!("{}", v.as_str());
            }
        }
        TypeDescriptor::VarLenAscii => {
            for v in ds
                .read_1d::<hdf5_metno::types::VarLenAscii>()
                .unwrap_or_else(|e| die(&format!("read: {}", e)))
                .iter()
            {
                println!("{}", v.as_str());
            }
        }
        _ => die("unsupported dtype for export"),
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

// Decode categorical codes to their string labels.
// Negative codes (pandas NA sentinel) become "NA".
fn print_categorical(codes_ds: &hdf5_metno::Dataset, categories: &[String]) {
    let desc = codes_ds
        .dtype()
        .and_then(|d| d.to_descriptor())
        .unwrap_or_else(|e| die(&format!("codes dtype: {}", e)));
    macro_rules! print_codes_signed {
        ($T:ty) => {{
            for &c in codes_ds
                .read_1d::<$T>()
                .unwrap_or_else(|e| die(&format!("codes read: {}", e)))
                .iter()
            {
                if c < 0 {
                    println!("NA");
                } else {
                    let idx = c as usize;
                    if idx < categories.len() {
                        println!("{}", categories[idx]);
                    } else {
                        die(&format!("code {} out of range", c));
                    }
                }
            }
        }};
    }
    macro_rules! print_codes_unsigned {
        ($T:ty) => {{
            for &c in codes_ds
                .read_1d::<$T>()
                .unwrap_or_else(|e| die(&format!("codes read: {}", e)))
                .iter()
            {
                let idx = c as usize;
                if idx < categories.len() {
                    println!("{}", categories[idx]);
                } else {
                    die(&format!("code {} out of range", c));
                }
            }
        }};
    }
    match desc {
        TypeDescriptor::Integer(IntSize::U1) => print_codes_signed!(i8),
        TypeDescriptor::Integer(IntSize::U2) => print_codes_signed!(i16),
        TypeDescriptor::Integer(IntSize::U4) => print_codes_signed!(i32),
        TypeDescriptor::Integer(IntSize::U8) => print_codes_signed!(i64),
        TypeDescriptor::Unsigned(IntSize::U1) => print_codes_unsigned!(u8),
        TypeDescriptor::Unsigned(IntSize::U2) => print_codes_unsigned!(u16),
        TypeDescriptor::Unsigned(IntSize::U4) => print_codes_unsigned!(u32),
        TypeDescriptor::Unsigned(IntSize::U8) => print_codes_unsigned!(u64),
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
    let file_dtype = ds
        .dtype()
        .unwrap_or_else(|e| die(&format!("dtype: {}", e)));
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
        .unwrap_or_else(|| die(&format!("field '{}' not found in compound dataset", col_name)));

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
    let categories: Option<Vec<String>> = file.dataset(&cats_uns_path).ok().map(|cats_ds| {
        read_string_dataset(&cats_ds)
    });

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
                println!("{}", f32::from_ne_bytes(buf[i * 4..(i + 1) * 4].try_into().unwrap()));
            }
        }
        TypeDescriptor::Float(FloatSize::U8) => {
            for i in 0..n_rows {
                println!("{}", f64::from_ne_bytes(buf[i * 8..(i + 1) * 8].try_into().unwrap()));
            }
        }
        _ => die(&format!(
            "unsupported compound field type for '{}': {:?}",
            col_name, field.ty
        )),
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
    let col_loc = file
        .loc_type_by_name(&col_path)
        .unwrap_or_else(|_| die(&format!("column '{}' not found in '{}'", col_name, group_name)));

    if col_loc == hdf5_metno::LocationType::Group {
        // New-style: group contains `categories` + `codes`, or `values` for strings.
        let codes_path = format!("{}/codes", col_path);
        let cats_path_new = format!("{}/categories", col_path);
        let values_path = format!("{}/values", col_path);

        if let (Ok(codes_ds), Ok(cats_ds)) =
            (file.dataset(&codes_path), file.dataset(&cats_path_new))
        {
            // Categorical column
            let categories = read_string_dataset(&cats_ds);
            print_categorical(&codes_ds, &categories);
        } else if let Ok(values_ds) = file.dataset(&values_path) {
            print_dataset_values(&values_ds);
        } else {
            die(&format!(
                "unrecognised group structure for column '{}' in '{}'",
                col_name, group_name
            ));
        }
        return;
    }

    // Column is a direct dataset.
    let col_ds = file
        .dataset(&col_path)
        .unwrap_or_else(|e| die(&format!("cannot open '{}': {}", col_path, e)));

    // Old-style categorical: flat codes dataset + __categories/<col>
    let cats_path_old = format!("{}/__categories/{}", group_name, col_name);
    if let Ok(cats_ds) = file.dataset(&cats_path_old) {
        let categories = read_string_dataset(&cats_ds);
        print_categorical(&col_ds, &categories);
    } else {
        print_dataset_values(&col_ds);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// X matrix helpers
// ──────────────────────────────────────────────────────────────────────────────

fn x_sparse_format(file: &hdf5_metno::File) -> Option<String> {
    let group = file.group("X").ok()?;
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

fn read_indptr(file: &hdf5_metno::File) -> Vec<usize> {
    let ds = file
        .dataset("X/indptr")
        .unwrap_or_else(|e| die(&format!("X/indptr: {}", e)));
    let desc = ds
        .dtype()
        .and_then(|d| d.to_descriptor())
        .unwrap_or_else(|e| die(&format!("X/indptr dtype: {}", e)));
    macro_rules! read_as_usize {
        ($T:ty) => {
            ds.read_1d::<$T>()
                .unwrap_or_else(|e| die(&format!("X/indptr read: {}", e)))
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
        _ => die("unsupported X/indptr dtype"),
    }
}

fn read_indices_slice(file: &hdf5_metno::File, start: usize, end: usize) -> Vec<usize> {
    let ds = file
        .dataset("X/indices")
        .unwrap_or_else(|e| die(&format!("X/indices: {}", e)));
    let desc = ds
        .dtype()
        .and_then(|d| d.to_descriptor())
        .unwrap_or_else(|e| die(&format!("X/indices dtype: {}", e)));
    macro_rules! read_slice_usize {
        ($T:ty) => {
            ds.read_slice_1d::<$T, _>(s![start..end])
                .unwrap_or_else(|e| die(&format!("X/indices read: {}", e)))
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
        _ => die("unsupported X/indices dtype"),
    }
}

fn read_data_slice(file: &hdf5_metno::File, start: usize, end: usize) -> Vec<f64> {
    let ds = file
        .dataset("X/data")
        .unwrap_or_else(|e| die(&format!("X/data: {}", e)));
    let desc = ds
        .dtype()
        .and_then(|d| d.to_descriptor())
        .unwrap_or_else(|e| die(&format!("X/data dtype: {}", e)));
    macro_rules! read_slice_f64 {
        ($T:ty) => {
            ds.read_slice_1d::<$T, _>(s![start..end])
                .unwrap_or_else(|e| die(&format!("X/data read: {}", e)))
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
        _ => die("unsupported X/data dtype"),
    }
}

// Export a CSR row: O(nnz_row) efficient
fn export_x_csr_row(file: &hdf5_metno::File, row_idx: usize, n_cols: usize) {
    let indptr = read_indptr(file);
    let start = indptr[row_idx];
    let end = indptr[row_idx + 1];
    let indices = read_indices_slice(file, start, end);
    let data = read_data_slice(file, start, end);
    let mut vals = vec![0.0f64; n_cols];
    for (k, &col) in indices.iter().enumerate() {
        vals[col] = data[k];
    }
    for v in vals {
        println!("{}", v);
    }
}

// Export a CSC column: O(nnz_col) efficient
fn export_x_csc_col(file: &hdf5_metno::File, col_idx: usize, n_rows: usize) {
    let indptr = read_indptr(file);
    let start = indptr[col_idx];
    let end = indptr[col_idx + 1];
    let indices = read_indices_slice(file, start, end);
    let data = read_data_slice(file, start, end);
    let mut vals = vec![0.0f64; n_rows];
    for (k, &row) in indices.iter().enumerate() {
        vals[row] = data[k];
    }
    for v in vals {
        println!("{}", v);
    }
}

// Export a CSR column (suboptimal: scans all rows)
fn export_x_csr_col(file: &hdf5_metno::File, col_idx: usize, n_rows: usize) {
    let indptr = read_indptr(file);
    // Read full indices + data — no way to avoid it for CSR column access
    let indices = read_indices_slice(file, 0, *indptr.last().unwrap_or(&0));
    let data = read_data_slice(file, 0, *indptr.last().unwrap_or(&0));
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
    for v in vals {
        println!("{}", v);
    }
}

// Export a CSC row (suboptimal: scans all columns)
fn export_x_csc_row(file: &hdf5_metno::File, row_idx: usize, n_cols: usize) {
    let indptr = read_indptr(file);
    let indices = read_indices_slice(file, 0, *indptr.last().unwrap_or(&0));
    let data = read_data_slice(file, 0, *indptr.last().unwrap_or(&0));
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
    for v in vals {
        println!("{}", v);
    }
}

fn export_x_row(file: &hdf5_metno::File, obs_idx_val: &str) {
    let obs_index = read_group_index(file, "obs");
    let var_index = read_group_index(file, "var");
    let obs_pos = find_index_position(&obs_index, obs_idx_val);
    let n_var = var_index.len();

    let x_loc = file
        .loc_type_by_name("X")
        .unwrap_or_else(|e| die(&format!("cannot locate X: {}", e)));

    if x_loc == hdf5_metno::LocationType::Group {
        let fmt = x_sparse_format(file).unwrap_or_else(|| die("cannot determine X sparse format"));
        match fmt.as_str() {
            "csr" => export_x_csr_row(file, obs_pos, n_var),
            "csc" => export_x_csc_row(file, obs_pos, n_var),
            _ => die(&format!("unknown sparse format: {}", fmt)),
        }
    } else {
        // Dense 2-D dataset
        let ds = file
            .dataset("X")
            .unwrap_or_else(|e| die(&format!("X dataset: {}", e)));
        let desc = ds
            .dtype()
            .and_then(|d| d.to_descriptor())
            .unwrap_or_else(|e| die(&format!("X dtype: {}", e)));
        macro_rules! print_dense_row {
            ($T:ty) => {
                for v in ds
                    .read_slice_1d::<$T, _>(s![obs_pos, ..])
                    .unwrap_or_else(|e| die(&format!("X row read: {}", e)))
                    .iter()
                {
                    println!("{}", *v as f64);
                }
            };
        }
        match desc {
            TypeDescriptor::Float(FloatSize::U4) => print_dense_row!(f32),
            TypeDescriptor::Float(FloatSize::U8) => print_dense_row!(f64),
            TypeDescriptor::Integer(IntSize::U4) => print_dense_row!(i32),
            TypeDescriptor::Unsigned(IntSize::U4) => print_dense_row!(u32),
            _ => die("unsupported dense X dtype"),
        }
    }
}

fn export_x_column(file: &hdf5_metno::File, var_idx_val: &str) {
    let obs_index = read_group_index(file, "obs");
    let var_index = read_group_index(file, "var");
    let var_pos = find_index_position(&var_index, var_idx_val);
    let n_obs = obs_index.len();

    let x_loc = file
        .loc_type_by_name("X")
        .unwrap_or_else(|e| die(&format!("cannot locate X: {}", e)));

    if x_loc == hdf5_metno::LocationType::Group {
        let fmt = x_sparse_format(file).unwrap_or_else(|| die("cannot determine X sparse format"));
        match fmt.as_str() {
            "csr" => export_x_csr_col(file, var_pos, n_obs),
            "csc" => export_x_csc_col(file, var_pos, n_obs),
            _ => die(&format!("unknown sparse format: {}", fmt)),
        }
    } else {
        let ds = file
            .dataset("X")
            .unwrap_or_else(|e| die(&format!("X dataset: {}", e)));
        let desc = ds
            .dtype()
            .and_then(|d| d.to_descriptor())
            .unwrap_or_else(|e| die(&format!("X dtype: {}", e)));
        macro_rules! print_dense_col {
            ($T:ty) => {
                for v in ds
                    .read_slice_1d::<$T, _>(s![.., var_pos])
                    .unwrap_or_else(|e| die(&format!("X col read: {}", e)))
                    .iter()
                {
                    println!("{}", *v as f64);
                }
            };
        }
        match desc {
            TypeDescriptor::Float(FloatSize::U4) => print_dense_col!(f32),
            TypeDescriptor::Float(FloatSize::U8) => print_dense_col!(f64),
            TypeDescriptor::Integer(IntSize::U4) => print_dense_col!(i32),
            TypeDescriptor::Unsigned(IntSize::U4) => print_dense_col!(u32),
            _ => die("unsupported dense X dtype"),
        }
    }
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
        if export_pos + 2 >= args.len() {
            eprintln!(
                "Usage: h5ad-inspect <filename> export obs|var|row|column <name>"
            );
            process::exit(1);
        }
        let sub_cmd = args[export_pos + 1].as_str();
        let name = args[export_pos + 2].as_str();
        // filename is whichever arg in args[1..] is not at export_pos, export_pos+1, export_pos+2
        let filename = args[1..]
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let abs = i + 1;
                abs != export_pos && abs != export_pos + 1 && abs != export_pos + 2
            })
            .map(|(_, a)| a.as_str())
            .next()
            .unwrap_or_else(|| {
                eprintln!(
                    "Usage: h5ad-inspect <filename> export obs|var|row|column <name>"
                );
                process::exit(1);
            });

        let file = hdf5_metno::File::open(filename).unwrap_or_else(|e| {
            eprintln!("Error opening {}: {}", filename, e);
            process::exit(1);
        });

        match sub_cmd {
            "obs" => export_obs_var_column(&file, "obs", name),
            "var" => export_obs_var_column(&file, "var", name),
            "row" => export_x_row(&file, name),
            "column" => export_x_column(&file, name),
            _ => {
                eprintln!("Error: export subcommand must be obs, var, row, or column");
                process::exit(1);
            }
        }
        return;
    }

    // Original inspect mode: h5ad-inspect <filename> obs|var|uns|obsm|layers|obs_index|var_index
    if args.len() != 3 {
        eprintln!("Usage: h5ad-inspect <filename> obs|var|uns|obsm|layers|obs_index|var_index");
        eprintln!("       h5ad-inspect <filename> export obs|var|row|column <name>");
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
                    names.retain(|n| n != "_index" && n != "__categories");
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
                    Ok(attr) => match attr
                        .read_scalar::<hdf5_metno::types::VarLenUnicode>()
                    {
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
                let mut values: Vec<String> =
                    arr.iter().map(|row| row.index.as_str().to_string()).collect();
                values.sort_unstable();
                for v in values {
                    println!("{}", v);
                }
            }
        }
        _ => {
            eprintln!(
                "Error: section must be one of obs, var, uns, obsm, layers, obs_index, var_index"
            );
            process::exit(1);
        }
    }
}
