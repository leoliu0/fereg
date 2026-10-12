//! Compiles `c/glue.c` (the only code touching the Stata plugin interface) into a static library.

fn main() {
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let system = if os == "windows" { "STWIN32" } else { "OPUNIX" };
    println!("cargo:rerun-if-changed=c/glue.c");
    println!("cargo:rerun-if-changed=c/stplugin.h");
    cc::Build::new()
        .file("c/glue.c")
        .include("c")
        .define("SYSTEM", system)
        .define("SD_FASTMODE", None)
        .opt_level(2)
        .flag_if_supported("-fvisibility=hidden")
        .warnings(true)
        .compile("rfglue");
}
