# r-universe runs this script before `R CMD build`: it copies the fereg-core crate from the
# repository into src/rust/fereg-core (the R build only sees this package directory).
sync <- file.path("..", "..", "scripts", "sync-core.sh")
if (!file.exists(sync)) stop("bootstrap.R must run from r/fereg inside the fereg repository")
status <- system2("sh", sync)
if (status != 0) stop("scripts/sync-core.sh failed")
