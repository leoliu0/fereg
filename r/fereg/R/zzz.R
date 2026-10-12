# Registers methods on generics of suggested packages when (and if) they are loaded:
# generics::tidy / glance (broom, modelsummary) and the fixest accessors, so that fereg results
# also work with those functions when fixest is attached after fereg.
s3_register <- function(pkg, generic, class, fun) {
  register <- function(...) {
    ns <- asNamespace(pkg)
    if (exists(generic, envir = ns, inherits = FALSE)) registerS3method(generic, class, fun, envir = ns)
  }
  if (isNamespaceLoaded(pkg)) register()
  setHook(packageEvent(pkg, "onLoad"), register)
}

.onLoad <- function(libname, pkgname) {
  s3_register("generics", "tidy", "fereg", tidy.fereg)
  s3_register("generics", "glance", "fereg", glance.fereg)
  s3_register("fixest", "se", "fereg", se.fereg)
  s3_register("fixest", "tstat", "fereg", tstat.fereg)
  s3_register("fixest", "pvalue", "fereg", pvalue.fereg)
  s3_register("fixest", "coeftable", "fereg", coeftable.fereg)
  s3_register("fixest", "fixef", "fereg", fixef.fereg)
  invisible(NULL)
}
