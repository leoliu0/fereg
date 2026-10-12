# Design columns of the linear part. This mirrors model.matrix (treatment contrasts, the
# coding pattern of terms(), first factor fully coded without intercept, first variable varying
# fastest in interactions) plus fixest's i(), without building a matrix: a plain numeric
# column is passed to Rust as is (no copy). Missing values stay NA; the core drops those rows.

# Variable kinds: numeric vector or matrix ("num"), categorical ("fac"), fixest i() ("i").
eval_var <- function(expr, data, env) {
  v <- eval(expr, data, env)
  if (is.null(v)) stop("fereg: variable `", deparse1l(expr), "` not found", call. = FALSE)
  v
}

fac_levels <- function(v) {
  if (is.factor(v)) levels(v) else if (is.logical(v)) c("FALSE", "TRUE") else sort(unique(v[!is.na(v)]))
}

is_categorical <- function(v) is.factor(v) || is.character(v) || is.logical(v)

as_num <- function(v, name) {
  if (is.double(v) && is.null(attributes(v))) return(v)
  if (is.numeric(v) || is.logical(v)) return(as.double(v))
  stop("fereg: variable `", name, "` must be numeric", call. = FALSE)
}

dummy <- function(codes, j) as.double(codes == j)

# Builds the right-hand side columns.
#   terms   list of list(expr, sign) (parse_fml)
#   has_fe  the model has fixed effects (fixest then treats i() as if there were an intercept)
#   meta    NULL at estimation; the returned `meta` when rebuilding on new data (predict)
# Returns list(cols, names, intercept, meta).
linear_design <- function(terms, data, env, has_fe, meta = NULL) {
  if (is.null(meta)) {
    tt <- stats::terms(stats::as.formula(call("~", join_terms(terms)), env = env))
    var_exprs <- as.list(attr(tt, "variables"))[-1]
    meta <- list(
      intercept = attr(tt, "intercept") == 1,
      labels = attr(tt, "term.labels"),
      pattern = attr(tt, "factors"),
      var_names = vapply(var_exprs, deparse1l, ""),
      eval_exprs = var_exprs,
      kinds = character(length(var_exprs)),
      levels = vector("list", length(var_exprs)),
      i_info = vector("list", length(var_exprs)),
      has_fe = has_fe
    )
    fresh <- TRUE
  } else {
    fresh <- FALSE
  }
  nterms <- length(meta$labels)
  if (nterms == 0) return(list(cols = list(), names = character(0), intercept = meta$intercept, meta = meta))
  pattern <- meta$pattern
  nvar <- length(meta$var_names)

  # i() terms: fixest drops the first item of every naked i() (no ref/keep given) when there is
  # an intercept or fixed effects, except the first one when there is neither.
  i_seen <- 0L
  values <- vector("list", nvar)
  for (k in seq_len(nvar)) {
    e <- meta$eval_exprs[[k]]
    if (is_call_to(e, "i")) {
      if (fresh) {
        i_seen <- i_seen + 1L
        ref_special <- meta$intercept || has_fe || i_seen > 1L
        meta$kinds[k] <- "i"
        meta$i_info[[k]] <- i_spec(e, data, env, ref_special)
      }
      values[[k]] <- i_columns(meta$i_info[[k]], data, env)
      next
    }
    v <- eval_var(e, data, env)
    if (fresh) {
      if (is_categorical(v)) {
        meta$kinds[k] <- "fac"
        meta$levels[[k]] <- fac_levels(v)
      } else {
        meta$kinds[k] <- "num"
      }
      # Data-dependent transforms (poly, scale, ...) are re-evaluated with the estimation
      # parameters on new data, as model.frame does.
      meta$eval_exprs[[k]] <- stats::makepredictcall(v, e)
    }
    values[[k]] <- v
  }

  # Coding: 1 = treatment contrasts, 2 = all levels. Without intercept the first factor of the
  # first term containing one is fully coded (model.matrix).
  coding <- pattern
  if (!meta$intercept) {
    done <- FALSE
    for (j in seq_len(nterms)) {
      for (k in seq_len(nvar)) {
        if (meta$kinds[k] == "fac" && coding[k, j] > 0) {
          coding[k, j] <- 2L
          done <- TRUE
          break
        }
      }
      if (done) break
    }
  }

  cols <- list()
  names <- character(0)
  for (j in seq_len(nterms)) {
    vars <- which(pattern[, j] > 0)
    block <- NULL
    for (k in vars) {
      b <- var_block(meta, k, values[[k]], coding[k, j])
      if (meta$kinds[k] == "i" && length(vars) > 1) {
        stop("fereg: i() cannot be interacted with other terms; use i(f, x) instead", call. = FALSE)
      }
      block <- if (is.null(block)) b else block_product(block, b)
    }
    cols <- c(cols, block$cols)
    names <- c(names, block$names)
  }
  list(cols = cols, names = names, intercept = meta$intercept, meta = meta)
}

var_block <- function(meta, k, v, code) {
  name <- meta$var_names[k]
  switch(meta$kinds[k],
    i = v,
    fac = {
      lev <- meta$levels[[k]]
      codes <- match(as.character(v), as.character(lev))
      use <- if (code == 2L) seq_along(lev) else seq_along(lev)[-1]
      list(cols = lapply(use, function(j) dummy(codes, j)), names = paste0(name, lev[use]))
    },
    num = {
      if (is.matrix(v)) {
        cn <- colnames(v)
        if (is.null(cn)) cn <- seq_len(ncol(v))
        list(cols = lapply(seq_len(ncol(v)), function(j) as_num(v[, j], name)), names = paste0(name, cn))
      } else {
        list(cols = list(as_num(v, name)), names = name)
      }
    }
  )
}

# Interaction of two column blocks; the first block varies fastest (model.matrix order).
block_product <- function(a, b) {
  cols <- list()
  names <- character(0)
  for (jb in seq_along(b$cols)) {
    for (ja in seq_along(a$cols)) {
      cols[[length(cols) + 1]] <- a$cols[[ja]] * b$cols[[jb]]
      names <- c(names, paste0(a$names[ja], ":", b$names[jb]))
    }
  }
  list(cols = cols, names = names)
}

# ---- fixest i() --------------------------------------------------------------------------------

i_formals <- function(factor_var, var, ref, keep, bin, ref2, keep2, bin2) NULL

# Evaluates the arguments of an i() call once (at estimation) and fixes its items.
i_spec <- function(e, data, env, ref_special) {
  mc <- match.call(i_formals, e)
  if (!is.null(mc$bin) || !is.null(mc$bin2)) stop("fereg: i(bin = ) is not supported", call. = FALSE)
  if (is.null(mc$factor_var)) stop("fereg: i() needs a factor variable", call. = FALSE)
  f_expr <- mc$factor_var
  f <- eval_var(f_expr, data, env)
  arg <- function(a) if (is.null(mc[[a]])) NULL else eval(mc[[a]], data, env)
  ref <- arg("ref")
  keep <- arg("keep")
  ref2 <- arg("ref2")
  keep2 <- arg("keep2")
  var_expr <- mc$var
  inter <- "none"
  if (!is.null(var_expr)) {
    var_txt <- deparse1l(var_expr)
    if (grepl("^i\\.", var_txt)) {
      var_expr <- str2lang(sub("^i\\.", "", var_txt))
      inter <- "factor"
    } else {
      v <- eval_var(var_expr, data, env)
      if (length(v) == length(f)) {
        inter <- if (is.numeric(v) || is.logical(v)) "numeric" else "factor"
      } else if (length(v) <= 2 && is.null(ref)) {
        ref <- v
        var_expr <- NULL
        ref_special <- FALSE
      } else {
        stop("fereg: in i(), `var` and `factor_var` must have the same length", call. = FALSE)
      }
    }
  }
  if (ref_special && is.null(ref) && is.null(keep) && is.null(ref2) && is.null(keep2) && inter != "numeric") ref <- TRUE

  f_name <- deparse1l(f_expr)
  var_name <- if (!is.null(var_expr)) deparse1l(var_expr)
  key_f <- i_items(f)
  if (inter == "factor") {
    v <- eval_var(var_expr, data, env)
    key_v <- i_items(v)
    present <- unique(data.frame(a = match(f, key_f), b = match(v, key_v)))
    present <- present[!is.na(present$a) & !is.na(present$b), , drop = FALSE]
    present <- present[order(present$a, present$b), , drop = FALSE]
    f_items <- key_f[present$a]
    v_items <- key_v[present$b]
  } else {
    f_items <- key_f
    v_items <- NULL
  }
  drop <- rep(FALSE, length(f_items))
  if (!is.null(ref)) {
    if (isTRUE(ref)) drop[1] <- TRUE else drop <- drop | i_match(f_items, ref)
  }
  if (!is.null(keep)) drop <- drop | !i_match(f_items, keep)
  if (inter == "factor") {
    if (!is.null(ref2)) drop <- drop | i_match(v_items, ref2)
    if (!is.null(keep2)) drop <- drop | !i_match(v_items, keep2)
  }
  if (all(drop)) stop("fereg: all items of ", deparse1l(e), " were removed", call. = FALSE)
  names <- switch(inter,
    none = paste0(f_name, "::", f_items[!drop]),
    numeric = paste0(f_name, "::", f_items[!drop], ":", var_name),
    factor = paste0(f_name, "::", f_items[!drop], ":", var_name, "::", v_items[!drop])
  )
  list(
    f_expr = f_expr, var_expr = var_expr, inter = inter,
    keys = i_keys(f_items, v_items), kept = which(!drop), names = names
  )
}

# Item keys: the value as text, or "f\rv" for factor x factor items.
i_keys <- function(f, v = NULL) {
  if (is.null(v)) as.character(f) else paste(as.character(f), as.character(v), sep = "\r")
}

# Sorted distinct non-missing values (factor: levels present, in level order).
i_items <- function(x) {
  if (is.factor(x)) return(levels(droplevels(x[!is.na(x)])))
  sort(unique(x[!is.na(x)]))
}

# fixest item matching: exact value, or a regular expression with a leading "@".
i_match <- function(items, what) {
  items_c <- as.character(items)
  out <- rep(FALSE, length(items))
  for (w in as.character(what)) {
    if (startsWith(w, "@")) out <- out | grepl(substring(w, 2), items_c) else out <- out | items_c == w
  }
  out
}

# The columns of an i() term on (possibly new) data. Values outside the estimation items are
# an error, as in fixest.
i_columns <- function(spec, data, env) {
  f <- eval_var(spec$f_expr, data, env)
  na <- is.na(f)
  if (spec$inter == "factor") {
    v <- eval_var(spec$var_expr, data, env)
    na <- na | is.na(v)
    codes <- match(i_keys(f, v), spec$keys)
  } else {
    codes <- match(i_keys(f), spec$keys)
  }
  if (any(is.na(codes) & !na)) {
    stop("fereg: in i(), predictions cannot be done for values that were not present at estimation time", call. = FALSE)
  }
  numeric_var <- if (spec$inter == "numeric") as_num(eval_var(spec$var_expr, data, env), deparse1l(spec$var_expr))
  if (!is.null(numeric_var)) na <- na | is.na(numeric_var)
  cols <- lapply(spec$kept, function(j) {
    col <- dummy(codes, j)
    if (!is.null(numeric_var)) col <- col * numeric_var
    col[na] <- NA
    col
  })
  list(cols = cols, names = spec$names)
}

# ---- fixed effects, slopes and clusters ----------------------------------------------------------

# Categorical column for the core: double (F64), integer (I32), or 1-based codes with a `levels`
# attribute (factors as is; character and logical recoded).
factor_column <- function(v, name) {
  if (is.factor(v)) return(v)
  if (is.double(v) || (is.integer(v) && is.null(attr(v, "levels")))) return(v)
  if (is.character(v)) {
    u <- unique(v)
    u <- u[!is.na(u)]
    codes <- match(v, u)
    attr(codes, "levels") <- u
    return(codes)
  }
  if (is.logical(v)) return(as.integer(v))
  if (is.numeric(v)) return(as.double(v))
  stop("fereg: fixed-effect / cluster variable `", name, "` must be a vector", call. = FALSE)
}

# Registry of the distinct factor columns of one model (FE and cluster variables). Evaluated
# columns are shared by the models of one call through `cache` (an environment).
new_factors <- function(data, env, cache) {
  reg <- new.env(parent = emptyenv())
  reg$data <- data
  reg$env <- env
  reg$cache <- cache
  reg$names <- character(0)
  reg$values <- list()
  reg$cols <- list()
  reg
}

# Index (1-based) of the factor `expr` in the registry, adding it when new.
factor_index <- function(reg, expr) {
  name <- deparse1l(expr)
  k <- match(name, reg$names)
  if (!is.na(k)) return(k)
  e <- cached(reg$cache, paste0("f:", name), function() {
    v <- eval_var(expr, reg$data, reg$env)
    list(value = v, col = factor_column(v, name))
  })
  reg$names <- c(reg$names, name)
  reg$values[[length(reg$values) + 1]] <- e$value
  reg$cols[[length(reg$cols) + 1]] <- e$col
  length(reg$names)
}

# FE structure of a model: terms for the core plus what printing / fixef() / predict() need.
fe_design <- function(fe_exprs, reg, slope_reg) {
  terms <- lapply(fe_exprs, parse_fe_term)
  core <- lapply(terms, function(t) {
    list(
      factors = vapply(t$vars, function(v) factor_index(reg, v), 0),
      intercept = t$intercept,
      slopes = vapply(t$slopes, function(s) slope_index(slope_reg, s), 0)
    )
  })
  list(terms = terms, core = core)
}

new_slopes <- function(data, env, cache) {
  reg <- new.env(parent = emptyenv())
  reg$data <- data
  reg$env <- env
  reg$cache <- cache
  reg$names <- character(0)
  reg$cols <- list()
  reg
}

slope_index <- function(reg, expr) {
  name <- deparse1l(expr)
  k <- match(name, reg$names)
  if (!is.na(k)) return(k)
  col <- cached(reg$cache, paste0("x:", name), function() as_num(eval_var(expr, reg$data, reg$env), name))
  reg$names <- c(reg$names, name)
  reg$cols[[length(reg$cols) + 1]] <- col
  length(reg$names)
}

# Level labels from the values of the term's variables at one row per level; fixest pastes the
# values of interacted variables with "_".
fe_labels <- function(values) {
  parts <- lapply(values, as.character)
  if (length(parts) == 1) parts[[1]] else do.call(paste, c(parts, sep = "_"))
}

# Level order: fixest sorts levels by value (factors by level order), interactions by the first
# variable, then the second, ...
fe_order <- function(values) {
  keys <- lapply(values, function(x) if (is.factor(x)) as.integer(x) else x)
  do.call(order, unname(keys))
}
