# Parsing of fixest formulas: `lhs ~ rhs | fixed effects | endo ~ inst`, multiple dependent
# variables `c(y1, y2)` and stepwise functions sw(), sw0(), csw(), csw0() on the right-hand side
# and in the fixed-effects part. The result is one model description per estimation, in fixest's
# order (fixed-effects outermost, then dependent variable, then right-hand side).

stepwise_funs <- c("sw", "sw0", "csw", "csw0")

is_call_to <- function(e, fun) is.call(e) && is.name(e[[1]]) && as.character(e[[1]]) %in% fun

# Splits `a + b - c` into its top-level terms: list of list(expr, sign).
split_plus <- function(e) {
  if (is_call_to(e, "(") && length(e) == 2) return(split_plus(e[[2]]))
  if (is_call_to(e, c("+", "-")) && length(e) == 3) {
    right <- split_plus_sign(e[[3]], if (as.character(e[[1]]) == "-") -1 else 1)
    return(c(split_plus(e[[2]]), right))
  }
  if (is_call_to(e, "-") && length(e) == 2) return(split_plus_sign(e[[2]], -1))
  list(list(expr = e, sign = 1))
}

split_plus_sign <- function(e, sign) {
  out <- split_plus(e)
  if (sign < 0) out <- lapply(out, function(t) {
    t$sign <- -t$sign
    t
  })
  out
}

# Splits `a | b | c` into its parts.
split_bar <- function(e) {
  if (is_call_to(e, "|") && length(e) == 3) return(c(split_bar(e[[2]]), list(e[[3]])))
  list(e)
}

# Recombines signed terms into one expression (`1` when empty).
join_terms <- function(terms) {
  if (length(terms) == 0) return(1)
  out <- NULL
  for (t in terms) {
    if (is.null(out)) {
      out <- if (t$sign > 0) t$expr else call("-", t$expr)
    } else {
      out <- call(if (t$sign > 0) "+" else "-", out, t$expr)
    }
  }
  out
}

deparse1l <- function(e) paste(deparse(e, width.cutoff = 500L), collapse = " ")

# Expands at most one stepwise function among top-level terms into alternatives.
# Returns list of list(terms, label); `label` is the text of the whole part (fixest naming).
expand_stepwise <- function(terms, part) {
  is_sw <- vapply(terms, function(t) is_call_to(t$expr, stepwise_funs), logical(1))
  if (sum(is_sw) > 1) stop("fereg: only one stepwise function (sw, sw0, csw, csw0) is allowed in the ", part, call. = FALSE)
  if (!any(is_sw)) return(list(list(terms = terms, label = NULL)))
  pos <- which(is_sw)
  sw <- terms[[pos]]$expr
  fun <- as.character(sw[[1]])
  args <- as.list(sw)[-1]
  if (length(args) == 0) stop("fereg: ", fun, "() needs at least one argument", call. = FALSE)
  pieces <- lapply(args, split_plus)
  alts <- switch(fun,
    sw = lapply(seq_along(pieces), function(k) pieces[k]),
    sw0 = c(list(list()), lapply(seq_along(pieces), function(k) pieces[k])),
    csw = lapply(seq_along(pieces), function(k) pieces[seq_len(k)]),
    csw0 = c(list(list()), lapply(seq_along(pieces), function(k) pieces[seq_len(k)]))
  )
  before <- terms[seq_len(pos - 1)]
  after <- terms[-seq_len(pos)]
  lapply(alts, function(a) {
    inner <- unlist(a, recursive = FALSE)
    all <- c(before, inner, after)
    list(terms = all, label = if (length(all)) deparse1l(join_terms(all)) else "1")
  })
}

# Parses one fixest formula. Returns a list of models, each
#   list(lhs, rhs_terms, fe_terms, endo, inst, fml, fml_fixef, fml_iv, id)
# where *_terms are lists of list(expr, sign) and `id` names the multiple-estimation slot.
parse_fml <- function(fml) {
  if (!inherits(fml, "formula") || length(fml) != 3) {
    stop("fereg: `fml` must be a two-sided formula, e.g. y ~ x | fe", call. = FALSE)
  }
  env <- environment(fml)
  if (is.null(env)) env <- parent.frame()
  inner <- fml[[2]]
  endo <- inst <- NULL
  if (is_call_to(inner, "~") && length(inner) == 3) {
    # `y ~ x | fe | endo ~ inst` parses as `(y ~ x | fe | endo) ~ inst`.
    inst <- fml[[3]]
    lhs <- inner[[2]]
    parts <- split_bar(inner[[3]])
    if (length(parts) < 2) stop("fereg: the IV part must follow `|`: y ~ x | fe | endo ~ inst", call. = FALSE)
    endo <- parts[[length(parts)]]
    parts <- parts[-length(parts)]
  } else {
    lhs <- fml[[2]]
    parts <- split_bar(fml[[3]])
  }
  if (length(parts) > 2) stop("fereg: a formula has at most three parts: y ~ x | fe | endo ~ inst", call. = FALSE)

  lhs_list <- if (is_call_to(lhs, "c")) as.list(lhs)[-1] else list(lhs)
  if (length(lhs_list) == 0) stop("fereg: empty dependent variable", call. = FALSE)
  rhs_alts <- expand_stepwise(split_plus(parts[[1]]), "right-hand side")
  fe_alts <- if (length(parts) == 2) expand_stepwise(split_plus(parts[[2]]), "fixed-effects part") else list(list(terms = list(), label = NULL))
  endo_terms <- if (!is.null(endo)) split_plus(endo)
  inst_terms <- if (!is.null(inst)) split_plus(inst)
  if (!is.null(endo)) {
    bad <- vapply(c(endo_terms, inst_terms), function(t) t$sign < 0 || is_call_to(t$expr, stepwise_funs) || (is.numeric(t$expr)), logical(1))
    if (any(bad)) stop("fereg: the IV part must list variables: endo1 + endo2 ~ inst1 + inst2", call. = FALSE)
  }

  multi_lhs <- length(lhs_list) > 1
  out <- list()
  for (fe in fe_alts) {
    fe_terms <- Filter(function(t) !(is.numeric(t$expr) && t$expr %in% c(0, 1)), fe$terms)
    for (l in lhs_list) {
      for (r in rhs_alts) {
        rhs_expr <- join_terms(r$terms)
        main <- stats::as.formula(call("~", l, rhs_expr), env = env)
        id <- c(
          fixef = if (!is.null(fe$label)) fe$label,
          lhs = if (multi_lhs) deparse1l(l),
          rhs = if (!is.null(r$label)) r$label
        )
        out[[length(out) + 1]] <- list(
          lhs = l,
          rhs_terms = r$terms,
          fe_terms = lapply(fe_terms, `[[`, "expr"),
          endo = lapply(endo_terms, `[[`, "expr"),
          inst = lapply(inst_terms, `[[`, "expr"),
          fml = main,
          fml_fixef = if (length(fe_terms)) stats::as.formula(call("~", join_terms(fe_terms)), env = env),
          fml_iv = if (!is.null(endo)) stats::as.formula(call("~", endo, inst), env = env),
          id = id
        )
      }
    }
  }
  out
}

# ---- fixed-effects terms ---------------------------------------------------------------------

# One FE term: list(vars = list of expressions (interaction when > 1), intercept, slopes = list of
# expressions, label). fixest: `a^b` interaction, `g[x]` FE + slope, `g[[x]]` slope only,
# `g[x, z]` FE + two slopes.
parse_fe_term <- function(e) {
  intercept <- TRUE
  slopes <- list()
  if (is_call_to(e, c("[", "[["))) {
    intercept <- as.character(e[[1]]) == "["
    slopes <- as.list(e)[-(1:2)]
    if (length(slopes) == 0) stop("fereg: varying slopes need a variable: fe[x] or fe[[x]]", call. = FALSE)
    e <- e[[2]]
  }
  vars <- split_hat(e)
  label <- paste(vapply(vars, deparse1l, ""), collapse = "^")
  list(vars = vars, intercept = intercept, slopes = slopes, label = label)
}

split_hat <- function(e) {
  if (is_call_to(e, "^") && length(e) == 3) return(c(split_hat(e[[2]]), split_hat(e[[3]])))
  list(e)
}
