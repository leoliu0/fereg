test_that("formula parts: linear, fixed effects, IV", {
  m <- fereg:::parse_fml(y ~ x1 + x2 | id + year^firm | e1 ~ z1 + z2)
  expect_length(m, 1)
  m <- m[[1]]
  expect_identical(m$lhs, quote(y))
  expect_identical(lapply(m$rhs_terms, `[[`, "expr"), list(quote(x1), quote(x2)))
  expect_identical(m$fe_terms, list(quote(id), quote(year^firm)))
  expect_identical(m$endo, list(quote(e1)))
  expect_identical(m$inst, list(quote(z1), quote(z2)))

  iv_nofe <- fereg:::parse_fml(y ~ x1 | e1 + e2 ~ z1 + z2)[[1]]
  expect_length(iv_nofe$fe_terms, 0)
  expect_identical(iv_nofe$endo, list(quote(e1), quote(e2)))

  nocons <- fereg:::parse_fml(y ~ x1 - 1)[[1]]
  expect_identical(vapply(nocons$rhs_terms, `[[`, 0, "sign"), c(1, -1))
})

test_that("fixed-effect terms: interactions and varying slopes", {
  t1 <- fereg:::parse_fe_term(quote(a^b^c))
  expect_identical(t1$vars, list(quote(a), quote(b), quote(c)))
  expect_true(t1$intercept)
  expect_identical(t1$label, "a^b^c")
  t2 <- fereg:::parse_fe_term(quote(g[x, z]))
  expect_true(t2$intercept)
  expect_identical(t2$slopes, list(quote(x), quote(z)))
  t3 <- fereg:::parse_fe_term(quote(g[[x]]))
  expect_false(t3$intercept)
  expect_identical(t3$label, "g")
})

test_that("multiple estimation follows fixest's order", {
  ms <- fereg:::parse_fml(c(y, w) ~ sw(x1, x2) | csw0(id, year))
  ids <- vapply(ms, function(m) paste0(names(m$id), "=", m$id, collapse = ";"), "")
  expect_identical(ids, c(
    "fixef=1;lhs=y;rhs=x1", "fixef=1;lhs=y;rhs=x2", "fixef=1;lhs=w;rhs=x1", "fixef=1;lhs=w;rhs=x2",
    "fixef=id;lhs=y;rhs=x1", "fixef=id;lhs=y;rhs=x2", "fixef=id;lhs=w;rhs=x1", "fixef=id;lhs=w;rhs=x2",
    "fixef=id + year;lhs=y;rhs=x1", "fixef=id + year;lhs=y;rhs=x2", "fixef=id + year;lhs=w;rhs=x1",
    "fixef=id + year;lhs=w;rhs=x2"
  ))
  cs <- fereg:::parse_fml(y ~ x3 + csw(x1, x2))
  expect_identical(lapply(cs[[2]]$rhs_terms, `[[`, "expr"), list(quote(x3), quote(x1), quote(x2)))
  expect_error(fereg:::parse_fml(y ~ sw(x1, x2) + csw(x3)), "only one stepwise")
})

test_that("vcov requests", {
  expect_identical(fereg:::parse_vcov("hetero")$type, "hetero")
  expect_identical(fereg:::parse_vcov(NULL)$type, "iid")
  v <- fereg:::parse_vcov(~ id + year^firm)
  expect_identical(v$dims, list(list(quote(id)), list(quote(year), quote(firm))))
  expect_identical(fereg:::vcov_label(v), "Clustered (id & year^firm)")
  tw <- fereg:::resolve_vcov(fereg:::parse_vcov("twoway"), lapply(list(quote(id), quote(g[[x]]), quote(year)), fereg:::parse_fe_term))
  expect_identical(tw$dims, list(list(quote(id)), list(quote(year))))
  expect_identical(fereg:::parse_vcov(cluster = ~id)$type, "cluster")
  expect_error(fereg:::resolve_vcov(fereg:::parse_vcov("twoway"), list()), "needs 2 fixed effects")
})

test_that("ssc() validates and maps former names", {
  s <- ssc(adj = FALSE, cluster.df = "conventional")
  expect_false(s$K.adj)
  expect_identical(s$G.df, "conventional")
  expect_error(ssc(K.fixef = "bad"))
  expect_identical(fereg:::effective_ssc(NULL, "hetero")$K.fixef, "full")
  expect_identical(fereg:::effective_ssc(NULL, "cluster")$K.fixef, "nonnested")
})
