/*
 * rfmock.c -- a fake fereg.plugin for ado-level dry runs (tests/stata/test_ado_mock.do).
 *
 * It implements the plugin protocol (cmd=version|prepare|ols|glm) without estimating anything:
 *   prepare  applies the protocol's sample rule (touse = last variable; touse is zeroed for
 *            observations with a missing value, or "" for strings, in any other variable),
 *            counts the sample, returns token 777 and copies the staged scalars
 *            __mock_N_full / __mock_num_singletons when present;
 *   ols      copies the staged matrices __mock_b, __mock_V, __mock_doft into the matrices named by
 *            b= V= doft= (dimensions must match), copies staged scalars __mock_<name> to __rf_<name>,
 *            returns the status list from global $rfmock_status, and writes 1000*k + obs into the
 *            k-th output variable for every sample observation (except the groupvar output when
 *            $rfmock_nogroup is 1);
 *   glm      the same with the glm scalars and the bias-correction matrices b0= b2= V2= (staged
 *            __mock_b0 __mock_b2 __mock_V2); the varlist is y x... [offset] bcvars... touse out...
 *            (nx=, offset=, nbc=), and every bias-correction variable must be nonmissing at every
 *            sample observation (the real plugin reads them there), else r(416).
 * The argv of the last calls are recorded in globals $rfmock_prepare, $rfmock_ols and $rfmock_glm.
 * Build: tests/stata/mock/build.sh
 */
#include "stplugin.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

ST_plugin *_stata_;

STDLL pginit(ST_plugin *p) {
  _stata_ = p;
  return SD_PLUGINVER;
}
static const char *key(int argc, char *argv[], const char *k) {
  size_t n = strlen(k);
  for (int i = 1; i < argc; i++) if (!strncmp(argv[i], k, n) && argv[i][n] == '=') return argv[i] + n + 1;
  return NULL;
}
static void record(const char *glob, int argc, char *argv[]) {
  char buf[8000]; buf[0] = 0;
  for (int i = 0; i < argc; i++) { strcat(buf, argv[i]); strcat(buf, " "); }
  SF_macro_save((char *)glob, buf);
}
static void copy_scalar(const char *n) {
  char src[80], dst[80]; ST_double v;
  snprintf(src, 80, "__mock_%s", n); snprintf(dst, 80, "__rf_%s", n);
  if (SF_scal_use(src, &v) == 0) SF_scal_save(dst, v);
}
static int copy_mat(const char *src, const char *dst) {
  if (!dst) return 0;
  int r = SF_row((char *)src), c = SF_col((char *)src);
  if (r != SF_row((char *)dst) || c != SF_col((char *)dst)) {
    char m[200]; snprintf(m, 200, "fereg: matrix %s has wrong dimensions (%dx%d vs mock %dx%d)\n", dst, SF_row((char*)dst), SF_col((char*)dst), r, c);
    SF_error(m); return 503;
  }
  for (int i = 1; i <= r; i++) for (int j = 1; j <= c; j++) { ST_double v; SF_mat_el((char *)src, i, j, &v); SF_mat_store((char *)dst, i, j, v); }
  return 0;
}
STDLL stata_call(int argc, char *argv[]) {
  if (argc < 1) return 198;
  if (!strcmp(argv[0], "version")) { SF_macro_save("__rf_version", "0.5.0"); return 0; }
  if (!strcmp(argv[0], "prepare")) {
    record("rfmock_prepare", argc, argv);
    int tv = SF_nvars(); double n = 0; ST_double t, z; char s[2048];
    for (int o = SF_in1(); o <= SF_in2(); o++) {
      if (!SF_ifobs(o)) continue;
      SF_vdata(tv, o, &t); if (t == 0) continue;
      int miss = 0;
      for (int v = 1; v < tv && !miss; v++) {
        if (SF_var_is_string(v)) {
          if (SF_var_is_strl(v)) miss = SF_sdatalen(v, o) == 0;
          else { SF_sdata(v, o, s); miss = s[0] == 0; }
        } else { SF_vdata(v, o, &z); miss = SF_is_missing(z); }
      }
      if (miss) SF_vstore(tv, o, 0); else n++;
    }
    SF_scal_save("__rf_N_full", n); SF_scal_save("__rf_N", n);
    SF_scal_save("__rf_num_singletons", 0); SF_scal_save("__rf_num_fe_separated", 0);
    copy_scalar("N_full"); copy_scalar("num_singletons");
    char nv[32]; snprintf(nv, 32, "%d", SF_nvars()); SF_macro_save("rfmock_prepare_nvars", nv);
    SF_macro_save("__rf_token", "777");
    return 0;
  }
  if (!strcmp(argv[0], "ols")) {
    record("rfmock_ols", argc, argv);
    const char *tok = key(argc, argv, "token");
    if (!tok || strcmp(tok, "777")) { SF_error("fereg: stale plugin session; rerun the command\n"); return 198; }
    int rc;
    if ((rc = copy_mat("__mock_b", key(argc, argv, "b")))) return rc;
    if ((rc = copy_mat("__mock_V", key(argc, argv, "V")))) return rc;
    if ((rc = copy_mat("__mock_doft", key(argc, argv, "doft")))) return rc;
    const char *names[] = {"N","rank","df_r","df_m","tss","tss_within","rss","mss","rmse","F","ll","ll_0","r2","r2_within","r2_a","r2_a_within","sumweights","ic","converged","report_constant","N_clust","N_clustervars","N_clust1","N_clust2","N_clust3","df_a","df_a_initial","df_a_redundant","df_a_nested","num_singletons","N_full","y_absorbed","dkraay_bw","bw","cutoff","hac_T",NULL};
    for (int i = 0; names[i]; i++) copy_scalar(names[i]);
    char st[4000] = ""; SF_macro_use("rfmock_status", st, 4000); SF_macro_save("__rf_status", st);
    char nv[32]; snprintf(nv, 32, "%d", SF_nvars()); SF_macro_save("rfmock_ols_nvars", nv);
    int nx = atoi(key(argc, argv, "nx")), nout = atoi(key(argc, argv, "nout"));
    int tv = 1 + nx + 1;
    /* $rfmock_nogroup == 1 simulates "no pair of intercept FEs": the groupvar output stays missing */
    char ng[8] = ""; SF_macro_use("rfmock_nogroup", ng, 8);
    int gpos = atoi(key(argc, argv, "groupvar"));
    for (int k = 1; k <= nout; k++) for (int o = SF_in1(); o <= SF_in2(); o++) {
      ST_double t; SF_vdata(tv, o, &t); if (t == 0) continue;
      if (k == gpos && ng[0] == '1') continue;
      SF_vstore(tv + k, o, 1000.0 * k + o);
    }
    if (SF_nvars() != tv + nout) { SF_error("mock: varlist length mismatch\n"); return 198; }
    return 0;
  }
  if (!strcmp(argv[0], "glm")) {
    record("rfmock_glm", argc, argv);
    const char *tok = key(argc, argv, "token");
    if (!tok || strcmp(tok, "777")) { SF_error("fereg: stale plugin session; rerun the command\n"); return 198; }
    int rc;
    if ((rc = copy_mat("__mock_b", key(argc, argv, "b")))) return rc;
    if ((rc = copy_mat("__mock_V", key(argc, argv, "V")))) return rc;
    if ((rc = copy_mat("__mock_doft", key(argc, argv, "doft")))) return rc;
    if ((rc = copy_mat("__mock_b0", key(argc, argv, "b0")))) return rc;
    if ((rc = copy_mat("__mock_b2", key(argc, argv, "b2")))) return rc;
    if ((rc = copy_mat("__mock_V2", key(argc, argv, "V2")))) return rc;
    const char *names[] = {"N","rank","ic","ic2","converged","ll","ll_0","df_m","chi2","r2_p","deviance","df_a","df_a_initial","df_a_redundant","df_a_nested","num_singletons","num_fe_separated","num_separated","N_full","separation_complete","N_clust","N_clustervars","N_clust1","N_clust2","dkraay_bw","bw","cutoff","hac_T","bc_nfits",NULL};
    for (int i = 0; names[i]; i++) copy_scalar(names[i]);
    char st[4000] = ""; SF_macro_use("rfmock_status", st, 4000); SF_macro_save("__rf_status", st);
    char nv[32]; snprintf(nv, 32, "%d", SF_nvars()); SF_macro_save("rfmock_glm_nvars", nv);
    const char *s;
    int nx = atoi(key(argc, argv, "nx")), nout = atoi(key(argc, argv, "nout"));
    int off = (s = key(argc, argv, "offset")) ? atoi(s) : 0;
    int nbc = (s = key(argc, argv, "nbc")) ? atoi(s) : 0;
    int tv = 1 + nx + off + nbc + 1;
    if (SF_nvars() != tv + nout) { SF_error("mock: varlist length mismatch\n"); return 198; }
    for (int o = SF_in1(); o <= SF_in2(); o++) {
      ST_double t, z; SF_vdata(tv, o, &t); if (t == 0) continue;
      for (int v = tv - nbc; v < tv; v++) {
        SF_vdata(v, o, &z);
        if (SF_is_missing(z)) { SF_error("mock: missing bias-correction variable in the sample\n"); return 416; }
      }
    }
    for (int k = 1; k <= nout; k++) for (int o = SF_in1(); o <= SF_in2(); o++) {
      ST_double t; SF_vdata(tv, o, &t); if (t == 0) continue;
      SF_vstore(tv + k, o, 1000.0 * k + o);
    }
    return 0;
  }
  SF_error("mock: unknown cmd\n"); return 198;
}
