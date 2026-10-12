/*
    mockhost.c — a fake Stata host for ABI tests of the fereg plugin on every OS (plan §9.2).

    Implements the SPI 3.0 ST_plugin table over an in-memory 1000-observation dataset with two
    fixed effects (numeric fe1, string fe2), plus scalar/matrix/macro registries, loads the
    plugin with dlopen/LoadLibrary and runs version, selftest, prepare (absorber=dense and
    fast) and ols, comparing against the JSON written by
        cargo run --release -p fereg-core --example mock_expected
    Steps whose core implementation is still a stub are reported as PENDING (not failures).

    Build/run (from the repository root):
        cc -O2 -DSYSTEM=OPUNIX -Icrates/fereg-stata/c tests/mockhost/mockhost.c -ldl -o target/mockhost
        target/mockhost stata/fereg.linux64.plugin [target/mock_expected.json]
    Exit status: 0 when nothing FAILed.
*/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stplugin.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define NOBS 1000
#define MISSVAL 8.98846567431158e307

/* ---------------------------------------------------------------- dataset */

enum { V_FE1, V_FE2, V_Y, V_X1, V_X2, V_TOUSE, V_RESID, V_SPARE, NVAR };

static double num[NVAR][NOBS];
static char fe2[NOBS][8];

static int64_t h(uint64_t i, uint64_t k) {
    return (int64_t)(((i * 2654435761ULL + k * 40503ULL) % 4294967296ULL) % 2001ULL) - 1000;
}

/* Mirrors mock_data() in crates/fereg-core/examples/mock_expected.rs. */
static void make_data(void) {
    for (uint64_t i = 0; i < NOBS; i++) {
        num[V_FE1][i] = i < 990 ? (double)(1 + (i * 7) % 50) : (double)(1000 + i);
        snprintf(fe2[i], sizeof fe2[i], "g%02d", (int)((i * 13 + i / 7) % 20));
        num[V_X1][i] = (double)h(i, 1) / 100.0;
        num[V_X2][i] = (double)h(i, 2) / 100.0;
        num[V_Y][i] = (double)(h(i, 3) + 2 * h(i, 1) - h(i, 2)) / 100.0;
        num[V_TOUSE][i] = 1.0;
        num[V_RESID][i] = MISSVAL;
        num[V_SPARE][i] = MISSVAL;
    }
}

/* Current call: varlist (dataset variable per 1-based position) and in-range. */
static int varlist[16];
static int nvarlist = 0;
static int in1 = 1, in2 = NOBS;
static ST_int stopflag = 0;

static int pos_ok(int i, int j) { return i >= 1 && i <= nvarlist && j >= in1 && j <= NOBS; }

/* ---------------------------------------------------------------- registries */

#define MAXREG 64
static struct { char name[64]; char value[512]; } macros[MAXREG];
static int nmacros = 0;
static struct { char name[64]; double value; } scalars[MAXREG];
static int nscalars = 0;
static struct { char name[64]; int rows, cols; double v[64]; } mats[8];
static int nmats = 0;
static char last_err[1024];

static const char *macro_get(const char *name) {
    for (int k = 0; k < nmacros; k++) if (!strcmp(macros[k].name, name)) return macros[k].value;
    return NULL;
}

static int scalar_get(const char *name, double *v) {
    for (int k = 0; k < nscalars; k++) if (!strcmp(scalars[k].name, name)) { *v = scalars[k].value; return 1; }
    return 0;
}

static int mat_find(const char *name) {
    for (int k = 0; k < nmats; k++) if (!strcmp(mats[k].name, name)) return k;
    return -1;
}

static void mat_define(const char *name, int rows, int cols) {
    int k = mat_find(name);
    if (k < 0) k = nmats++;
    snprintf(mats[k].name, sizeof mats[k].name, "%s", name);
    mats[k].rows = rows;
    mats[k].cols = cols;
    for (int e = 0; e < rows * cols; e++) mats[k].v[e] = MISSVAL;
}

/* ---------------------------------------------------------------- ST_plugin functions */

static ST_int m_spoutsml(char *s) { fputs(s, stdout); return 0; }
static ST_int m_spouterr(char *s) {
    fputs(s, stdout);
    snprintf(last_err, sizeof last_err, "%s", s);
    return 0;
}
static void m_spoutflush(void) { fflush(stdout); }
static ST_int m_poll(void) { return 0; }
static ST_int m_macresave(char *name, char *value) {
    int k = 0;
    while (k < nmacros && strcmp(macros[k].name, name)) k++;
    if (k == nmacros) { if (nmacros == MAXREG) return 920; nmacros++; }
    snprintf(macros[k].name, sizeof macros[k].name, "%s", name);
    snprintf(macros[k].value, sizeof macros[k].value, "%s", value);
    return 0;
}
static ST_int m_scalsave(char *name, ST_double v) {
    int k = 0;
    while (k < nscalars && strcmp(scalars[k].name, name)) k++;
    if (k == nscalars) { if (nscalars == MAXREG) return 920; nscalars++; }
    snprintf(scalars[k].name, sizeof scalars[k].name, "%s", name);
    scalars[k].value = v;
    return 0;
}
static ST_int m_matstore(char *name, ST_int r, ST_int c, ST_double v) {
    int k = mat_find(name);
    if (k < 0) return 111;
    if (r < 1 || c < 1 || r > mats[k].rows || c > mats[k].cols) return 503;
    mats[k].v[(r - 1) * mats[k].cols + (c - 1)] = v;
    return 0;
}
static ST_int m_matel(char *name, ST_int r, ST_int c, ST_double *v) {
    int k = mat_find(name);
    if (k < 0) return 111;
    if (r < 1 || c < 1 || r > mats[k].rows || c > mats[k].cols) return 503;
    *v = mats[k].v[(r - 1) * mats[k].cols + (c - 1)];
    return 0;
}
static ST_int m_rowsof(char *name) { int k = mat_find(name); return k < 0 ? 0 : mats[k].rows; }
static ST_int m_colsof(char *name) { int k = mat_find(name); return k < 0 ? 0 : mats[k].cols; }
static ST_int m_vdata(ST_int i, ST_int j, ST_double *z) {
    if (!pos_ok(i, j) || varlist[i - 1] == V_FE2) return 198;
    *z = num[varlist[i - 1]][j - 1];
    return 0;
}
static ST_int m_store(ST_int i, ST_int j, ST_double v) {
    if (!pos_ok(i, j) || varlist[i - 1] == V_FE2) return 198;
    num[varlist[i - 1]][j - 1] = v;
    return 0;
}
static ST_int m_sdata(ST_int i, ST_int j, char *s) {
    if (!pos_ok(i, j) || varlist[i - 1] != V_FE2) return 198;
    strcpy(s, fe2[j - 1]);
    return 0;
}
static ST_int m_sstore(ST_int i, ST_int j, char *s) { (void)i; (void)j; (void)s; return 198; }
static ST_boolean m_isstr(ST_int i) { return i >= 1 && i <= nvarlist && varlist[i - 1] == V_FE2; }
static ST_boolean m_isstrl(ST_int i) { (void)i; return 0; }
static ST_boolean m_isbinary(ST_int i, ST_int j) { (void)i; (void)j; return 0; }
static ST_int m_sdatalen(ST_int i, ST_int j) { return pos_ok(i, j) ? (ST_int)strlen(fe2[j - 1]) : -198; }
static ST_int m_strldata(ST_int i, ST_int j, char *s, ST_int len) { (void)i; (void)j; (void)s; (void)len; return -198; }
static ST_boolean m_selobs(ST_int j) { return j >= in1 && j <= in2; }
static ST_int m_nobs1(void) { return in1; }
static ST_int m_nobs2(void) { return in2; }
static ST_int m_nobs(void) { return NOBS; }
static ST_int m_nvar(void) { return NVAR; }
static ST_int m_nvars(void) { return nvarlist; }
static ST_boolean m_ismissing(ST_double z) { return z >= MISSVAL; }

static ST_plugin table;

static void init_table(void) {
    memset(&table, 0, sizeof table);
    table.spoutsml = m_spoutsml;
    table.spoutnosml = m_spoutsml;
    table.spoutflush = m_spoutflush;
    table.pollstd = m_poll;
    table.pollnow = m_poll;
    table.macresave = m_macresave;
    table.scalsave = m_scalsave;
    table.matel = m_matel;
    table.safematel = m_matel;
    table.matstore = m_matstore;
    table.safematstore = m_matstore;
    table.rowsof = m_rowsof;
    table.colsof = m_colsof;
    table.vdata = m_vdata;
    table.safevdata = m_vdata;
    table.store = m_store;
    table.safestore = m_store;
    table.sdata = m_sdata;
    table.sstore = m_sstore;
    table.isstr = m_isstr;
    table.isstrl = m_isstrl;
    table.isbinary = m_isbinary;
    table.sdatalen = m_sdatalen;
    table.strldata = m_strldata;
    table.selobs = m_selobs;
    table.nobs1 = m_nobs1;
    table.nobs2 = m_nobs2;
    table.nobs = m_nobs;
    table.nvar = m_nvar;
    table.nvars = m_nvars;
    table.missval = MISSVAL;
    table.ismissing = m_ismissing;
    table.stopflag = &stopflag;
    table.spouterr = m_spouterr;
    table.major = 3;
    table.minor = 0;
}

/* ---------------------------------------------------------------- plugin loading */

typedef ST_retcode (*pginit_fn)(ST_plugin *);
typedef ST_retcode (*call_fn)(int, char **);
static call_fn stata_call_p;

static int load_plugin(const char *path) {
    void *pg, *sc;
#ifdef _WIN32
    HMODULE lib = LoadLibraryA(path);
    if (!lib) { fprintf(stderr, "cannot load %s (error %lu)\n", path, GetLastError()); return 0; }
    pg = (void *)GetProcAddress(lib, "pginit");
    sc = (void *)GetProcAddress(lib, "stata_call");
#else
    void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "cannot load %s: %s\n", path, dlerror()); return 0; }
    pg = dlsym(lib, "pginit");
    sc = dlsym(lib, "stata_call");
#endif
    if (!pg || !sc) { fprintf(stderr, "plugin lacks pginit/stata_call\n"); return 0; }
    ST_retcode ver = ((pginit_fn)pg)(&table);
    if (ver != SD_PLUGINVER) { fprintf(stderr, "pginit returned %d, expected %d\n", ver, SD_PLUGINVER); return 0; }
    stata_call_p = (call_fn)sc;
    return 1;
}

/* Calls the plugin: vars = dataset variables (terminated by -1), args = argv (NULL-terminated). */
static int call(const int *vars, int first, int last, const char **args) {
    nvarlist = 0;
    while (vars[nvarlist] >= 0) { varlist[nvarlist] = vars[nvarlist]; nvarlist++; }
    in1 = first;
    in2 = last;
    int argc = 0;
    char *argv[32];
    while (args[argc]) { argv[argc] = (char *)args[argc]; argc++; }
    last_err[0] = 0;
    return stata_call_p(argc, argv);
}

/* ---------------------------------------------------------------- expected values */

static char *expected_json = NULL;

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    buf[n] = 0;
    fclose(f);
    return buf;
}

/* Parses "key": <number|null|[numbers]> into out (cap entries). Returns the count, -1 for null,
   -2 when the key is missing. */
static int json_get(const char *key, double *out, int cap) {
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\":", key);
    const char *p = strstr(expected_json, pat);
    if (!p) return -2;
    p += strlen(pat);
    while (*p == ' ') p++;
    if (!strncmp(p, "null", 4)) return -1;
    if (*p != '[') { out[0] = strtod(p, NULL); return 1; }
    p++;
    int n = 0;
    while (*p && *p != ']') {
        char *end;
        double v = strtod(p, &end);
        if (end == p) break;
        if (n < cap) out[n] = v;
        n++;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    return n;
}

static int fails = 0, pendings = 0;

static void pass(const char *what) { printf("PASS    %s\n", what); }
static void fail(const char *what, const char *why) { printf("FAIL    %s: %s\n", what, why); fails++; }
static void pending(const char *what, const char *why) { printf("PENDING %s: %s", what, why); pendings++; }

/* No libm calls: the V-4 build line does not link -lm. */
static double absd(double x) { return x < 0 ? -x : x; }

static int close_enough(double a, double b, double rtol) {
    double s = absd(a) > absd(b) ? absd(a) : absd(b);
    if (s < 1.0) s = 1.0;
    return absd(a - b) <= rtol * s;
}

/* A stub in fereg-core reports r(9999) with one of these phrases. */
static int is_core_stub(int rc) {
    return rc == 9999 && (strstr(last_err, "not implemented yet") || strstr(last_err, "not built yet"));
}

/* ---------------------------------------------------------------- tests */

static void test_version(void) {
    const char *args[] = {"version", NULL};
    int vars[] = {-1};
    int rc = call(vars, 1, NOBS, args);
    const char *v = macro_get("__rf_version");
    if (rc || !v || !*v) { fail("version", "no _rf_version local"); return; }
    printf("        plugin version %s\n", v);
    pass("version");
}

static void test_selftest(void) {
    const char *args[] = {"selftest", "serialio=0", "threads=4", "write=4", NULL};
    int vars[] = {V_Y, V_X1, V_X2, V_SPARE, -1};
    int rc = call(vars, 101, 900, args);
    double chk = 0.0, got, nobs;
    for (int i = 100; i < 900; i++) {
        /* volatile products: compilers may otherwise fuse "chk += a*b" into an FMA
           (clang does on arm64), which rounds differently from the plugin's sum */
        volatile double t1 = 1.0 * num[V_Y][i];
        volatile double t2 = 2.0 * num[V_X1][i];
        volatile double t3 = 3.0 * num[V_X2][i];
        chk += t1;
        chk += t2;
        chk += t3;
        /* V_SPARE was missing before the call: contributes 0 */
    }
    if (rc) { fail("selftest", "nonzero rc"); return; }
    if (!scalar_get("__rf_chk", &got) || !scalar_get("__rf_nobs", &nobs)) { fail("selftest", "missing scalars"); return; }
    if (got != chk || nobs != 800) { fail("selftest", "checksum or nobs mismatch"); return; }
    for (int i = 0; i < NOBS; i++) {
        double want = (i >= 100 && i < 900) ? (double)(i + 1) : MISSVAL;
        if (num[V_SPARE][i] != want) { fail("selftest", "write= stored wrong values"); return; }
        num[V_SPARE][i] = MISSVAL;
    }
    pass("selftest (parallel read, if/in, write=)");
}

static void test_break(void) {
    const char *args[] = {"selftest", "spin=5", NULL};
    int vars[] = {V_Y, -1};
    stopflag = 1;
    int rc = call(vars, 1, NOBS, args);
    stopflag = 0;
    if (rc != 1) { fail("break", "Break did not return r(1)"); return; }
    pass("break (stopflag -> r(1))");
}

static void test_errors(void) {
    int vars[] = {V_FE1, V_TOUSE, -1};
    const char *a1[] = {"prepare", "nfac=1", "factype=n", "absorb=1|1|-|0", "bogus=1", NULL};
    const char *a2[] = {"ols", "token=1", "nx=0", NULL};
    const char *a3[] = {"nosuchcmd", NULL};
    if (call(vars, 1, NOBS, a1) != 198) { fail("errors", "unknown key not r(198)"); return; }
    if (call(vars, 1, NOBS, a2) != 198 || !strstr(last_err, "stale")) { fail("errors", "stale token not r(198)"); return; }
    if (call(vars, 1, NOBS, a3) != 198) { fail("errors", "unknown cmd not r(198)"); return; }
    pass("error paths (unknown key, stale token, unknown cmd)");
}

/* prepare with the given absorber; returns 1 if the session was built (token in the macro registry). */
static int test_prepare(const char *absorber) {
    char what[64], kabs[32];
    snprintf(what, sizeof what, "prepare (absorber=%s)", absorber);
    snprintf(kabs, sizeof kabs, "absorber=%s", absorber);
    for (int i = 0; i < NOBS; i++) num[V_TOUSE][i] = 1.0;
    const char *args[] = {"prepare", "nfac=2", "factype=ns", "absorb=1|1|-|0;2|1|-|0", "vce=unadjusted",
                          "wtype=none", "singletons=1", "threads=2", kabs, NULL};
    int vars[] = {V_FE1, V_FE2, V_TOUSE, -1};
    int rc = call(vars, 1, NOBS, args);
    if (is_core_stub(rc)) { pending(what, last_err); return 0; }
    if (rc) { fail(what, last_err); return 0; }
    double N, ns, nf, eN, ens, dropped[NOBS];
    const char *tok = macro_get("__rf_token");
    if (!tok || !*tok) { fail(what, "no _rf_token local"); return 0; }
    if (!scalar_get("__rf_N", &N) || !scalar_get("__rf_num_singletons", &ns) || !scalar_get("__rf_N_full", &nf)) {
        fail(what, "missing scalars");
        return 0;
    }
    json_get("N", &eN, 1);
    json_get("num_singletons", &ens, 1);
    int nd = json_get("dropped", dropped, NOBS);
    if (N != eN || ns != ens || nf != NOBS) { fail(what, "N / num_singletons / N_full mismatch"); return 0; }
    int k = 0;
    for (int i = 0; i < NOBS; i++) {
        int is_dropped = k < nd && (int)dropped[k] == i;
        if (is_dropped) k++;
        if (num[V_TOUSE][i] != (is_dropped ? 0.0 : 1.0)) { fail(what, "touse not zeroed exactly for dropped obs"); return 0; }
    }
    printf("        N=%.0f singletons=%.0f token=%s\n", N, ns, tok);
    pass(what);
    return 1;
}

static void test_ols(const char *absorber) {
    char what[64], tok[96];
    snprintf(what, sizeof what, "ols (absorber=%s)", absorber);
    snprintf(tok, sizeof tok, "token=%s", macro_get("__rf_token"));
    double eb[3], eV[9], eres[NOBS], dropped[NOBS];
    int nb = json_get("b", eb, 3), nV = json_get("V", eV, 9), nr = json_get("resid", eres, NOBS);
    int nd = json_get("dropped", dropped, NOBS);
    mat_define("b", 1, 3);
    mat_define("V", 3, 3);
    for (int i = 0; i < NOBS; i++) num[V_RESID][i] = MISSVAL;
    const char *args[] = {"ols", tok, "nx=2", "nout=1", "resid=1", "b=b", "V=V", "tol=1e-14", "threads=2", NULL};
    int vars[] = {V_Y, V_X1, V_X2, V_TOUSE, V_RESID, -1};
    int rc = call(vars, 1, NOBS, args);
    if (is_core_stub(rc)) {
        if (nb >= 0) fail(what, "core is a stub but mock_expected.json has results (stale file?)");
        else pending(what, last_err);
        return;
    }
    if (rc) { fail(what, last_err); return; }
    if (nb != 3 || nV != 9) { fail(what, "plugin ran OLS but mock_expected.json has no b/V; rerun the example"); return; }
    int kb = mat_find("b"), kV = mat_find("V");
    for (int e = 0; e < 3; e++) if (!close_enough(mats[kb].v[e], eb[e], 1e-10)) { fail(what, "b differs"); return; }
    for (int e = 0; e < 9; e++) if (!close_enough(mats[kV].v[e], eV[e], 1e-10)) { fail(what, "V differs"); return; }
    int k = 0, r = 0;
    for (int i = 0; i < NOBS; i++) {
        int is_dropped = k < nd && (int)dropped[k] == i;
        if (is_dropped) { k++; if (num[V_RESID][i] != MISSVAL) { fail(what, "resid written for a dropped obs"); return; } continue; }
        if (r >= nr || !close_enough(num[V_RESID][i], eres[r], 1e-9)) { fail(what, "resid differs"); return; }
        r++;
    }
    printf("        b = [%.10g %.10g %.10g]\n", mats[kb].v[0], mats[kb].v[1], mats[kb].v[2]);
    pass(what);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <plugin> [expected.json]\n", argv[0]); return 2; }
    const char *json = argc > 2 ? argv[2] : "target/mock_expected.json";
    expected_json = read_file(json);
    if (!expected_json) { fprintf(stderr, "cannot read %s (run the mock_expected example)\n", json); return 2; }
    make_data();
    init_table();
    if (!load_plugin(argv[1])) return 1;

    test_version();
    test_selftest();
    test_break();
    test_errors();
    if (test_prepare("dense")) test_ols("dense");
    if (test_prepare("fast")) test_ols("fast");

    printf("mockhost: %d failed, %d pending\n", fails, pendings);
    return fails ? 1 : 0;
}
