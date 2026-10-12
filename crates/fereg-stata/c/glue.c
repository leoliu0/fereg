/*
    glue.c — the only code in fereg that touches the Stata plugin interface (SPI 3.0).

    Compiled by build.rs with -DSYSTEM=OPUNIX|STWIN32 -DSD_FASTMODE and hidden visibility:
    nothing here is exported from the plugin (only the Rust `pginit` / `stata_call` are).
    Observation and variable indices are 1-based, as in Stata.
*/

#include <math.h>
#include <string.h>
#include "stplugin.h"

#if defined(__GNUC__) || defined(__clang__)
#define RF_HIDDEN __attribute__((visibility("hidden")))
#else
#define RF_HIDDEN
#endif

RF_HIDDEN ST_plugin *_stata_ = 0;

RF_HIDDEN void rf_set_plugin(void *p) { _stata_ = (ST_plugin *)p; }

RF_HIDDEN int rf_nobs(void) { return SF_nobs(); }
RF_HIDDEN int rf_nvars(void) { return SF_nvars(); }
RF_HIDDEN int rf_in1(void) { return SF_in1(); }
RF_HIDDEN int rf_in2(void) { return SF_in2(); }
RF_HIDDEN int rf_ifobs(int obs) { return SF_ifobs(obs) ? 1 : 0; }

/* Row-major read: out[k*nv + j] = value of vars[j] at obs[k] (observation-major access).
   Stata missing values (. and .a-.z) are all >= SV_missval; they are returned as NaN and
   counted in *nmiss. */
RF_HIDDEN int rf_read_rows(const int *vars, int nv, const int *obs, int n, double *out, long long *nmiss) {
    ST_double mv = SV_missval;
    long long miss = 0;
    for (int k = 0; k < n; k++) {
        double *row = out + (size_t)k * (size_t)nv;
        int o = obs[k];
        for (int j = 0; j < nv; j++) {
            ST_double z;
            ST_retcode rc = SF_vdata(vars[j], o, &z);
            if (rc) return rc;
            if (z >= mv) {
                row[j] = NAN;
                miss++;
            } else {
                row[j] = z;
            }
        }
    }
    *nmiss = miss;
    return 0;
}

/* Number of observations in o1..o2 whose touse value (variable tv) is not 0, or -rc. With
   tv = 0 every observation counts (nothing is read). */
RF_HIDDEN int rf_count_sample(int tv, int o1, int o2) {
    if (tv == 0) return o2 >= o1 ? o2 - o1 + 1 : 0;
    int c = 0;
    for (int o = o1; o <= o2; o++) {
        ST_double t;
        ST_retcode rc = SF_vdata(tv, o, &t);
        if (rc) return -rc;
        c += (t != 0);
    }
    return c;
}

/* Sample scan for cmd=prepare over o1..o2: for the c-th observation whose touse value
   (variable tv; tv = 0: every observation, touse is not read) is not 0, writes the observation
   to obs_out[c] and the value of vars[j] to cols[j][c * strides[j]] (NaN for missing, counted
   in *nmiss); every output has room for cap observations. Returns the number of observations
   written, -rc on a Stata error, or -9999 when more than cap are found. */
RF_HIDDEN int rf_read_sample(int tv, const int *vars, int nv, int o1, int o2, int cap, int *obs_out, double *const *cols,
                             const int *strides, long long *nmiss) {
    ST_double mv = SV_missval;
    long long miss = 0;
    int c = 0;
    for (int o = o1; o <= o2; o++) {
        ST_retcode rc;
        if (tv) {
            ST_double t;
            rc = SF_vdata(tv, o, &t);
            if (rc) return -rc;
            if (t == 0) continue;
        }
        if (c == cap) return -9999;
        obs_out[c] = o;
        for (int j = 0; j < nv; j++) {
            ST_double z;
            rc = SF_vdata(vars[j], o, &z);
            if (rc) return -rc;
            miss += (z >= mv);
            cols[j][(size_t)c * (size_t)strides[j]] = (z >= mv) ? NAN : z;
        }
        c++;
    }
    *nmiss = miss;
    return c;
}

/* NaN is stored as Stata missing. Uses the checked `safestore` even under SD_FASTMODE: the
   unchecked `store` indexes variables by dataset position, not by varlist position (verified
   in Stata 19.5: store(2, ...) wrote into the dataset's second variable). */
RF_HIDDEN int rf_store_num(int var, const int *obs, int n, const double *v, int stride) {
    ST_double mv = SV_missval;
    for (int k = 0; k < n; k++) {
        double z = v[(size_t)k * (size_t)stride];
        ST_retcode rc = _stata_->safestore(var, obs[k], isnan(z) ? mv : z);
        if (rc) return rc;
    }
    return 0;
}

RF_HIDDEN int rf_is_str(int var) { return SF_var_is_string(var) ? 1 : 0; }
RF_HIDDEN int rf_is_strl(int var) { return SF_var_is_strl(var) ? 1 : 0; }

/* str# value into buf (at least 2046 bytes). Returns the length, or -rc on error. */
RF_HIDDEN int rf_read_str(int var, int obs, char *buf) {
    buf[0] = 0;
    ST_retcode rc = SF_sdata(var, obs, buf);
    if (rc) return -rc;
    return (int)strlen(buf);
}

/* strL: byte length of the value (excluding any terminator), or -rc. */
RF_HIDDEN int rf_strl_len(int var, int obs) {
    if (SF_var_is_binary(var, obs)) return -198;
    return SF_sdatalen(var, obs);
}

/* strL value into buf of capacity cap (len + 1). Returns the rc of SF_strldata (bytes copied or <0). */
RF_HIDDEN int rf_read_strl(int var, int obs, char *buf, int cap) { return SF_strldata(var, obs, buf, cap); }

RF_HIDDEN int rf_scal_save(const char *name, double v) {
    return SF_scal_save((char *)name, isnan(v) ? SV_missval : v);
}

RF_HIDDEN int rf_mat_rows(const char *name) { return SF_row((char *)name); }
RF_HIDDEN int rf_mat_cols(const char *name) { return SF_col((char *)name); }

/* Stores a row-major rows x cols block into an existing matrix of the same dimensions. */
RF_HIDDEN int rf_mat_store_all(const char *name, int rows, int cols, const double *v) {
    ST_double mv = SV_missval;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double z = v[(size_t)i * (size_t)cols + (size_t)j];
            ST_retcode rc = _stata_->safematstore((char *)name, i + 1, j + 1, isnan(z) ? mv : z);
            if (rc) return rc;
        }
    }
    return 0;
}

/* name "_x" sets local x of the calling program. */
RF_HIDDEN int rf_macro_save(const char *name, const char *value) {
    return SF_macro_save((char *)name, (char *)value);
}

RF_HIDDEN void rf_display(const char *s) {
    SF_display((char *)s);
    _stata_->spoutflush();
}

RF_HIDDEN void rf_error(const char *s) { SF_error((char *)s); }

/* Lets Stata process pending events, then reports whether Break was pressed. */
RF_HIDDEN int rf_poll_break(void) {
    SF_poll();
    return SW_stopflag ? 1 : 0;
}
