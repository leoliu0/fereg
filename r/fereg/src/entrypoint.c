#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

/* Called from Rust (Ctx::interrupt) on the R main thread between solver iterations.
 * R_CheckUserInterrupt longjmps on a pending interrupt; running it under R_ToplevelExec keeps
 * that longjmp inside this frame, so it never unwinds through Rust code. The Rust side then
 * returns an "interrupted" error, which R raises as a normal error. */
static void check_interrupt(void *data) {
    (void) data;
    R_CheckUserInterrupt();
}

int fereg_interrupted(void) {
    return R_ToplevelExec(check_interrupt, NULL) == FALSE;
}

void R_init_fereg_extendr(void *dll);

void R_init_fereg(void *dll) {
    R_init_fereg_extendr(dll);
}
