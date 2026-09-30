This directory can build an optional cuSolverDx benchmark target when the
MathDx SDK is available under third_party/mathdx or via
-DKERMAC_CUSOLVERDX_MATHDX_ROOT=<path>.

The benchmark target name is:
  kermac_cusolverdx_posv_bench

It is intentionally standalone and does not depend on the kermac library.
