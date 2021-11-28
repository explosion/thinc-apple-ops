<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# thinc-apple-ops

Make [spaCy](https://spacy.io) and [Thinc](https://thinc.ai) **up to 8 &times; faster**
on macOS by calling into Apple's native libraries.

## ⏳ Install

Make sure you have [Xcode](https://developer.apple.com/xcode/) installed and
then install with `pip`:

```bash
pip install thinc-apple-ops
```

## 🏫 Motivation

Matrix multiplication is one of the primary operations in machine learning.
Since matrix multiplication is computationally expensive, using a fast matrix
multiplication implementation can speed up training and prediction
significantly.

Most linear algebra libraries provide matrix multiplication in the form of the
standardized
[BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) `gemm`
functions. The work behind scences is done by a set of matrix multiplication
kernels that are meticulously tuned for specific architectures. Matrix
multiplication kernels use architecture-specific
[SIMD](https://en.wikipedia.org/wiki/SIMD) instructions for data-level parallism
and can take factors such as cache sizes and intstruction latency into account.
[Thinc](https://github.com/explosion/thinc) uses the
[BLIS](https://github.com/flame/blis) linear algebra library, which provides
optimized matrix multiplication kernels for most x86_64 and some ARM CPUs.

Recent [Apple Silicon](https://en.wikipedia.org/wiki/Apple_silicon) CPUs, such
as the [M-series](https://en.wikipedia.org/wiki/Apple_silicon#M_series) used in
Macs, differ from traditional x86_64 and ARM CPUs in that they have a separate
matrix co-processor(s) called AMX. Since AMX is not well-documented, it is
unclear how many AMX units Apple M CPUs have. It is certain that the (single)
performance cluster of the M1 has an AMX unit and there is [empirical
evidence](https://twitter.com/danieldekok/status/1454383754512945155?s=20) that
both performance clusters of the M1 Pro/Max have an AMX unit.


Even though AMX units use a set of [undocumented
instructions](https://gist.github.com/dougallj/7a75a3be1ec69ca550e7c36dc75e0d6f),
the units can be used through Apple's
[Accelerate](https://developer.apple.com/documentation/accelerate) linear
algebra library. Since Accelerate implements the BLAS interface, it can be used
as a replacement of the BLIS library that is used by Thinc. This is where the
`thinc-apple-ops` package comes in. `thinc-apple-ops` extends the default Thinc
ops, so that `gemm` matrix multiplication from Accelerate is used in place of
the BLIS implementation of `gemm`. As a result, matrix multiplication in Thinc
is performed on the fast AMX unit(s).

## ⏱ Benchmarks

Using `thinc-apple-ops` leads to large speedups in prediction and training on
Apple Silicon Macs, as shown by the benchmarks below.

### Prediction

This first benchmark compares prediction speed of the `de_core_news_lg` spaCy
model between the M1 with and without `thinc-apple-ops`. Results for an Intel
Mac Mini and AMD Ryzen 5900X are also provided for comparison. Results are in
words per second. In this prediction benchmark, using `thinc-apple-ops` improves
performance by **4.3** times.

| *CPU*                      | *BLIS* | *thinc-apple-ops* | *Package power (Watt)* |
| -------------------------- | -----: | ----------------: | ---------------------: |
| Mac Mini (M1)              |   6492 |             27676 |                      5 |
| MacBook Air Core i5 2020   |   9790 |             10983 |                      9 |
| Mac Mini Core i7 Late 2018 |  16364 |             14858 |                     31 |
| AMD Ryzen 5900X            |  22568 |               N/A |                     52 |

### Training

In the second benchmark, we compare the training speed of the `de_core_news_lg`
spaCy model (without NER). The results are in training iterations per second.
Using `thinc-apple-ops` improves training time by **3.0** times.

| *CPU*                      | *BLIS* | *thinc-apple-ops* | *Package power (Watt)* |
| -------------------------- | -----: | ----------------: | ---------------------: |
| Mac Mini M1 2020           |   3.34 |             10.07 |                      5 |
| MacBook Air Core i5 2020   |   3.10 |              3.27 |                     10 |
| Mac Mini Core i7 Late 2018 |   4.71 |              4.93 |                     32 |
| AMD Ryzen 5900X            |   6.53 |               N/A |                     53 |
