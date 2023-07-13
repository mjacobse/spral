#include "spral.h"

#include <benchmark/benchmark.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

struct SymmetricMatrixView {
   const std::int64_t* ptr;
   const int* row;
   const double* val;
   int n;
   bool posdef;
};

struct SsidsAnalysisKeepDeleter {
   void operator()(void* akeep) const noexcept {
      spral_ssids_free_akeep(&akeep);
   }
};
using SsidsAnalysisKeepUnique = std::unique_ptr<void, SsidsAnalysisKeepDeleter>;

struct SsidsFactorKeepDeleter {
   void operator()(void* fkeep) const noexcept {
      spral_ssids_free_fkeep(&fkeep);
   }
};
using SsidsFactorKeepUnique = std::unique_ptr<void, SsidsFactorKeepDeleter>;

void benchmarkFactorize(benchmark::State &state,
                        const SymmetricMatrixView &matrix) {
   const auto options = [&matrix] {
      struct spral_ssids_options options;
      spral_ssids_default_options(&options);
      options.print_level = -1;
      options.array_base = 1;
      return options;
   }();
   struct spral_ssids_inform inform;

   auto akeep = [&matrix, &options, &inform] {
      const bool check = true;
      void* akeep = nullptr;
      spral_ssids_analyse(check, matrix.n, nullptr, matrix.ptr, matrix.row,
            matrix.val, &akeep, &options, &inform);
      return SsidsAnalysisKeepUnique(akeep);
   }();
   if (inform.flag != 0) {
      throw std::runtime_error("Error during analysis. Error code: " +
                               std::to_string(inform.flag));
   }
   state.counters["num_factor_analyse"] = inform.num_factor;
   state.counters["num_flops_analyse"] = inform.num_flops;
   state.counters["maxfront_analyse"] = inform.maxfront;
   state.counters["num_supernodes"] = inform.num_sup;

   auto fkeep = [&matrix, &akeep, &options, &inform] {
      void* fkeep = nullptr;
      spral_ssids_factor(matrix.posdef, nullptr, nullptr, matrix.val, nullptr,
            akeep.get(), &fkeep, &options, &inform);
      return SsidsFactorKeepUnique(fkeep);
   }();
   if (inform.flag != 0 && inform.flag != 7) {
      throw std::runtime_error("Error during factorization. Error code: " +
                               std::to_string(inform.flag));
   }
   state.counters["num_factor"] = inform.num_factor;
   state.counters["num_flops"] = inform.num_flops;
   state.counters["maxfront"] = inform.maxfront;
   state.counters["num_delay"] = inform.num_delay;
   state.counters["num_two"] = inform.num_two;

   for (auto _ : state) {
      void* fkeep_new = fkeep.get();
      spral_ssids_factor(matrix.posdef, nullptr, nullptr, matrix.val, nullptr,
            akeep.get(), &fkeep_new, &options, &inform);
      assert(fkeep_new == fkeep.get());
      assert(inform.flag == 0 ||  inform.flag == 7);
   }
}

struct SpralFileHandleDeleter {
   void operator()(void* handle) const noexcept {
      spral_rb_free_handle(&handle);
   }
};
using SpralFileHandleUnique = std::unique_ptr<void, SpralFileHandleDeleter>;

class SymmetricMatrixFile {
public:
   explicit SymmetricMatrixFile(const char* filepath) {
      const auto options = [] {
         struct spral_rb_read_options options;
         spral_rb_default_read_options(&options);
         options.array_base = 1;
         options.add_diagonal = true;
         return options;
      }();

      void* handle;
      spral_matrix_type matrix_type;
      int m;
      std::int64_t* ptr;
      int* row;
      double* val = nullptr;
      const int error_code = spral_rb_read(
            filepath, &handle, &matrix_type, &m, &m_view.n, &ptr, &row, &val,
            &options, m_title, nullptr, nullptr);
      // important to do this first, so that it is free'd in case of exiting
      // with failure.
      m_handle = SpralFileHandleUnique(handle);
      if (error_code != 0) {
         throw std::runtime_error("Error reading matrix file. Error code: " +
                                  std::to_string(error_code));
      }

      if (matrix_type == SPRAL_MATRIX_REAL_SYM_PSDEF ||
          matrix_type == SPRAL_MATRIX_CPLX_HERM_PSDEF) {
         m_view.posdef = true;
      } else if (matrix_type == SPRAL_MATRIX_REAL_SYM_INDEF ||
                 matrix_type == SPRAL_MATRIX_CPLX_HERM_INDEF) {
         m_view.posdef = false;
      } else {
         throw std::runtime_error("Matrix is not symmetric/hermitian");
      }

      if (m != m_view.n) {
         throw std::runtime_error("Matrix is not square");
      }

      m_view.ptr = ptr;
      m_view.row = row;
      m_view.val = val;
      if (!m_view.val) {
         throw std::runtime_error("Got no matrix values");
      }
   }

   SymmetricMatrixView view() const { return m_view; }

private:
   SpralFileHandleUnique m_handle;
   char m_title[73];  // Maximum length 72 character + '\0'
   SymmetricMatrixView m_view;
};

}  // anonymous namespace


int main(int argc, char** argv) {
   if (argc < 2) {
      std::puts("Usage: ssids_benchmark MATRIX.rb");
      return EXIT_FAILURE;
   }

   const char* filepath_matrix = argv[1];
   SymmetricMatrixFile matrix(filepath_matrix);

   benchmark::RegisterBenchmark("factorize", benchmarkFactorize, matrix.view())
      ->Unit(benchmark::kMillisecond)
      ->MeasureProcessCPUTime()
      ->UseRealTime();

   benchmark::Initialize(&argc, argv);
   benchmark::RunSpecifiedBenchmarks();
}
