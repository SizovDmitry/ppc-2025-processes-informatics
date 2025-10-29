#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "sizov_d_string_mismatch_count/common/include/common.hpp"
#include "sizov_d_string_mismatch_count/mpi/include/ops_mpi.hpp"
#include "sizov_d_string_mismatch_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sizov_d_string_mismatch_count {

class SizovDRunFuncTestsStringMismatchCount : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_sizov_d_string_mismatch_count, "strings.txt");

    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open strings.txt");
    }

    std::string a;
    std::string b;
    std::getline(file, a);
    std::getline(file, b);
    file.close();

    input_data_ = std::make_tuple(a, b);
    is_valid_ = !a.empty() && a.size() == b.size();

    expected_result_ = 0;
    if (is_valid_) {
      for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
          ++expected_result_;
        }
      }
    }
  }

  InType GetTestInputData() override {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!is_valid_ || rank != 0) {
      return true;
    }
    return output_data == expected_result_;
  }

 private:
  InType input_data_;
  OutType expected_result_ = 0;
  bool is_valid_ = true;
};

namespace {

TEST_P(SizovDRunFuncTestsStringMismatchCount, CompareStringsFromFile) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 1> kTestParam = {"default"};

const auto kTaskList = std::tuple_cat(ppc::util::AddFuncTask<SizovDStringMismatchCountMPI, InType>(
                                          kTestParam, PPC_SETTINGS_sizov_d_string_mismatch_count),
                                      ppc::util::AddFuncTask<SizovDStringMismatchCountSEQ, InType>(
                                          kTestParam, PPC_SETTINGS_sizov_d_string_mismatch_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTaskList);
const auto kTestName = SizovDRunFuncTestsStringMismatchCount::PrintFuncTestName<SizovDRunFuncTestsStringMismatchCount>;

INSTANTIATE_TEST_SUITE_P(CompareFromFile, SizovDRunFuncTestsStringMismatchCount, kGtestValues, kTestName);  // NOLINT

}  // namespace
}  // namespace sizov_d_string_mismatch_count
