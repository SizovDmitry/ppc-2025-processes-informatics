#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "sizov_d_bubble_sort/common/include/common.hpp"
#include "sizov_d_bubble_sort/mpi/include/ops_mpi.hpp"
#include "sizov_d_bubble_sort/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sizov_d_bubble_sort {

class SizovDRunFuncTestsBubbleSort : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return ppc::util::test::SanitizeToken(test_param);
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string file_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_sizov_d_bubble_sort, param + ".txt");

    std::ifstream file(file_path);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file: " + file_path);
    }

    std::string unsorted_line;
    std::string sorted_line;

    std::getline(file, unsorted_line);
    std::getline(file, sorted_line);
    file.close();

    TrimString(unsorted_line);
    TrimString(sorted_line);

    if (unsorted_line.empty() || sorted_line.empty()) {
      throw std::runtime_error("Input lines cannot be empty in " + file_path);
    }

    std::istringstream unsorted_stream(unsorted_line);
    std::istringstream sorted_stream(sorted_line);

    std::vector<int> unsorted_vec;
    std::vector<int> expected_vec;

    int value = 0;
    while (unsorted_stream >> value) {
      unsorted_vec.push_back(value);
    }
    while (sorted_stream >> value) {
      expected_vec.push_back(value);
    }

    if (unsorted_vec.size() != expected_vec.size()) {
      throw std::runtime_error("Input and expected vectors must have equal size in " + file_path);
    }

    input_data_ = unsorted_vec;
    expected_result_ = expected_vec;
    is_valid_ = true;
  }

  InType GetTestInputData() override {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    if (!is_valid_) {
      return true;
    }
    return output_data == expected_result_;
  }

 private:
  static void TrimString(std::string &s) {
    const auto is_edge_escape = [](unsigned char c) { return c == '\r' || c == '\n' || c == '\t'; };

    std::size_t left = 0;
    while (left < s.size() && is_edge_escape(static_cast<unsigned char>(s[left]))) {
      ++left;
    }

    std::size_t right = s.size();
    while (right > left && is_edge_escape(static_cast<unsigned char>(s[right - 1]))) {
      --right;
    }

    if (left == 0 && right == s.size()) {
      return;
    }

    s.erase(right);
    s.erase(0, left);
  }

  InType input_data_;
  OutType expected_result_;
  bool is_valid_ = true;
};

namespace {

TEST_P(SizovDRunFuncTestsBubbleSort, CompareStringsFromFile) {
  ExecuteTest(GetParam());
}

std::vector<TestType> GetTestParamsFromData() {
  namespace fs = std::filesystem;

  const fs::path data_dir = ppc::util::GetAbsoluteTaskPath(PPC_ID_sizov_d_bubble_sort, "");
  std::error_code ec;
  std::vector<std::pair<int, TestType>> numbered_tests;

  for (const auto &entry : fs::directory_iterator(data_dir, ec)) {
    if (ec) {
      throw std::runtime_error("Failed to read data directory: " + data_dir.string());
    }
    if (!entry.is_regular_file()) {
      continue;
    }

    const fs::path &path = entry.path();
    if (path.extension() != ".txt") {
      continue;
    }

    const std::string stem = path.stem().string();
    constexpr std::string_view kPrefix = "test";
    if (!stem.starts_with(kPrefix)) {
      continue;
    }

    const std::string number_part = stem.substr(kPrefix.size());
    if (number_part.empty() || !std::ranges::all_of(number_part, [](char c) { return std::isdigit(c) != 0; })) {
      continue;
    }

    numbered_tests.emplace_back(std::stoi(number_part), stem);
  }

  if (ec) {
    throw std::runtime_error("Failed to iterate data directory: " + data_dir.string());
  }

  if (numbered_tests.empty()) {
    throw std::runtime_error("No test data files found in " + data_dir.string());
  }

  std::ranges::sort(numbered_tests, [](const auto &lhs, const auto &rhs) {
    if (lhs.first != rhs.first) {
      return lhs.first < rhs.first;
    }
    return lhs.second < rhs.second;
  });

  std::vector<TestType> params;
  params.reserve(numbered_tests.size());
  for (const auto &[_, name] : numbered_tests) {
    params.push_back(name);
  }

  return params;
}

using FuncParam = ppc::util::FuncTestParam<InType, OutType, TestType>;

std::vector<FuncParam> BuildTestTasks(const std::vector<TestType> &tests) {
  std::vector<FuncParam> tasks;
  tasks.reserve(tests.size() * 2);

  for (const auto &test : tests) {
    tasks.emplace_back(
        ppc::task::TaskGetter<SizovDBubbleSortMPI, InType>,
        std::string(ppc::util::GetNamespace<SizovDBubbleSortMPI>()) + "_" +
            ppc::task::GetStringTaskType(SizovDBubbleSortMPI::GetStaticTypeOfTask(), PPC_SETTINGS_sizov_d_bubble_sort),
        test);
    tasks.emplace_back(
        ppc::task::TaskGetter<SizovDBubbleSortSEQ, InType>,
        std::string(ppc::util::GetNamespace<SizovDBubbleSortSEQ>()) + "_" +
            ppc::task::GetStringTaskType(SizovDBubbleSortSEQ::GetStaticTypeOfTask(), PPC_SETTINGS_sizov_d_bubble_sort),
        test);
  }

  return tasks;
}

const auto kTestParam = GetTestParamsFromData();
const auto kTestTasksList = BuildTestTasks(kTestParam);
const auto kGtestValues = ::testing::ValuesIn(kTestTasksList);

const auto kTestName = SizovDRunFuncTestsBubbleSort::PrintFuncTestName<SizovDRunFuncTestsBubbleSort>;

INSTANTIATE_TEST_SUITE_P(SizovDBubbleSort, SizovDRunFuncTestsBubbleSort, kGtestValues, kTestName);

}  // namespace

}  // namespace sizov_d_bubble_sort
