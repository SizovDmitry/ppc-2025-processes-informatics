#include "sizov_d_string_mismatch_count/seq/include/ops_seq.hpp"

#include <cstddef>

#include "sizov_d_string_mismatch_count/common/include/common.hpp"

namespace sizov_d_string_mismatch_count {

SizovDStringMismatchCountSEQ::SizovDStringMismatchCountSEQ(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput() = 0;
}

bool SizovDStringMismatchCountSEQ::ValidationImpl() {
  const auto &input = GetInput();
  const auto &a = std::get<0>(input);
  const auto &b = std::get<1>(input);
  return !a.empty() && a.size() == b.size();
}

bool SizovDStringMismatchCountSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  str_a_ = std::get<0>(input);
  str_b_ = std::get<1>(input);
  result_ = 0;
  return true;
}

bool SizovDStringMismatchCountSEQ::RunImpl() {
  result_ = 0;
  for (std::size_t i = 0; i < str_a_.size(); ++i) {
    if (str_a_[i] != str_b_[i]) {
      ++result_;
    }
  }
  return true;
}

bool SizovDStringMismatchCountSEQ::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace sizov_d_string_mismatch_count
