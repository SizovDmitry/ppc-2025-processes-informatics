#include "sizov_d_bubble_sort/seq/include/ops_seq.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "sizov_d_bubble_sort/common/include/common.hpp"

namespace sizov_d_bubble_sort {

SizovDBubbleSortSEQ::SizovDBubbleSortSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SizovDBubbleSortSEQ::ValidationImpl() {
  const auto &input = GetInput();
  return !input.empty();
}

bool SizovDBubbleSortSEQ::PreProcessingImpl() {
  data_ = GetInput();
  return true;
}

bool SizovDBubbleSortSEQ::RunImpl() {
  const std::size_t n = data_.size();
  if (n <= 1) {
    return true;
  }

  auto pass = [&](std::size_t start) {
    bool swapped = false;
    for (std::size_t i = start; i + 1 < n; i += 2) {
      if (data_[i] > data_[i + 1]) {
        std::swap(data_[i], data_[i + 1]);
        swapped = true;
      }
    }
    return swapped;
  };

  while (true) {
    bool swapped = pass(0);
    swapped = pass(1) || swapped;
    if (!swapped) {
      break;
    }
  }

  return true;
}

bool SizovDBubbleSortSEQ::PostProcessingImpl() {
  GetOutput() = data_;
  return true;
}

}  // namespace sizov_d_bubble_sort
