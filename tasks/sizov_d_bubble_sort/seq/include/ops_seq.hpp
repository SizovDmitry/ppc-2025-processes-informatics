#pragma once

#include <vector>

#include "sizov_d_bubble_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sizov_d_bubble_sort {

class SizovDBubbleSortSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit SizovDBubbleSortSEQ(const InType &in);

 private:
  std::vector<int> data_;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sizov_d_bubble_sort
