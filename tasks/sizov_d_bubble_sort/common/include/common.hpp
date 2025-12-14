#pragma once

#include <string>
#include <vector>

#include "task/include/task.hpp"

namespace sizov_d_bubble_sort {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sizov_d_bubble_sort
