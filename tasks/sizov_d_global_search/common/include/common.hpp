#pragma once

#include <functional>
#include <string>

#include "task/include/task.hpp"

namespace sizov_d_global_search {

using Function = std::function<double(double)>;

struct Problem {
  Function func;
  double left = 0.0;
  double right = 0.0;
  double accuracy = 1e-4;
  double reliability = 3.0;
  int max_iterations = 300;
  bool enable_stable_heuristic = true;
  bool enable_point_limit = true;
};

struct Solution {
  double argmin = 0.0;
  double value = 0.0;
  int iterations = 0;
  bool converged = false;
};

struct TestCase {
  std::string name;
  Problem problem;
};

static constexpr double kDefaultAccuracy = 1e-4;
static constexpr double kDefaultReliability = 3.0;
static constexpr int kDefaultMaxIterations = 300;

using InType = Problem;
using OutType = Solution;
using TestType = TestCase;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sizov_d_global_search
