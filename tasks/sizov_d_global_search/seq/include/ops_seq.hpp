#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sizov_d_global_search {

class SizovDGlobalSearchSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit SizovDGlobalSearchSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] double EstimateM(double reliability) const;
  [[nodiscard]] double Characteristic(std::size_t idx, double m) const;
  [[nodiscard]] double NewPoint(std::size_t idx, double m) const;

  [[nodiscard]] std::pair<std::size_t, double> FindBestInterval(double m) const;
  [[nodiscard]] std::size_t InsertSample(double x_new, double y_new);

  std::vector<double> x_;
  std::vector<double> y_;

  double best_x_ = 0.0;
  double best_y_ = 0.0;

  int iterations_ = 0;
  bool converged_ = false;
};

}  // namespace sizov_d_global_search
