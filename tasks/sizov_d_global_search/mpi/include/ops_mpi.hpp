#pragma once

#include <cstddef>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sizov_d_global_search {

class SizovDGlobalSearchMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit SizovDGlobalSearchMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  struct IntervalChar {
    double characteristic = 0.0;
    int index = -1;
  };

  struct InsertMsg {
    double x_new = 0.0;
    double y_new = 0.0;
    int idx = -1;
  };

  static void GetChunk(std::size_t intervals, int rank, int size, std::size_t &begin, std::size_t &end);

  [[nodiscard]] double EstimateM(double reliability, int rank, int size) const;
  [[nodiscard]] double Characteristic(std::size_t i, double m) const;
  [[nodiscard]] double NewPoint(std::size_t i, double m) const;

  [[nodiscard]] IntervalChar ComputeLocalBestInterval(double m, int rank, int size) const;
  static int ReduceBestIntervalIndex(const IntervalChar &local, int n, int size);

  [[nodiscard]] bool CheckStopByAccuracy(const Problem &p, int best_idx, int rank) const;

  void BroadcastState(int rank);
  void BroadcastNewPoint(int best_idx, double m, const Problem &p, int rank);
  void BroadcastResult(int rank);

  static void BroadcastInsertMsg(InsertMsg &msg, int rank);

  std::vector<double> x_;
  std::vector<double> y_;

  double best_x_ = 0.0;
  double best_y_ = 0.0;
  int iterations_ = 0;
  bool converged_ = false;
};

}  // namespace sizov_d_global_search
