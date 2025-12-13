#include "sizov_d_global_search/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"

namespace sizov_d_global_search {

SizovDGlobalSearchMPI::SizovDGlobalSearchMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool SizovDGlobalSearchMPI::ValidationImpl() {
  const auto &p = GetInput();
  if (!p.func) {
    return false;
  }
  if (!(p.left < p.right)) {
    return false;
  }
  if (!(p.accuracy > 0.0)) {
    return false;
  }
  if (!(p.reliability > 0.0)) {
    return false;
  }
  if (p.max_iterations <= 0) {
    return false;
  }
  return true;
}

bool SizovDGlobalSearchMPI::PreProcessingImpl() {
  const auto &p = GetInput();

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int ok = 1;

  if (rank == 0) {
    x_.clear();
    y_.clear();

    x_.reserve(static_cast<std::size_t>(p.max_iterations) + 8U);
    y_.reserve(static_cast<std::size_t>(p.max_iterations) + 8U);

    const double left = p.left;
    const double right = p.right;

    const double f_left = p.func(left);
    const double f_right = p.func(right);

    if (!std::isfinite(f_left) || !std::isfinite(f_right)) {
      ok = 0;
    } else {
      x_.push_back(left);
      x_.push_back(right);
      y_.push_back(f_left);
      y_.push_back(f_right);

      if (f_left <= f_right) {
        best_x_ = left;
        best_y_ = f_left;
      } else {
        best_x_ = right;
        best_y_ = f_right;
      }

      iterations_ = 0;
      converged_ = false;
    }
  }

  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (ok == 0) {
    return false;
  }

  BroadcastState(rank);
  return true;
}

void SizovDGlobalSearchMPI::GetChunk(std::size_t intervals, int rank, int size, std::size_t &begin, std::size_t &end) {
  if (intervals == 0U || size <= 0) {
    begin = 0;
    end = 0;
    return;
  }

  const auto s = static_cast<std::size_t>(size);
  const auto r = static_cast<std::size_t>(rank);

  const std::size_t base = intervals / s;
  const std::size_t rem = intervals % s;

  begin = (r * base) + std::min(r, rem);
  end = begin + base + ((r < rem) ? 1U : 0U);
  end = std::min(end, intervals);
}

double SizovDGlobalSearchMPI::EstimateM(double reliability, int rank, int size) const {
  constexpr double kMinSlope = 1e-2;

  const std::size_t n = x_.size();
  if (n < 2U) {
    return reliability * kMinSlope;
  }

  const std::size_t intervals = n - 1U;

  std::size_t begin = 0;
  std::size_t end = 0;
  GetChunk(intervals, rank, size, begin, end);

  double local_max = 0.0;
  for (std::size_t k = begin; k < end; ++k) {
    const std::size_t i = k + 1U;

    const double dx = x_[i] - x_[i - 1U];
    if (!(dx > 0.0)) {
      continue;
    }

    const double y1 = y_[i];
    const double y2 = y_[i - 1U];
    if (!std::isfinite(y1) || !std::isfinite(y2)) {
      continue;
    }

    const double slope = std::abs(y1 - y2) / dx;
    if (std::isfinite(slope)) {
      local_max = std::max(local_max, slope);
    }
  }

  double global_max = local_max;
  if (size > 1) {
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  }

  return reliability * std::max(global_max, kMinSlope);
}

double SizovDGlobalSearchMPI::Characteristic(std::size_t i, double m) const {
  const double x_right = x_[i];
  const double x_left = x_[i - 1U];
  const double y_right = y_[i];
  const double y_left = y_[i - 1U];

  const double dx = x_right - x_left;
  const double df = y_right - y_left;

  return (m * dx) + ((df * df) / (m * dx)) - (2.0 * (y_right + y_left));
}

double SizovDGlobalSearchMPI::NewPoint(std::size_t i, double m) const {
  const double x_right = x_[i];
  const double x_left = x_[i - 1U];
  const double y_right = y_[i];
  const double y_left = y_[i - 1U];

  const double mid = 0.5 * (x_left + x_right);
  const double shift = (y_right - y_left) / (2.0 * m);

  double x_new = mid - shift;
  if (x_new <= x_left || x_new >= x_right) {
    x_new = mid;
  }
  return x_new;
}

SizovDGlobalSearchMPI::IntervalChar SizovDGlobalSearchMPI::ComputeLocalBestInterval(double m, int rank,
                                                                                    int size) const {
  IntervalChar res{};
  res.characteristic = -std::numeric_limits<double>::infinity();
  res.index = -1;

  const std::size_t n = x_.size();
  if (n < 2U) {
    return res;
  }

  const std::size_t intervals = n - 1U;

  std::size_t begin = 0;
  std::size_t end = 0;
  GetChunk(intervals, rank, size, begin, end);

  for (std::size_t k = begin; k < end; ++k) {
    const std::size_t i = k + 1U;
    const double c = Characteristic(i, m);
    if (c > res.characteristic) {
      res.characteristic = c;
      res.index = static_cast<int>(i);
    }
  }

  return res;
}

int SizovDGlobalSearchMPI::ReduceBestIntervalIndex(const IntervalChar &local, int n, int size) {
  IntervalChar global = local;

  if (size > 1) {
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
  }

  if (global.index < 1 || global.index >= n) {
    return -1;
  }
  return global.index;
}

bool SizovDGlobalSearchMPI::CheckStopByAccuracy(const Problem &p, int best_idx, int rank) const {
  int flag = 0;

  if (rank == 0) {
    const auto i = static_cast<std::size_t>(best_idx);
    const double left = x_[i - 1U];
    const double right = x_[i];
    const double width = right - left;

    if (width <= p.accuracy) {
      flag = 1;
    }
  }

  MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return flag != 0;
}

void SizovDGlobalSearchMPI::BroadcastState(int rank) {
  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(x_.size());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  x_.resize(static_cast<std::size_t>(n));
  y_.resize(static_cast<std::size_t>(n));

  if (n > 0) {
    MPI_Bcast(x_.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y_.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

void SizovDGlobalSearchMPI::BroadcastInsertMsg(InsertMsg &msg, int rank) {
  MPI_Bcast(&msg.idx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&msg.x_new, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&msg.y_new, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  (void)rank;
}

void SizovDGlobalSearchMPI::BroadcastNewPoint(int best_idx, double m, const Problem &p, int rank) {
  InsertMsg msg{};

  if (rank == 0) {
    const auto i = static_cast<std::size_t>(best_idx);

    const double x_new = NewPoint(i, m);
    const double y_new = p.func(x_new);

    if (std::isfinite(y_new)) {
      auto pos = std::ranges::lower_bound(x_, x_new);
      const std::size_t idx = static_cast<std::size_t>(pos - x_.begin());

      msg.x_new = x_new;
      msg.y_new = y_new;
      msg.idx = static_cast<int>(idx);

      x_.insert(pos, x_new);
      y_.insert(y_.begin() + static_cast<std::ptrdiff_t>(idx), y_new);

      if (y_new < best_y_) {
        best_x_ = x_new;
        best_y_ = y_new;
      }
    } else {
      msg.idx = -1;
    }
  }

  BroadcastInsertMsg(msg, rank);

  if (rank != 0 && msg.idx >= 0) {
    const auto idx = static_cast<std::size_t>(msg.idx);
    if (idx <= x_.size()) {
      x_.insert(x_.begin() + static_cast<std::ptrdiff_t>(idx), msg.x_new);
      y_.insert(y_.begin() + static_cast<std::ptrdiff_t>(idx), msg.y_new);
    }
  }
}

void SizovDGlobalSearchMPI::BroadcastResult(int rank) {
  std::array<double, 2> result{};
  std::array<int, 2> meta{};

  if (rank == 0) {
    result[0] = best_x_;
    result[1] = best_y_;
    meta[0] = iterations_;
    meta[1] = converged_ ? 1 : 0;
  }

  MPI_Bcast(result.data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(meta.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);

  best_x_ = result[0];
  best_y_ = result[1];
  iterations_ = meta[0];
  converged_ = (meta[1] != 0);
}

bool SizovDGlobalSearchMPI::RunImpl() {
  const auto &p = GetInput();

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (x_.size() < 2U) {
    return false;
  }

  double m = EstimateM(p.reliability, rank, size);

  for (int iter = 0; iter < p.max_iterations; ++iter) {
    iterations_ = iter + 1;

    if ((iter % 10) == 0) {
      m = EstimateM(p.reliability, rank, size);
    }

    const int n = static_cast<int>(x_.size());
    if (n < 2) {
      converged_ = false;
      break;
    }

    const IntervalChar local = ComputeLocalBestInterval(m, rank, size);
    const int best_idx = ReduceBestIntervalIndex(local, n, size);
    if (best_idx < 1) {
      converged_ = false;
      break;
    }

    if (CheckStopByAccuracy(p, best_idx, rank)) {
      if (rank == 0) {
        converged_ = true;
      }
      break;
    }

    BroadcastNewPoint(best_idx, m, p, rank);
  }

  BroadcastResult(rank);

  GetOutput() = Solution{
      .argmin = best_x_,
      .value = best_y_,
      .iterations = iterations_,
      .converged = converged_,
  };
  return true;
}

bool SizovDGlobalSearchMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_global_search
