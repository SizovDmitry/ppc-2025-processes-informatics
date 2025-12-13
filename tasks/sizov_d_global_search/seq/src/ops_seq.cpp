#include "sizov_d_global_search/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"

namespace sizov_d_global_search {

SizovDGlobalSearchSEQ::SizovDGlobalSearchSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool SizovDGlobalSearchSEQ::ValidationImpl() {
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

bool SizovDGlobalSearchSEQ::PreProcessingImpl() {
  const auto &p = GetInput();

  x_.clear();
  y_.clear();

  x_.reserve(static_cast<std::size_t>(p.max_iterations) + 8U);
  y_.reserve(static_cast<std::size_t>(p.max_iterations) + 8U);

  const double left = p.left;
  const double right = p.right;

  const double f_left = p.func(left);
  const double f_right = p.func(right);

  if (!std::isfinite(f_left) || !std::isfinite(f_right)) {
    return false;
  }

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

  return true;
}

double SizovDGlobalSearchSEQ::EstimateM(double reliability) const {
  constexpr double kMinSlope = 1e-2;

  const std::size_t n = x_.size();
  if (n < 2U) {
    return reliability * kMinSlope;
  }

  double max_slope = 0.0;
  for (std::size_t i = 1; i < n; ++i) {
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
      max_slope = std::max(max_slope, slope);
    }
  }

  return reliability * std::max(max_slope, kMinSlope);
}

double SizovDGlobalSearchSEQ::Characteristic(std::size_t idx, double m) const {
  const double x_right = x_[idx];
  const double x_left = x_[idx - 1U];
  const double y_right = y_[idx];
  const double y_left = y_[idx - 1U];

  const double dx = x_right - x_left;
  const double df = y_right - y_left;

  return (m * dx) + ((df * df) / (m * dx)) - (2.0 * (y_right + y_left));
}

double SizovDGlobalSearchSEQ::NewPoint(std::size_t idx, double m) const {
  const double x_right = x_[idx];
  const double x_left = x_[idx - 1U];
  const double y_right = y_[idx];
  const double y_left = y_[idx - 1U];

  const double mid = 0.5 * (x_left + x_right);
  const double shift = (y_right - y_left) / (2.0 * m);

  double x_new = mid - shift;
  if (x_new <= x_left || x_new >= x_right) {
    x_new = mid;
  }
  return x_new;
}

std::pair<std::size_t, double> SizovDGlobalSearchSEQ::FindBestInterval(double m) const {
  double best_char = -std::numeric_limits<double>::infinity();
  std::size_t best_idx = 1U;

  for (std::size_t i = 1; i < x_.size(); ++i) {
    const double c = Characteristic(i, m);
    if (c > best_char) {
      best_char = c;
      best_idx = i;
    }
  }

  return {best_idx, best_char};
}

std::size_t SizovDGlobalSearchSEQ::InsertSample(double x_new, double y_new) {
  auto pos = std::ranges::lower_bound(x_, x_new);
  const std::size_t idx = static_cast<std::size_t>(pos - x_.begin());

  x_.insert(pos, x_new);
  y_.insert(y_.begin() + static_cast<std::ptrdiff_t>(idx), y_new);
  return idx;
}

bool SizovDGlobalSearchSEQ::RunImpl() {
  const auto &p = GetInput();

  if (x_.size() < 2U) {
    return false;
  }

  double m = EstimateM(p.reliability);

  for (int iter = 0; iter < p.max_iterations; ++iter) {
    iterations_ = iter + 1;

    if ((iter % 10) == 0) {
      m = EstimateM(p.reliability);
    }

    if (x_.size() < 2U) {
      converged_ = false;
      break;
    }

    const auto [best_idx, _best_char] = FindBestInterval(m);
    const double left = x_[best_idx - 1U];
    const double right = x_[best_idx];
    const double width = right - left;

    if (width <= p.accuracy) {
      converged_ = true;
      break;
    }

    const double x_new = NewPoint(best_idx, m);
    const double y_new = p.func(x_new);
    if (!std::isfinite(y_new)) {
      continue;
    }

    (void)InsertSample(x_new, y_new);

    if (y_new < best_y_) {
      best_y_ = y_new;
      best_x_ = x_new;
    }
  }

  GetOutput() = Solution{
      .argmin = best_x_,
      .value = best_y_,
      .iterations = iterations_,
      .converged = converged_,
  };
  return true;
}

bool SizovDGlobalSearchSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_global_search
