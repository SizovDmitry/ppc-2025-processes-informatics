#include "sizov_d_bubble_sort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "sizov_d_bubble_sort/common/include/common.hpp"

namespace sizov_d_bubble_sort {

namespace {

void ComputeScatter(int n, int size, std::vector<int> &counts, std::vector<int> &displs) {
  counts.assign(size, 0);
  displs.assign(size, 0);

  const int base = n / size;
  const int rem = n % size;

  int offset = 0;
  for (int i = 0; i < size; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = offset;
    offset += counts[i];
  }
}

void BubblePassLocal(std::vector<int> &local) {
  const int n = static_cast<int>(local.size());
  for (int i = 0; i + 1 < n; ++i) {
    if (local[i] > local[i + 1]) {
      std::swap(local[i], local[i + 1]);
    }
  }
}

void ExchangeBorders(std::vector<int> &local, const std::vector<int> &counts, int rank, int size) {
  if (local.empty()) {
    return;
  }

  if (rank + 1 < size && counts[rank + 1] > 0) {
    int send_val = local.back();
    int recv_val = 0;

    MPI_Sendrecv(&send_val, 1, MPI_INT, rank + 1, 0, &recv_val, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    local.back() = std::min(send_val, recv_val);
  }

  if (rank - 1 >= 0 && counts[rank - 1] > 0) {
    int send_val = local.front();
    int recv_val = 0;

    MPI_Sendrecv(&send_val, 1, MPI_INT, rank - 1, 0, &recv_val, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    local.front() = std::max(send_val, recv_val);
  }
}

}  // namespace

SizovDBubbleSortMPI::SizovDBubbleSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SizovDBubbleSortMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return !GetInput().empty();
  }
  return true;
}

bool SizovDBubbleSortMPI::PreProcessingImpl() {
  data_ = GetInput();
  return true;
}

bool SizovDBubbleSortMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = (rank == 0) ? static_cast<int>(data_.size()) : 0;
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 1) {
    if (rank != 0) {
      data_.assign(n, 0);
    }
    if (n > 0) {
      MPI_Bcast(data_.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
    }
    GetOutput() = data_;
    return true;
  }

  std::vector<int> counts;
  std::vector<int> displs;
  ComputeScatter(n, size, counts, displs);

  const int local_n = counts[rank];
  std::vector<int> local(local_n);

  MPI_Scatterv(rank == 0 ? data_.data() : nullptr, counts.data(), displs.data(), MPI_INT, local.data(), local_n,
               MPI_INT, 0, MPI_COMM_WORLD);

  for (int pass = 0; pass < n; ++pass) {
    BubblePassLocal(local);
    ExchangeBorders(local, counts, rank, size);
  }

  std::vector<int> result;
  if (rank == 0) {
    result.resize(n);
  }

  MPI_Gatherv(local.data(), local_n, MPI_INT, rank == 0 ? result.data() : nullptr, counts.data(), displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    result.resize(n);
  }
  MPI_Bcast(result.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = result;

  return true;
}

bool SizovDBubbleSortMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_bubble_sort
