#include "morozova_s_connected_components/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

#include "morozova_s_connected_components/common/include/common.hpp"

namespace morozova_s_connected_components {

MorozovaSConnectedComponentsMPI::MorozovaSConnectedComponentsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool MorozovaSConnectedComponentsMPI::ValidationImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    return false;
  }
  const std::size_t cols = input[0].size();
  for (const auto &row : input) {
    if (row.size() != cols) {
      return false;
    }
    for (int v : row) {
      if (v != 0 && v != 1) {
        return false;
      }
    }
  }
  return true;
}

bool MorozovaSConnectedComponentsMPI::PreProcessingImpl() {
  const auto &input = GetInput();
  rows_ = static_cast<int>(input.size());
  cols_ = static_cast<int>(input[0].size());
  grid_ = input;
  visited_.assign(rows_, std::vector<bool>(cols_, false));
  GetOutput().assign(rows_, std::vector<int>(cols_, 0));
  return true;
}

std::vector<std::pair<int, int>> MorozovaSConnectedComponentsMPI::GetNeighbors(int row, int col) const {
  std::vector<std::pair<int, int>> neighbors;
  const std::array<int, 8> dr = {-1, -1, -1, 0, 0, 1, 1, 1};
  const std::array<int, 8> dc = {-1, 0, 1, -1, 1, -1, 0, 1};
  for (std::size_t k = 0; k < dr.size(); ++k) {
    const int nr = row + dr[k];
    const int nc = col + dc[k];
    if (nr >= 0 && nr < rows_ && nc >= 0 && nc < cols_ && grid_[nr][nc] == 1) {
      neighbors.emplace_back(nr, nc);
    }
  }
  return neighbors;
}

void MorozovaSConnectedComponentsMPI::FloodFill(int row, int col, int label) {
  std::queue<std::pair<int, int>> q;
  q.emplace(row, col);
  visited_[row][col] = true;
  GetOutput()[row][col] = label;
  while (!q.empty()) {
    const auto [r, c] = q.front();
    q.pop();
    for (const auto &[nr, nc] : GetNeighbors(r, c)) {
      if (!visited_[nr][nc]) {
        visited_[nr][nc] = true;
        GetOutput()[nr][nc] = label;
        q.emplace(nr, nc);
      }
    }
  }
}

void MorozovaSConnectedComponentsMPI::RunLabeling() {
  int label = 1;
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (grid_[i][j] == 1 && !visited_[i][j]) {
        FloodFill(i, j, label);
        ++label;
      }
    }
  }
}

bool MorozovaSConnectedComponentsMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int rows_per_proc = rows_ / size;
  int remainder = rows_ % size;
  int start_row = rank * rows_per_proc + std::min(rank, remainder);
  int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);
  int local_label = 1;
  std::vector<int> local_labels;
  for (int i = start_row; i < end_row; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (grid_[i][j] == 1 && !visited_[i][j]) {
        FloodFill(i, j, local_label);
        local_labels.push_back(local_label);
        ++local_label;
      }
    }
  }
  if (rank == 0) {
    for (int proc = 1; proc < size; ++proc) {
      int proc_start_row = proc * rows_per_proc + std::min(proc, remainder);
      int proc_end_row = proc_start_row + rows_per_proc + (proc < remainder ? 1 : 0);
      int proc_rows = proc_end_row - proc_start_row;
      std::vector<int> recv_buffer(proc_rows * cols_);
      MPI_Recv(recv_buffer.data(), proc_rows * cols_, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (int i = 0; i < proc_rows; ++i) {
        for (int j = 0; j < cols_; ++j) {
          GetOutput()[proc_start_row + i][j] = recv_buffer[i * cols_ + j];
        }
      }
    }
  } else {
    int local_rows = end_row - start_row;
    std::vector<int> send_buffer(local_rows * cols_);
    for (int i = 0; i < local_rows; ++i) {
      for (int j = 0; j < cols_; ++j) {
        send_buffer[i * cols_ + j] = GetOutput()[start_row + i][j];
      }
    }
    MPI_Send(send_buffer.data(), local_rows * cols_, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
  if (rank == 0) {
    for (int proc = 1; proc < size; ++proc) {
      MPI_Send(GetOutput().data()->data(), rows_ * cols_, MPI_INT, proc, 1, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(GetOutput().data()->data(), rows_ * cols_, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  return true;
}

bool MorozovaSConnectedComponentsMPI::PostProcessingImpl() {
  int max_label = 0;
  for (const auto &row : GetOutput()) {
    for (int v : row) {
      max_label = std::max(max_label, v);
    }
  }
  GetOutput().push_back({max_label});
  return true;
}

}  // namespace morozova_s_connected_components
