#include "morozova_s_connected_components/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <queue>
#include <tuple>
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
  std::size_t cols = input[0].size();
  for (const auto &row : input) {
    if (row.size() != cols) {
      return false;
    }
    for (int pixel : row) {
      if (pixel != 0 && pixel != 1) {
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
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
  int rows_per_process = rows_ / size_;
  int remainder = rows_ % size_;
  start_row_ = (rank_ * rows_per_process) + std::min(rank_, remainder);
  end_row_ = start_row_ + rows_per_process + (rank_ < remainder ? 1 : 0);
  grid_ = input;
  visited_.assign(rows_, std::vector<bool>(cols_, false));
  GetOutput() = std::vector<std::vector<int>>(rows_, std::vector<int>(cols_, 0));
  return true;
}

std::pair<int, int> MorozovaSConnectedComponentsMPI::CalculateProcessBounds(int rows, int size, int process_rank) {
  int rows_per_process = rows / size;
  int remainder = rows % size;
  int start = (process_rank * rows_per_process) + std::min(process_rank, remainder);
  int end = start + rows_per_process + (process_rank < remainder ? 1 : 0);
  return {start, end};
}

std::vector<std::pair<int, int>> MorozovaSConnectedComponentsMPI::GetNeighbors(int row, int col) const {
  std::vector<std::pair<int, int>> neighbors;
  const std::array<int, 8> dr = {-1, -1, -1, 0, 0, 1, 1, 1};
  const std::array<int, 8> dc = {-1, 0, 1, -1, 1, -1, 0, 1};
  for (size_t i = 0; i < 8; ++i) {
    int new_row = row + dr[i];
    int new_col = col + dc[i];
    if (new_row >= 0 && new_row < rows_ && new_col >= 0 && new_col < cols_ && grid_[new_row][new_col] == 1) {
      neighbors.emplace_back(new_row, new_col);
    }
  }
  return neighbors;
}

void MorozovaSConnectedComponentsMPI::LabelLocalComponents() {
  int label_offset = rank_ * 1000000;
  int current_label = label_offset + 1;
  for (int i = start_row_; i < end_row_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (grid_[i][j] != 1 || visited_[i][j]) {
        continue;
      }
      std::queue<std::pair<int, int>> q;
      q.emplace(i, j);
      visited_[i][j] = true;
      GetOutput()[i][j] = current_label;
      while (!q.empty()) {
        auto [row, col] = q.front();
        q.pop();
        auto neighbors = GetNeighbors(row, col);
        for (const auto &neighbor : neighbors) {
          int nr = neighbor.first;
          int nc = neighbor.second;

          if (nr >= start_row_ && nr < end_row_ && !visited_[nr][nc]) {
            visited_[nr][nc] = true;
            GetOutput()[nr][nc] = current_label;
            q.emplace(nr, nc);
          }
        }
      }
      ++current_label;
    }
  }
}

void MorozovaSConnectedComponentsMPI::MergeGlobalLabels() {
  std::vector<std::vector<bool>> global_visited(rows_, std::vector<bool>(cols_, false));
  std::vector<std::vector<int>> new_labels(rows_, std::vector<int>(cols_, 0));
  int global_label = 1;
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (grid_[i][j] != 1 || global_visited[i][j]) {
        continue;
      }
      std::queue<std::pair<int, int>> q;
      q.emplace(i, j);
      global_visited[i][j] = true;
      new_labels[i][j] = global_label;
      while (!q.empty()) {
        auto [row, col] = q.front();
        q.pop();
        auto neighbors = GetNeighbors(row, col);
        for (const auto &neighbor : neighbors) {
          int nr = neighbor.first;
          int nc = neighbor.second;

          if (!global_visited[nr][nc]) {
            global_visited[nr][nc] = true;
            new_labels[nr][nc] = global_label;
            q.emplace(nr, nc);
          }
        }
      }
      ++global_label;
    }
  }
  GetOutput() = std::move(new_labels);
}

void MorozovaSConnectedComponentsMPI::ProcessBoundaries() {
  if (rank_ == 0) {
    for (int src = 1; src < size_; ++src) {
      auto [src_start, src_end] = CalculateProcessBounds(rows_, size_, src);
      for (int i = src_start; i < src_end; ++i) {
        MPI_Recv(GetOutput()[i].data(), cols_, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  } else {
    for (int i = start_row_; i < end_row_; ++i) {
      MPI_Send(GetOutput()[i].data(), cols_, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank_ == 0) {
    MergeGlobalLabels();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank_ == 0) {
    for (int dest = 1; dest < size_; ++dest) {
      auto [dest_start, dest_end] = CalculateProcessBounds(rows_, size_, dest);
      for (int i = dest_start; i < dest_end; ++i) {
        MPI_Send(GetOutput()[i].data(), cols_, MPI_INT, dest, 1, MPI_COMM_WORLD);
      }
    }
  } else {
    for (int i = start_row_; i < end_row_; ++i) {
      MPI_Recv(GetOutput()[i].data(), cols_, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

bool MorozovaSConnectedComponentsMPI::RunImpl() {
  LabelLocalComponents();
  ProcessBoundaries();
  return true;
}

bool MorozovaSConnectedComponentsMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int max_label = 0;
  if (rank == 0) {
    for (const auto &row : GetOutput()) {
      for (int label : row) {
        max_label = std::max(label, max_label);
      }
    }
    auto &output = GetOutput();
    output.push_back({max_label});
  }
  MPI_Bcast(&max_label, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    auto &output = GetOutput();
    output.push_back({max_label});
  }
  return true;
}

}  // namespace morozova_s_connected_components
