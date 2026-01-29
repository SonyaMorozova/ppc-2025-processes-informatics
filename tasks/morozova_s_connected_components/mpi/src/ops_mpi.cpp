#include "morozova_s_connected_components/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <queue>
#include <utility>
#include <vector>

#include "morozova_s_connected_components/common/include/common.hpp"
#include "util/include/util.hpp"

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
  size_t cols = input[0].size();
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
  grid_ = GetInput();
  rows_ = static_cast<int>(grid_.size());
  cols_ = static_cast<int>(grid_[0].size());
  visited_.resize(rows_, std::vector<bool>(cols_, false));
  GetOutput() = std::vector<std::vector<int>>(rows_, std::vector<int>(cols_, 0));
  return true;
}

std::vector<std::pair<int, int>> MorozovaSConnectedComponentsMPI::GetNeighbors(int row, int col) {
  std::vector<std::pair<int, int>> neighbors;
  const int dr[] = {-1, -1, -1, 0, 0, 1, 1, 1};
  const int dc[] = {-1, 0, 1, -1, 1, -1, 0, 1};
  for (int i = 0; i < 8; ++i) {
    int new_row = row + dr[i];
    int new_col = col + dc[i];
    if (new_row >= 0 && new_row < rows_ && new_col >= 0 && new_col < cols_ && grid_[new_row][new_col] == 1) {
      neighbors.emplace_back(new_row, new_col);
    }
  }
  return neighbors;
}

void MorozovaSConnectedComponentsMPI::BFSLabeling(int start_row, int start_col, int label) {
  std::queue<std::pair<int, int>> q;
  q.emplace(start_row, start_col);
  visited_[start_row][start_col] = true;
  GetOutput()[start_row][start_col] = label;
  while (!q.empty()) {
    auto [row, col] = q.front();
    q.pop();
    auto neighbors = GetNeighbors(row, col);
    for (const auto &neighbor : neighbors) {
      if (!visited_[neighbor.first][neighbor.second]) {
        visited_[neighbor.first][neighbor.second] = true;
        GetOutput()[neighbor.first][neighbor.second] = label;
        q.push(neighbor);
      }
    }
  }
}

bool MorozovaSConnectedComponentsMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int rows_per_process = rows_ / size;
  int remainder = rows_ % size;
  int start_row = rank * rows_per_process + std::min(rank, remainder);
  int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);
  int local_component_count = 0;
  std::vector<int> local_labels_start;
  for (int i = start_row; i < end_row; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (grid_[i][j] == 1 && !visited_[i][j]) {
        local_component_count++;
      }
    }
  }
  int global_offset = 0;
  MPI_Scan(&local_component_count, &global_offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  int local_label_start = global_offset - local_component_count + 1;
  int current_label = local_label_start;
  for (int i = start_row; i < end_row; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (grid_[i][j] == 1 && !visited_[i][j]) {
        BFSLabeling(i, j, current_label);
        current_label++;
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    for (int p = 1; p < size; ++p) {
      int p_rows_per_process = rows_ / size;
      int p_remainder = rows_ % size;
      int p_start_row = p * p_rows_per_process + std::min(p, p_remainder);
      int p_end_row = p_start_row + p_rows_per_process + (p < p_remainder ? 1 : 0);
      for (int i = p_start_row; i < p_end_row; ++i) {
        MPI_Recv(GetOutput()[i].data(), cols_, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  } else {
    for (int i = start_row; i < end_row; ++i) {
      MPI_Send(GetOutput()[i].data(), cols_, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  }
  std::vector<int> upper_row(cols_, 0);
  std::vector<int> lower_row(cols_, 0);
  if (rank > 0) {
    MPI_Send(GetOutput()[start_row].data(), cols_, MPI_INT, rank - 1, 1, MPI_COMM_WORLD);
    if (start_row > 0) {
      MPI_Recv(upper_row.data(), cols_, MPI_INT, rank - 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  if (rank < size - 1 && end_row < rows_) {
    MPI_Recv(lower_row.data(), cols_, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (end_row - 1 >= start_row) {
      MPI_Send(GetOutput()[end_row - 1].data(), cols_, MPI_INT, rank + 1, 2, MPI_COMM_WORLD);
    }
  }
  return true;
}

bool MorozovaSConnectedComponentsMPI::PostProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int max_label = 0;
  if (rank == 0) {
    for (const auto &row : GetOutput()) {
      for (int label : row) {
        if (label > max_label) {
          max_label = label;
        }
      }
    }
    auto &output = GetOutput();
    output.push_back({max_label});
  }
  MPI_Bcast(&max_label, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return true;
}

}  // namespace morozova_s_connected_components
