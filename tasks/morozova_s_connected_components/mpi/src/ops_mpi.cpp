#include "morozova_s_connected_components/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <queue>
#include <unordered_map>
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
    return true;
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

bool MorozovaSConnectedComponentsMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int rows_per_proc = rows_ / size;
  int remainder = rows_ % size;
  int start_row = rank * rows_per_proc + std::min(rank, remainder);
  int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);
  int base_label = rank * 1000000;
  int local_label = 1;
  for (int i = start_row; i < end_row; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (grid_[i][j] == 1 && !visited_[i][j]) {
        FloodFill(i, j, base_label + local_label);
        local_label++;
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
    std::unordered_map<int, int> label_map;
    int next_label = 1;
    for (int proc = 1; proc < size; ++proc) {
      int boundary_row = proc * rows_per_proc + std::min(proc, remainder);
      if (boundary_row > 0 && boundary_row < rows_) {
        for (int j = 0; j < cols_; ++j) {
          if (grid_[boundary_row - 1][j] == 1 && grid_[boundary_row][j] == 1) {
            int upper_label = GetOutput()[boundary_row - 1][j];
            int lower_label = GetOutput()[boundary_row][j];
            if (upper_label != lower_label) {
              int root_upper = upper_label;
              while (label_map.find(root_upper) != label_map.end()) {
                root_upper = label_map[root_upper];
              }
              int root_lower = lower_label;
              while (label_map.find(root_lower) != label_map.end()) {
                root_lower = label_map[root_lower];
              }
              if (root_upper != root_lower) {
                int min_label = std::min(root_upper, root_lower);
                int max_label = std::max(root_upper, root_lower);
                label_map[max_label] = min_label;
              }
            }
          }
        }
        for (int j = 0; j < cols_; ++j) {
          if (grid_[boundary_row - 1][j] == 1) {
            for (int dj = -1; dj <= 1; ++dj) {
              int nj = j + dj;
              if (nj >= 0 && nj < cols_ && grid_[boundary_row][nj] == 1) {
                int upper_label = GetOutput()[boundary_row - 1][j];
                int lower_label = GetOutput()[boundary_row][nj];
                if (upper_label != lower_label) {
                  int root_upper = upper_label;
                  while (label_map.find(root_upper) != label_map.end()) {
                    root_upper = label_map[root_upper];
                  }
                  int root_lower = lower_label;
                  while (label_map.find(root_lower) != label_map.end()) {
                    root_lower = label_map[root_lower];
                  }
                  if (root_upper != root_lower) {
                    int min_label = std::min(root_upper, root_lower);
                    int max_label = std::max(root_upper, root_lower);
                    label_map[max_label] = min_label;
                  }
                }
              }
            }
          }
        }
      }
    }
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        if (GetOutput()[i][j] > 0) {
          int label = GetOutput()[i][j];
          while (label_map.find(label) != label_map.end()) {
            label = label_map[label];
          }
          GetOutput()[i][j] = label;
        }
      }
    }
    std::unordered_map<int, int> final_label_map;
    int current_label = 1;
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        if (GetOutput()[i][j] > 0) {
          int old_label = GetOutput()[i][j];
          if (final_label_map.find(old_label) == final_label_map.end()) {
            final_label_map[old_label] = current_label++;
          }
          GetOutput()[i][j] = final_label_map[old_label];
        }
      }
    }
    std::vector<int> send_buffer(rows_ * cols_);
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        send_buffer[i * cols_ + j] = GetOutput()[i][j];
      }
    }

    for (int proc = 1; proc < size; ++proc) {
      MPI_Send(send_buffer.data(), rows_ * cols_, MPI_INT, proc, 1, MPI_COMM_WORLD);
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
    std::vector<int> recv_buffer(rows_ * cols_);
    MPI_Recv(recv_buffer.data(), rows_ * cols_, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        GetOutput()[i][j] = recv_buffer[i * cols_ + j];
      }
    }
  }

  return true;
}

bool MorozovaSConnectedComponentsMPI::PostProcessingImpl() {
  int max_label = 0;
  for (const auto &row : GetOutput()) {
    for (int v : row) {
      if (v > max_label) {
        max_label = v;
      }
    }
  }
  GetOutput().push_back({max_label});
  return true;
}

}  // namespace morozova_s_connected_components
