#include "morozova_s_connected_components/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
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
  const std::size_t cols = input.front().size();
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

  if (rows_ == 0) {
    cols_ = 0;
    GetOutput().clear();
    return true;
  }

  cols_ = static_cast<int>(input.front().size());
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
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rows_ == 0 || cols_ == 0) {
    return true;
  }

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
        ++local_label;
      }
    }
  }

  if (rank == 0) {
    for (int proc = 1; proc < size; ++proc) {
      int ps = proc * rows_per_proc + std::min(proc, remainder);
      int pe = ps + rows_per_proc + (proc < remainder ? 1 : 0);
      int pr = pe - ps;

      std::vector<int> buf(pr * cols_);
      MPI_Recv(buf.data(), pr * cols_, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int i = 0; i < pr; ++i) {
        for (int j = 0; j < cols_; ++j) {
          GetOutput()[ps + i][j] = buf[i * cols_ + j];
        }
      }
    }

    std::unordered_map<int, int> parent;

    for (int proc = 1; proc < size; ++proc) {
      int br = proc * rows_per_proc + std::min(proc, remainder);
      if (br <= 0 || br >= rows_) {
        continue;
      }

      for (int j = 0; j < cols_; ++j) {
        for (int dj = -1; dj <= 1; ++dj) {
          int nj = j + dj;
          if (nj < 0 || nj >= cols_) {
            continue;
          }
          if (grid_[br - 1][j] == 1 && grid_[br][nj] == 1) {
            int a = GetOutput()[br - 1][j];
            int b = GetOutput()[br][nj];
            if (a != b) {
              parent[std::max(a, b)] = std::min(a, b);
            }
          }
        }
      }
    }

    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        int v = GetOutput()[i][j];
        while (parent.count(v)) {
          v = parent[v];
        }
        GetOutput()[i][j] = v;
      }
    }

    std::unordered_map<int, int> remap;
    int next = 1;
    for (auto &row : GetOutput()) {
      for (int &v : row) {
        if (v > 0) {
          if (!remap.count(v)) {
            remap[v] = next++;
          }
          v = remap[v];
        }
      }
    }

    std::vector<int> full(rows_ * cols_);
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        full[i * cols_ + j] = GetOutput()[i][j];
      }
    }

    for (int proc = 1; proc < size; ++proc) {
      MPI_Send(full.data(), rows_ * cols_, MPI_INT, proc, 1, MPI_COMM_WORLD);
    }
  } else {
    int lr = end_row - start_row;
    std::vector<int> send(lr * cols_);
    for (int i = 0; i < lr; ++i) {
      for (int j = 0; j < cols_; ++j) {
        send[i * cols_ + j] = GetOutput()[start_row + i][j];
      }
    }

    MPI_Send(send.data(), lr * cols_, MPI_INT, 0, 0, MPI_COMM_WORLD);

    std::vector<int> recv(rows_ * cols_);
    MPI_Recv(recv.data(), rows_ * cols_, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        GetOutput()[i][j] = recv[i * cols_ + j];
      }
    }
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
