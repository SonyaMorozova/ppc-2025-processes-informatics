#include "morozova_s_connected_components/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <queue>

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

std::vector<std::pair<int, int>> MorozovaSConnectedComponentsMPI::GetNeighbors(int r, int c) const {
  std::vector<std::pair<int, int>> n;
  constexpr std::array<int, 8> dr{-1, -1, -1, 0, 0, 1, 1, 1};
  constexpr std::array<int, 8> dc{-1, 0, 1, -1, 1, -1, 0, 1};
  for (std::size_t i = 0; i < 8; ++i) {
    const int nr = r + dr[i];
    const int nc = c + dc[i];
    if (nr >= 0 && nr < rows_ && nc >= 0 && nc < cols_ && grid_[nr][nc] == 1) {
      n.emplace_back(nr, nc);
    }
  }
  return n;
}

void MorozovaSConnectedComponentsMPI::LabelComponentsSEQ() {
  int label = 1;
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (grid_[i][j] != 1 || visited_[i][j]) {
        continue;
      }
      std::queue<std::pair<int, int>> q;
      q.emplace(i, j);
      visited_[i][j] = true;
      GetOutput()[i][j] = label;

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
      ++label;
    }
  }
}

bool MorozovaSConnectedComponentsMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    LabelComponentsSEQ();
  }

  MPI_Bcast(GetOutput().data()->data(), rows_ * cols_, MPI_INT, 0, MPI_COMM_WORLD);
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
