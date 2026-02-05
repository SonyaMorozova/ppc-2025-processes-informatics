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

namespace {
constexpr int kLabelOffset = 1000000;

constexpr std::array<std::pair<int, int>, 8> kShifts = {
    {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}};
}  // namespace

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
  const size_t cols = input.front().size();
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
  rows_ = static_cast<int>(GetInput().size());

  if (rows_ == 0) {
    cols_ = 0;
    GetOutput().clear();
    return true;
  }

  cols_ = static_cast<int>(GetInput().front().size());
  GetOutput().assign(rows_, std::vector<int>(cols_, 0));
  return true;
}

void MorozovaSConnectedComponentsMPI::InitMPI() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
  rows_per_proc_ = rows_ / size_;
  remainder_ = rows_ % size_;
}

std::pair<int, int> MorozovaSConnectedComponentsMPI::ComputeRowRange() const {
  const int start = (rank_ * rows_per_proc_) + std::min(rank_, remainder_);
  const int end = start + rows_per_proc_ + (rank_ < remainder_ ? 1 : 0);
  return {start, end};
}

std::vector<std::pair<int, int>> MorozovaSConnectedComponentsMPI::GetNeighbors(int row, int col) {
  std::vector<std::pair<int, int>> neighbors;
  const auto &input = GetInput();

  for (const auto &[dr, dc] : kShifts) {
    const int nr = row + dr;
    const int nc = col + dc;
    if (nr >= 0 && nr < rows_ && nc >= 0 && nc < cols_ && input[nr][nc] == 1) {
      neighbors.emplace_back(nr, nc);
    }
  }
  return neighbors;
}

void MorozovaSConnectedComponentsMPI::FloodFill(int row, int col, int label) {
  auto &output = GetOutput();

  std::vector<std::vector<bool>> visited(rows_, std::vector<bool>(cols_, false));
  std::queue<std::pair<int, int>> q;
  q.emplace(row, col);
  visited[row][col] = true;
  output[row][col] = label;

  while (!q.empty()) {
    const auto [r, c] = q.front();
    q.pop();
    for (const auto &[nr, nc] : GetNeighbors(r, c)) {
      if (!visited[nr][nc]) {
        visited[nr][nc] = true;
        output[nr][nc] = label;
        q.emplace(nr, nc);
      }
    }
  }
}

void MorozovaSConnectedComponentsMPI::ComputeLocalComponents(int start_row, int end_row, int base_label) {
  const auto &input = GetInput();
  auto &output = GetOutput();

  std::vector<std::vector<bool>> visited(rows_, std::vector<bool>(cols_, false));
  int local_label = 1;

  for (int i = start_row; i < end_row; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (input[i][j] == 1 && output[i][j] == 0) {
        FloodFill(i, j, base_label + local_label);
        ++local_label;
      }
    }
  }
}

void MorozovaSConnectedComponentsMPI::GatherLocalResults() {
  auto &output = GetOutput();

  for (int proc = 1; proc < size_; ++proc) {
    const int ps = (proc * rows_per_proc_) + std::min(proc, remainder_);
    const int pe = ps + rows_per_proc_ + (proc < remainder_ ? 1 : 0);
    const int pr = pe - ps;

    std::vector<int> buf(static_cast<size_t>(pr) * static_cast<size_t>(cols_));
    MPI_Recv(buf.data(), static_cast<int>(buf.size()), MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = 0; i < pr; ++i) {
      for (int j = 0; j < cols_; ++j) {
        const size_t idx = (static_cast<size_t>(i) * static_cast<size_t>(cols_)) + static_cast<size_t>(j);
        output[ps + i][j] = buf[idx];
      }
    }
  }
}

void MorozovaSConnectedComponentsMPI::ProcessBoundaryCell(int proc, int j, int dj,
                                                          std::unordered_map<int, int> &parent) {
  const auto &input = GetInput();
  const auto &output = GetOutput();

  const int br = (proc * rows_per_proc_) + std::min(proc, remainder_);
  if (br <= 0 || br >= rows_) {
    return;
  }

  const int nj = j + dj;
  if (nj < 0 || nj >= cols_) {
    return;
  }

  if (input[br - 1][j] == 1 && input[br][nj] == 1) {
    const int a = output[br - 1][j];
    const int b = output[br][nj];
    if (a != b) {
      parent[std::max(a, b)] = std::min(a, b);
    }
  }
}

int MorozovaSConnectedComponentsMPI::FindRoot(std::unordered_map<int, int> &parent, int v) {
  while (parent.contains(v)) {
    v = parent[v];
  }
  return v;
}

void MorozovaSConnectedComponentsMPI::MergeBoundaries() {
  auto &output = GetOutput();
  std::unordered_map<int, int> parent;

  for (int proc = 1; proc < size_; ++proc) {
    const int br = (proc * rows_per_proc_) + std::min(proc, remainder_);
    if (br <= 0 || br >= rows_) {
      continue;
    }

    for (int j = 0; j < cols_; ++j) {
      ProcessBoundaryCell(proc, j, -1, parent);
      ProcessBoundaryCell(proc, j, 0, parent);
      ProcessBoundaryCell(proc, j, 1, parent);
    }
  }

  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      int &v = output[i][j];
      if (v > 0) {
        v = FindRoot(parent, v);
      }
    }
  }
}

void MorozovaSConnectedComponentsMPI::NormalizeLabels() {
  auto &output = GetOutput();
  std::unordered_map<int, int> remap;
  int next = 1;
  for (auto &row : output) {
    for (int &v : row) {
      if (v > 0) {
        if (!remap.contains(v)) {
          remap[v] = next++;
        }
        v = remap[v];
      }
    }
  }
}

void MorozovaSConnectedComponentsMPI::BroadcastResult() {
  auto &output = GetOutput();
  std::vector<int> flat(static_cast<size_t>(rows_) * static_cast<size_t>(cols_));
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      const size_t idx = (static_cast<size_t>(i) * static_cast<size_t>(cols_)) + static_cast<size_t>(j);
      flat[idx] = output[i][j];
    }
  }

  for (int proc = 1; proc < size_; ++proc) {
    MPI_Send(flat.data(), static_cast<int>(flat.size()), MPI_INT, proc, 1, MPI_COMM_WORLD);
  }
}

void MorozovaSConnectedComponentsMPI::SendLocalResult(int start_row, int end_row) {
  const auto &output = GetOutput();
  const int lr = end_row - start_row;
  std::vector<int> send(static_cast<size_t>(lr) * static_cast<size_t>(cols_));
  for (int i = 0; i < lr; ++i) {
    for (int j = 0; j < cols_; ++j) {
      const size_t idx = (static_cast<size_t>(i) * static_cast<size_t>(cols_)) + static_cast<size_t>(j);
      send[idx] = output[start_row + i][j];
    }
  }
  MPI_Send(send.data(), static_cast<int>(send.size()), MPI_INT, 0, 0, MPI_COMM_WORLD);
}

void MorozovaSConnectedComponentsMPI::ReceiveFinalResult() {
  auto &output = GetOutput();
  std::vector<int> recv(static_cast<size_t>(rows_) * static_cast<size_t>(cols_));
  MPI_Recv(recv.data(), static_cast<int>(recv.size()), MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      const size_t idx = (static_cast<size_t>(i) * static_cast<size_t>(cols_)) + static_cast<size_t>(j);
      output[i][j] = recv[idx];
    }
  }
}

bool MorozovaSConnectedComponentsMPI::RunImpl() {
  InitMPI();

  if (rows_ == 0 || cols_ == 0) {
    return true;
  }

  const auto [start, end] = ComputeRowRange();
  ComputeLocalComponents(start, end, rank_ * kLabelOffset);

  if (rank_ == 0) {
    GatherLocalResults();
    MergeBoundaries();
    NormalizeLabels();
    BroadcastResult();
  } else {
    SendLocalResult(start, end);
    ReceiveFinalResult();
  }

  return true;
}

bool MorozovaSConnectedComponentsMPI::PostProcessingImpl() {
  const auto &output = GetOutput();
  int max_label = 0;
  for (const auto &row : output) {
    for (int v : row) {
      max_label = std::max(max_label, v);
    }
  }

  auto &output_ref = GetOutput();
  output_ref.push_back({max_label});
  return true;
}

}  // namespace morozova_s_connected_components
