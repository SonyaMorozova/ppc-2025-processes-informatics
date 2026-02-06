#include "morozova_s_broadcast/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>

#include "morozova_s_broadcast/common/include/common.hpp"
#include "task/include/task.hpp"

namespace morozova_s_broadcast {

ppc::task::TypeOfTask MorozovaSBroadcastMPI::GetStaticTypeOfTask() {
  return ppc::task::TypeOfTask::kMPI;
}

MorozovaSBroadcastMPI::MorozovaSBroadcastMPI(const InType &in) : root_(0) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

MorozovaSBroadcastMPI::MorozovaSBroadcastMPI(const InType &in, int root) : root_(root) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MorozovaSBroadcastMPI::ValidationImpl() {
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (root_ < 0 || root_ >= size) {
    return false;
  }
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == root_) {
    return !GetInput().empty();
  }
  return true;
}

bool MorozovaSBroadcastMPI::PreProcessingImpl() {
  return true;
}

bool MorozovaSBroadcastMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int data_size = 0;
  if (rank == root_) {
    data_size = static_cast<int>(GetInput().size());
  }
  CustomBroadcast(&data_size, 1, MPI_INT, root_, MPI_COMM_WORLD);
  GetOutput().resize(static_cast<size_t>(data_size));
  if (data_size > 0) {
    if (rank == root_) {
      std::copy(GetInput().begin(), GetInput().end(), GetOutput().begin());
    }
    CustomBroadcast(GetOutput().data(), data_size, MPI_INT, root_, MPI_COMM_WORLD);
  }
  return true;
}

bool MorozovaSBroadcastMPI::PostProcessingImpl() {
  return true;
}

void MorozovaSBroadcastMPI::CustomBroadcast(void *buffer, int count, MPI_Datatype type, int root, MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  if (size <= 1) {
    return;
  }
  int vrank = (rank - root + size) % size;
  for (int step = 1; step < size; step <<= 1) {
    if (vrank < step) {
      int dst_vrank = vrank + step;
      if (dst_vrank < size) {
        int dst_rank = (dst_vrank + root) % size;
        MPI_Send(buffer, count, type, dst_rank, 0, comm);
      }
    } else if (vrank < 2 * step) {
      int src_vrank = vrank - step;
      int src_rank = (src_vrank + root) % size;
      MPI_Recv(buffer, count, type, src_rank, 0, comm, MPI_STATUS_IGNORE);
    }
  }
}

}  // namespace morozova_s_broadcast
